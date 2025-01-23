import math
import os
import cv2
import numpy as np
import skimage.morphology
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib
import math
from agent.utils.fmm_planner import FMMPlanner
from agent.utils.segmentation import SemanticPredMaskRCNN
from agent.utils.hdbscan_utils import HdbscanCluster
from Grounded_SAM.grounded_sam_demo import vis_semantics
from Grounded_SAM.gsam import GSAM, convert_SAM
from constants import color_palette
import agent.utils.pose as pu
import agent.utils.visualization as vu

from copy import deepcopy
from skimage.color import gray2rgb, rgb2gray
from skimage.feature import canny
from constants import color_palette, object_category
import utils as util
from scipy.spatial import KDTree

# The untrap helper for the bruteforce untrap mode (from Stubborn)
class UnTrapHelper:
    def __init__(self):
        self.total_id = 0
        self.epi_id = 0

    def reset(self, full=False):
        self.total_id += 1
        if full:
            self.total_id = 0
        self.epi_id = 0

    def get_action(self):
        self.epi_id += 1
        if self.epi_id > 30:
            return np.random.randint(2, 4)
        if self.epi_id > 18:
            if self.total_id % 2 == 0:
                return 2
            else:
                return 3
        if self.epi_id  < 3:
            if self.total_id % 2 == 0:
                return 2
            else:
                return 3
        else:
            if self.total_id % 2 == 0:
                return 3
            else:
                return 2

def tensor_to_image():

    return transforms.ToPILImage()

def image_to_tensor():

    return transforms.ToTensor()

class Agent_Helper:
    """
    Class containing functions for motion planning and visualization.
    """

    def __init__(self, args, agent_states, model=None, sde=None, S_sde=None):

        self.args = args

        self.device = torch.device("cuda:" + str(args.sem_gpu_id) if args.cuda else "cpu")

        self.model = model
        self.sde = sde
        self.S_sde=  S_sde
        
        self.object_category = deepcopy(object_category)
        # initialize transform for RGB observations
        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((args.frame_height, args.frame_width),
                               interpolation=Image.NEAREST)])

        # initialize semantic segmentation prediction model
        use_ram = False
        if args.use_gt_seg == 2:
            try:
                self.GSAM = GSAM(self.object_category[:-1], text_threshold=args.text_threshold, device=self.device, use_ram=use_ram)
            except Exception as ex:
                print(f"=====> [ERROR]: {ex}, use RedNet ...")
                self.seg_model = SemanticPredMaskRCNN(args)
                exit()
                
        else:
            self.seg_model = SemanticPredMaskRCNN(args)
        
        
        # initializations for planning:
        self.selem = skimage.morphology.disk(args.col_rad)

        self.obs = None
        self.obs_shape = None
        self.collision_map = None
        self.visited_vis = None
        self.col_width = None
        self.curr_loc = None
        self.last_loc = None
        self.last_action = None
        self.count_forward_actions = None
        self.last_start = None
        self.last_planning_window = None
        self.last_sem_map_vis = None
        self.count_windows = 0
        self.count_masks = 0
        self.rank = 0
        self.episode_no = 0
        self.stg = None
        self.goal_cat = -1
        self.untrap = UnTrapHelper()
        self.agent_states = agent_states

        # We move forward 1 extra step after approaching goal to make the agent closer to goal
        self.forward_after_stop_preset = self.args.move_forward_after_stop
        self.forward_after_stop = self.forward_after_stop_preset

        self.map_size = args.map_size_cm // args.map_resolution
        self.full_w, self.full_h = self.map_size, self.map_size
        self.local_w = int(self.full_w / args.global_downscaling)
        self.local_h = int(self.full_h / args.global_downscaling)
        self.found_goal = None

        self.edge_buffer = 10 if args.num_sem_categories <= 16 else 40

        self.cluster = HdbscanCluster()
        self.diffusion_output = None
        self.new_long_term_goal = None
        self.new_pred_goal_map = None

        if args.visualize:
            self.legend = cv2.imread('nav/new_hm3d_legend.png')[:118]
            self.vis_image = None
            self.rgb_vis = None

    def reset(self):
        args = self.args
        
        isChanged = (len(self.object_category) != len(object_category))
        self.object_category = deepcopy(object_category)
        if isChanged:
            self.GSAM.set_text(object_category[:-1])
            # self.set_legend()
        
        # obs = self._preprocess_obs(obs)

        self.obs_shape = None

        # Episode initializations
        map_shape = (args.map_size_cm // args.map_resolution,
                     args.map_size_cm // args.map_resolution)
        self.collision_map = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        self.col_width = 1
        self.count_forward_actions = 0
        self.curr_loc = [args.map_size_cm / 100.0 / 2.0,
                         args.map_size_cm / 100.0 / 2.0, 0.]
        self.last_action = None
        self.episode_no += 1
        self.timestep = 0
        self.last_planning_window = None
        self.last_sem_map_vis = None
        self.count_windows = 0
        self.count_masks = 0
        self.prev_blocked = 0
        self._previous_action = -1
        self.block_threshold = 4
        self.untrap.reset(full=True)
        self.forward_after_stop = self.forward_after_stop_preset
        self.diffusion_output = None
        self.new_long_term_goal_point = None


    def plan_act(self, planner_inputs, full_planner_inputs):
        """
        Function responsible for motion planning and visualization.

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'obstacle'  (ndarray): (M, M) map prediction
                    'exp_pred'  (ndarray): (M, M) exploration mask 
                    'goal'      (ndarray): (M, M) mat denoting goal locations
                    'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found
                    'goal_name' (str): name of target category

        Returns:
            action (dict): {'action': action}
        """

        self.timestep += 1
        self.goal_name = planner_inputs['goal_name']

        if self.args.only_explore == 0 and self.args.visualize:
            self.update_pred_local_and_full_map(planner_inputs, full_planner_inputs)
            self._visualize(planner_inputs, full_planner_inputs)

        action = self._plan(planner_inputs)

        # if self.timestep > 111:
        #     exit()

        if self.args.only_explore and self.args.visualize:
            self._visualize(planner_inputs, full_planner_inputs)
            self._visualize_semmap(planner_inputs, full_planner_inputs)
            # print(planner_inputs['sem_map_pred'])
            # print(full_planner_inputs['sem_map_pred'])

        action = {'action': action}
        self.last_action = action['action']
        return action

    
    def find_closest_color(self, pixel, color_pal):
        """
        找到与给定像素最接近的颜色在 color_pal 中的索引。
        
        :param pixel: 输入像素的 RGB 值 (tuple 或 list)
        :param color_pal: 颜色调色板，形状为 (n, 3) 的 NumPy 数组
        :return: 最接近颜色在 color_pal 中的索引
        """
        distances = np.sqrt(np.sum((color_pal - pixel) ** 2, axis=1))
        return np.argmin(distances)

    def find_intersection(self, start, end, gx1, gx2, gy1, gy2):
        """
        计算线段与矩形边界的交点。
        
        参数:
            start (tuple): 线段起点 (x1, y1)。
            end (tuple): 线段终点 (x2, y2)。
            gx1, gx2, gy1, gy2 (float): 矩形边界。
        
        返回:
            tuple: 交点坐标，如果没有交点则返回 None。
        """
        x1, y1 = start
        x2, y2 = end
        
        # 计算线段的方向向量
        dx = x2 - x1
        dy = y2 - y1
        
        # 计算与矩形边界的交点
        def compute_intersection(x, y, dx, dy, boundary):
            t = (boundary - x) / dx if dx != 0 else np.inf
            return (x + t * dx, y + t * dy) if 0 <= t <= 1 else None
        
        # 检查与四条边的交点
        intersections = []
        for boundary in [gx1, gx2]:
            intersection = compute_intersection(x1, y1, dx, dy, boundary)
            if intersection and gy1 <= intersection[1] <= gy2:
                intersections.append(intersection)
        for boundary in [gy1, gy2]:
            intersection = compute_intersection(y1, x1, dy, dx, boundary)
            if intersection and gx1 <= intersection[1] <= gx2:
                intersections.append((intersection[1], intersection[0]))
        
        # 返回离起点最近的交点
        if intersections:
            distances = [np.linalg.norm(np.array(start) - np.array(p)) for p in intersections]
            return intersections[np.argmin(distances)]
        return None

    def find_navigation_target(self, centers, gx1, gx2, gy1, gy2, start):
        """
        找到长期导航目标点。
        
        参数:
            centers (np.ndarray): 聚类中心点矩阵，形状为 (N, 2)。
            gx1, gx2, gy1, gy2 (float): 区间范围，gx1 < gx2，gy1 < gy2。
            start (tuple): 当前坐标 (start_x, start_y)。
        
        返回:
            tuple: 长期导航目标点。
        """
        # 筛选出在 [gx1, gx2] x [gy1, gy2] 区间内的点

        
        mask = (centers[:, 0] >= gx1) & (centers[:, 0] <= gx2) & \
            (centers[:, 1] >= gy1) & (centers[:, 1] <= gy2)
        filtered_centers = centers[mask]
        
        if len(filtered_centers) > 0:
            # 如果存在满足条件的点，找到与 start 最近的点
            start = np.array(start)
            distances = np.linalg.norm(filtered_centers - start, axis=1)
            nearest_index = np.argmin(distances)
            return tuple(filtered_centers[nearest_index])
        else:
            # 如果不存在满足条件的点，找到全局最近的点
            start = np.array(start) 
            distances = np.linalg.norm(centers - start, axis=1)
            nearest_index = np.argmin(distances)
            nearest_center = tuple(centers[nearest_index])
            
            # 计算 start 到 nearest_center 的连线与边界的交点
            intersection = self.find_intersection(start, nearest_center, gx1, gx2, gy1, gy2)
            if intersection:
                return intersection
            else:
                print("No intersection is found, return to the nearest cluster center point.")
                return nearest_center
    
    def remove_sparse_coordinates(self, coords, radius=2, min_neighbors=6):
        """
        删除所有稀疏坐标（周围半径内坐标数量 < min_neighbors）返回剩余坐标。
        
        参数:
            coords (list): 输入坐标列表，格式为 [[x1, y1], [x2, y2], ...]。
            radius (float): 邻域半径。
            min_neighbors (int): 最小邻域数量阈值。
        
        返回:
            list: 过滤后的坐标列表。
        """
        if len(coords) == 0:
            return coords
        
        # 将坐标转换为 NumPy 数组
        points = np.array(coords)
        
        # 构建 KDTree 加速邻域搜索
        tree = KDTree(points)
        
        # 查询每个点的半径邻域内的所有点（包括自身）
        neighbor_indices = tree.query_ball_point(points, r=radius)
        
        # 统计每个点的邻域数量（排除自身）
        neighbor_counts = np.array([len(indices) - 1 for indices in neighbor_indices])
        
        # 保留邻域数量 >= min_neighbors 的点
        mask = neighbor_counts >= min_neighbors
        filtered_points = points[mask]
        
        return filtered_points

    def update_pred_local_and_full_map(self, inputs, f_inputs):
        """Generate semmap and save."""
        args = self.args

        map_pred = inputs['obstacle']
        exp_pred = inputs['exp_pred']

        f_map_pred = f_inputs['obstacle']
        f_exp_pred = f_inputs['exp_pred']

        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']
        # goal = inputs['goal']

        planning_window = [gx1, gx2, gy1, gy2]
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        new_pred_local_goal_map = None

        # print("planning_window:",planning_window)#[480*480]

        # Record Previous Step Window
        # if planning_window == self.last_planning_window or self.timestep == 1: # timestep==1是为了记录第一步的last_sem_map_vis
        sem_map = inputs['sem_map_pred'] #[480,480]
        
        # self.vis_image = vu.init_vis_image(self.goal_name, self.legend)

        sem_map += 5
        sem_map[self.collision_map[gx1:gx2, gy1:gy2] == 1] = 14
        # if int(self.stg[0]) < self.local_w and int(self.stg[1]) < self.local_h:
        #     sem_map[int(self.stg[0]),int(self.stg[1])] = 15

        no_cat_mask = sem_map == args.num_sem_categories + 4
        map_mask = np.rint(map_pred) == 1
        exp_mask = np.rint(exp_pred) == 1
        # vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1

        # Del Traj
        sem_map[sem_map == 3] = 2
        # Del long-time goal
        sem_map[sem_map == 4] = 0

        # Draw goal dot
        # selem = skimage.morphology.disk(4)
        # goal_mat = 1 - skimage.morphology.binary_dilation(
        #     goal, selem) != True
        
        # goal_mask = goal_mat == 1
        # sem_map[goal_mask] = 4

        color_pal = [int(x * 255.) for x in color_palette]
        sem_map_vis = Image.new("P", (sem_map.shape[1],
                                    sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                interpolation=cv2.INTER_NEAREST) ## build local map
        self.last_sem_map_vis = sem_map_vis

        # Record mask every local_steps 20 (begin from 1st step) 保留这段代码
        # if self.timestep % args.num_local_steps == 1:
        #     self.count_masks += 1
        #     mask_img = np.zeros_like(sem_map_vis)                
        #     lower_white = np.array([252, 252, 252])  
        #     upper_white = np.array([255, 255, 255])
        #     mask = cv2.inRange(sem_map_vis, lower_white, upper_white)
        #     white_only = cv2.bitwise_and(sem_map_vis, sem_map_vis, mask=mask)
        #     mask_img[mask != 0] = white_only[mask != 0]
            
        #     fn = '{}/semmap_mask/eps_{}/{}.jpg'.format(
        #         dump_dir, self.episode_no - 1,
        #         self.count_masks)
        #     cv2.imwrite(fn, mask_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        # else: #(self.count_windows == 0 and planning_window != self.last_planning_window) or (planning_window != self.last_planning_window and self.count_windows > 0):
        # self.count_windows += 1
        # sem_map_vis = self.last_sem_map_vis
        # inputs['sem_map_pred'] += 5

        # cv2.imwrite('sem_map_output_origin.png',sem_map_vis)

        # # Bulid full map
        # f_sem_map = f_inputs['sem_map_pred'] #[960,960]
        # f_sem_map += 5
        # f_sem_map[self.collision_map == 1] = 14

        # f_no_cat_mask = f_sem_map == args.num_sem_categories + 4
        # f_map_mask = np.rint(f_map_pred) == 1
        # f_exp_mask = np.rint(f_exp_pred) == 1
        # # vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1


        # f_sem_map[f_no_cat_mask] = 0
        # f_m1 = np.logical_and(f_no_cat_mask, f_exp_mask)
        # f_sem_map[f_m1] = 2

        # f_m2 = np.logical_and(f_no_cat_mask, f_map_mask)
        # f_sem_map[f_m2] = 1

        # color_pal = [int(x * 255.) for x in color_palette]
        # f_sem_map_vis = Image.new("P", (f_sem_map.shape[1],
        #                             f_sem_map.shape[0]))
        # f_sem_map_vis.putpalette(color_pal)
        # f_sem_map_vis.putdata(f_sem_map.flatten().astype(np.uint8))
        # f_sem_map_vis = f_sem_map_vis.convert("RGB")
        # f_sem_map_vis = np.flipud(f_sem_map_vis)
        # f_sem_map_vis = f_sem_map_vis[:, :, [2, 1, 0]]

        # f_sem_map_vis_origin = f_sem_map_vis
        
        # # f_sem_map_vis = self.crop_and_resize(f_sem_map_vis)
        # f_sem_map_vis = self.only_crop(f_sem_map_vis)
        # f_sem_map_vis = cv2.resize(f_sem_map_vis, (256, 256),
        #                         interpolation=cv2.INTER_NEAREST) ## build local map

        f_sem_map_vis = cv2.resize(sem_map_vis, (256, 256),
                                interpolation=cv2.INTER_NEAREST) ## build local map
        
        transform = transforms.Compose([
            transforms.Resize(size=(256, 256), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

#################################################===Diffusion Process===#################################################
        if self.timestep > 79 and self.timestep % 40 == 0 and inputs['found_goal'] == 0:
            # 方案1：直接预测全局语义地图
            ## 创建一个与f_sem_map_vis大小相同的全白mask (255表示白色)
            print("TimeStep {}, diffusion process...".format(self.timestep))
            mask = np.full(f_sem_map_vis.shape[:2], 255, dtype=np.uint8)
            white_threshold = np.array([245, 245, 245])
            white_mask = np.all(f_sem_map_vis > white_threshold, axis=-1)
            mask[white_mask] = 0 # 将这些部分在mask中设为黑色(0)
            
            mask = transform(Image.fromarray(mask))
            mask = mask.unsqueeze(0)

            f_sem_map_vis_n = f_sem_map_vis.astype(np.float32) / 255.
            gray_image = rgb2gray(np.array(tensor_to_image()(f_sem_map_vis_n)))
            edge = image_to_tensor()(Image.fromarray(canny(gray_image, sigma=2.)))
            gray_image = image_to_tensor()(Image.fromarray(gray_image))
            Y_GT, X_GT, X_LQ = f_sem_map_vis_n,gray_image,edge ##completed grayscale and edge images

            # transform = transforms.Compose([
            #     transforms.Resize(size=(256, 256), interpolation=Image.NEAREST),
            #     transforms.ToTensor(),
            # ])

            Y_GT = torch.from_numpy(Y_GT)
            Y_GT = Y_GT.permute(2, 0, 1).unsqueeze(0)


            noisy_state = self.sde.noise_state(Y_GT * mask) # 原本是有*mask的
            noisy_states = self.S_sde.noise_state(X_LQ * mask) # * mask
            self.model.feed_data(noisy_state, Y_GT * mask, Y_GT, mask, self.S_sde, X_GT,  X_LQ * mask)
            self.model.test(self.sde, save_states=True, GT = Y_GT, mask = mask, \
                            S_sde = self.S_sde, S_GT = X_GT, S_LQ = noisy_states, dis = self.model.dis, save_dir=None)
            
            # toc = time.time()#
            # test_times.append(toc - tic)
            visuals = self.model.get_current_visuals()
            SR_img = visuals["Output"]
            output = util.tensor2img(SR_img.squeeze())  # uint8
            LQ_ = util.tensor2img(visuals["Input"].squeeze())  # uint8
            GT_ = util.tensor2img(visuals["GT"].squeeze())  # uint8

            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB) # predict_img RGB
            # cv2.imwrite('sem_map_output_origin.png', output)
            output = cv2.resize(output, (480, 480), interpolation=cv2.INTER_NEAREST)
            # output = cv2.copyMakeBorder(output, 240, 240, 240, 240, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            black_pixels = np.all(output <= 35, axis=-1)
            output[black_pixels] = [255, 255, 255]

            # # OUTPUT
            # save_img_path = '/home/szx/project/PEANUT/data/Nav_test_result/predict.png'
            # util.save_img(output, save_img_path)
            # # MASK
            # SR_img_y = SR_img*mask
            # output_y = util.tensor2img(SR_img_y.squeeze())  # uint8
            # save_img_path = "/home/szx/project/PEANUT/data/Nav_test_result/mask.png"
            # output_y = cv2.cvtColor(output_y, cv2.COLOR_BGR2RGB)
            # util.save_img(output_y, save_img_path)
            color_pal = [int(x * 255.) for x in color_palette]
            color_pal_resize = np.array(color_pal).reshape(-1, 3)

            # 创建一个与原图像大小相同的单通道矩阵，用于存储索引
            # global_map_prediction_mapped = np.zeros((960, 960), dtype=np.uint8)
            # for y in range(960):
            #     for x in range(960):
            #         pixel = output[y, x]
            #         closest_color_index = self.find_closest_color(pixel, color_pal_resize)
            #         global_map_prediction_mapped[y, x] = closest_color_index
            # global_map_prediction_mapped = np.flipud(global_map_prediction_mapped)


            # # print(f_sem_map)
            # cv2.imwrite('f_sem_map.png', f_map_test)
            # exit()

            lmb = f_inputs['lmb']

            # local_sem_map = f_sem_map_vis_origin[lmb[0]:lmb[1], lmb[2]:lmb[3], :]
            local_sem_map = sem_map_vis
            local_sem_map = local_sem_map.copy()
            # print(color_pal_resize)
            global_goals = np.argwhere(output == color_pal_resize[self.goal_cat+5]) # goal_cat是基于map_category_names的映射  
            global_goals = self.remove_sparse_coordinates(global_goals)
            # 如果存在导航目标点
            # 考虑一个问题，如果存在多个呢？->建议使用简单的聚类方法
            if len(global_goals) > 0:
                global_goal = self.cluster.predict(global_goals)
                global_goal = [goal[:-1] for goal in global_goal]

                # r, c = start_y, start_x
                # start = [int(r * 100.0 / args.map_resolution - gx1),
                #         int(c * 100.0 / args.map_resolution - gy1)]
                # start = pu.threshold_poses(start, np.rint(inputs['obstacle']).shape)
                # (start_x,start_y)=start

                # print("1:",[gx1, gx2, gy1, gy2])
                # print("2:",lmb)
                
                # # 绘制所有点
                for global_goal2 in global_goals:
                    center2 = (global_goal2[1],global_goal2[0])
                    cv2.circle(local_sem_map, center2, radius=1, color=(255, 0, 0), thickness=-1)  # 蓝色点
                    cv2.circle(output, (global_goal2[1],global_goal2[0]), radius=1, color=(255, 0, 0), thickness=-1)  # 蓝色点

                # 在图像上绘制聚类中心点
                for center in global_goal:
                    local_center = (center[1], center[0])
                    cv2.circle(local_sem_map, local_center, radius=5, color=(0, 0, 255), thickness=-1)  # 红色点
                    cv2.circle(output, (center[1],center[0]), radius=5, color=(0, 0, 255), thickness=-1)  # 红色点
                
                pos = (
                    int((start_x * 100. / args.map_resolution - gy1)
                    * 480 / map_pred.shape[0]),
                    int((map_pred.shape[1] - start_y * 100. / args.map_resolution + gx1)
                    * 480 / map_pred.shape[1])
                )
                cv2.circle(output, pos, radius=8, color=(0, 255, 0), thickness=-1)  # 绿色点(start_x,start_y)

                # cv2.imwrite('sem_map.png', local_sem_map)
                # cv2.imwrite('sem_map_pred.png', output)

                ## 如果导航目标点在planning_window内
                # print("planning_window:",planning_window)
                if len(global_goal)>0:
                    fullmap_new_goal_point = self.find_navigation_target(np.array(global_goal), lmb[0],lmb[1],lmb[2],lmb[3], (start_x,start_y))
                    self.new_long_term_goal_point = (fullmap_new_goal_point[0],fullmap_new_goal_point[1])
                    
                    # for center in centers:
                    #     cv2.circle(image, tuple(center), radius=5, color=(0, 0, 255), thickness=-1)
                    new_pred_local_goal_map = np.zeros((inputs['goal'].shape[0],inputs['goal'].shape[1]))
                    new_pred_local_goal_map[self.new_long_term_goal_point[0],self.new_long_term_goal_point[1]] = 1

                    self.new_pred_goal_map = np.flipud(new_pred_local_goal_map)

            # self.diffusion_output = local_sem_map
            self.diffusion_output = output
            # self.diffusion_output = cv2.resize(self.diffusion_output, (480, 480),
            #                  interpolation=cv2.INTER_NEAREST)
            
            
            print("new long-term goal:",self.new_long_term_goal_point)

            # 方案2：先预测局部语义地图，在预测全局语义地图
            ## 一阶段扩散，获取预测局部语义地图



            # 方案3：只预测局部语义地图

            

        # and inputs['found_goal'] == 0


        if self.new_pred_goal_map is not None and inputs['found_goal'] == 0:
            inputs['goal'] = self.new_pred_goal_map

        self.last_planning_window = planning_window
    
    def set_goal_cat(self, goal_cat):
        self.goal_cat = goal_cat
        

    def preprocess_inputs(self, rgb, depth, info):
        obs = self._preprocess_obs(rgb, depth, info)

        self.obs = obs
        self.info = info

        return obs, info

    
    def _preprocess_obs(self, rgb, depth, info):
        args = self.args
        
        if args.use_gt_seg == 1:
            sem_seg_pred[:, :, self.goal_cat] = info['goalseg']

        elif args.use_gt_seg == 2: # GSAM

            # semantic = obs[:,:,4:5].squeeze()
            # BGR to RGB
            self.rgb_vis = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                                    (640, 480), interpolation=cv2.INTER_NEAREST)
            # print("obs: ", semantic.shape)
            semantic_output = self._get_sem_pred_GSAM(
                rgb.astype(np.uint8), depth, use_seg=True)
            sam_semantic_pred = semantic_output['sam_semantic_pred']
            sam_all_cls = convert_SAM(sam_semantic_pred, self.object_category)
            sem_seg_pred = sam_all_cls
        else:
            sem_seg_pred = self._get_sem_pred(rgb.astype(np.uint8), depth=depth)

        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)

        ds = args.env_frame_width // args.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred),
                               axis=2).transpose(2, 0, 1)

        return state

    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            # Invalid pixels have value = 0
            invalid = depth[:, i] == 0.
            if np.mean(invalid) > 0.9:
                depth[:, i][invalid] = depth[:, i].max()
            else:
                depth[:, i][invalid] = 100.0 

        # Also ignore too-far pixels
        mask2 = depth > 0.99
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0 

        # Convert to cm 
        depth = min_d * 100.0 + depth * (max_d - min_d) * 100.0
        return depth

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)


    def show_box(self, box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
        ax.text(x0, y0, label)

    def _get_sem_pred_GSAM(self, rgb, depth, use_seg=True):
        if use_seg:
            # # save rgb and depth
            # skimage.io.imsave("current_rgb.png", rgb)
            # skimage.io.imsave("current_depth.png", (np.repeat(depth, 3, axis=2) * 255).astype(np.uint8))
            
            self.semantics_vis = None
            image = torch.from_numpy(rgb).to(self.device).unsqueeze_(0).float()
            depth = torch.from_numpy(depth).to(self.device).unsqueeze_(0).float()
            with torch.no_grad():
                # print(image.shape, depth.shape) # torch.Size([1, 480, 640, 3]) torch.Size([1, 480, 640, 1])
                try:
                    rgb_Image = Image.fromarray(rgb).convert('RGB')
                    # if self.args.tag_freq > 0:
                    #     import random
                    #     random_num = random.randint(1, self.args.tag_freq)
                    #     if random_num == 1:
                    #         # tag and update
                    #         print("========> [DiscoverVLM]: tagging...")
                    #         ndo_list = tmp_llm.discover_objects(rgb_Image, self.object_category)
                    #         self.object_category = self.object_category[:-2] + ndo_list + self.object_category[-2:]
                    #         self.GSAM.add_text(ndo_list)
                    #         self.set_legend()
                    
                    sam_semantic_pred = self.GSAM.predict(rgb_Image) # (N, 1, 480, 640), we need (480, 640, 16)
                    self.semantics_vis = self.GSAM.get_vis(rgb_Image, sam_semantic_pred)


                except Exception as ex:
                    print(f"========> [SAM]: no object detected: {ex}")
                    sam_semantic_pred = None
                    
                if self.semantics_vis is None:
                    self.semantics_vis = self.rgb_vis
        else:
            raise NotImplementedError
        outputs = {
            "sam_semantic_pred": sam_semantic_pred,
        }
        return outputs

    def _get_sem_pred(self, rgb, depth=None):
        if self.args.visualize:
            self.rgb_vis = rgb[:, :, ::-1]

        sem_pred, sem_vis = self.seg_model.get_prediction(rgb, depth, goal_cat=self.goal_cat)
        return sem_pred.astype(np.float32)
    
    
    def _plan(self, planner_inputs):
        """
        Function responsible for planning.

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'obstacle'  (ndarray): (M, M) map prediction
                    'exp_pred'  (ndarray): (M, M) exploration mask 
                    'goal'      (ndarray): (M, M) mat denoting goal locations
                    'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found
                    'goal_name' (str): name of target category

        Returns:
            action (int): action id
        """
        args = self.args

        self.last_loc = self.curr_loc

        # Obstacle map
        map_pred = np.rint(planner_inputs['obstacle'])
        
        self.found_goal = planner_inputs['found_goal']
        goal = planner_inputs['goal']

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start_exact = [r * 100.0 / args.map_resolution - gx1,
                       c * 100.0 / args.map_resolution - gy1]
        start = [int(r * 100.0 / args.map_resolution - gx1),
                 int(c * 100.0 / args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        # print("start:",start)
        # print("planning_window:",planning_window)
        # print([self.timestep, start])

        # Get last loc
        last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0 / args.map_resolution - gx1),
                      int(c * 100.0 / args.map_resolution - gy1)]
        last_start = pu.threshold_poses(last_start, map_pred.shape)
        self.last_start = last_start
        self.visited_vis[gx1:gx2, gy1:gy2] = \
            vu.draw_line(last_start, start,
                         self.visited_vis[gx1:gx2, gy1:gy2])

        # Collision check
        if self.last_action == 1:
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4 if self.prev_blocked < self.block_threshold else 2
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                self.col_width = min(self.col_width, 1)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold:  # Collision
                self.prev_blocked += 1
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * \
                            ((i + buf) * np.cos(np.deg2rad(t1))
                             + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05 * \
                            ((i + buf) * np.sin(np.deg2rad(t1))
                             - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / args.map_resolution), \
                            int(c * 100 / args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map[r, c] = 1

            else:
                if self.prev_blocked >= self.block_threshold:
                    self.untrap.reset()
                self.prev_blocked = 0

        # Deterministic Local Policy
        stg, stop = self._get_stg(map_pred, start_exact, np.copy(goal),
                                  planning_window)

        if self.forward_after_stop < 0:
            self.forward_after_stop = self.forward_after_stop_preset
        if self.forward_after_stop != self.forward_after_stop_preset:
            if self.forward_after_stop == 0:
                self.forward_after_stop -= 1
                action = 0
            else:
                self.forward_after_stop -= 1
                action = 1
        elif stop and planner_inputs['found_goal'] == 1:
            if self.forward_after_stop == 0:
                action = 0  # Stop
            else:
                self.forward_after_stop -= 1
                action = 1
        else:
            (stg_x, stg_y) = stg
            
            # Stay within global map
            stg_x = np.clip(stg_x, self.edge_buffer, self.local_w - self.edge_buffer - 1)
            stg_y = np.clip(stg_y, self.edge_buffer, self.local_h - self.edge_buffer - 1)
            
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                    stg_y - start[1]))
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360
                
            if relative_angle > self.args.turn_angle / 2.:
                action = 3  # Right
            elif relative_angle < -self.args.turn_angle / 2.:
                action = 2  # Left
            else:
                action = 1  # Forward
                
        if self.prev_blocked >= self.block_threshold:
            if self._previous_action == 1:
                action = self.untrap.get_action()
            else:
                action = 1
        self._previous_action = action
        return action


    def _get_stg(self, grid, start, goal, planning_window):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        if gx2 == self.full_w:
            grid[x2 - 1] = 1
        if gy2 == self.full_h:
            grid[:, y2 - 1] = 1
            
        if gx1 == 0:
            grid[x1] = 1
        if gy1 == 0:
            grid[y1] = 1
            
        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        def surrounded_by_obstacle(mat,i,j):
            i = int(i)
            j = int(j)
            i1 = max(0,i-3)
            i2 = min(mat.shape[0],i+2)
            j1 = max(0,j-3)
            j2 = min(mat.shape[1],j+2)
            return np.sum(mat[i1:i2,j1:j2]) > 0
        

        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2],
            self.selem) != True

        traversible[self.collision_map[gx1:gx2, gy1:gy2]
                    [x1:x2, y1:y2] == 1] = 0
        traversible[self.visited_vis[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1


        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)

        planner = FMMPlanner(traversible)
        
        selem = skimage.morphology.disk(8 if self.found_goal == 1 else 2)

        # Smalller radius for toilet
        is_toilet = self.info['goal_name'] == 'toilet'
        if is_toilet:
            selem = skimage.morphology.disk(6 if self.found_goal == 1 else 2)

        goal = skimage.morphology.binary_dilation(
            goal, selem) != True

        goal = 1 - goal * 1.
        planner.set_multi_goal(goal)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        # assume replan true suggests failure in planning
        stg_x, stg_y, distance, stop, replan = planner.get_short_term_goal(state)
        
        # Failed to plan a path
        if replan:
            if self.args.only_explore:
                self.agent_states.next_preset_goal()

            # Try again with eroded obstacle map
            grid = skimage.morphology.binary_erosion(grid.astype(bool)).astype(int)
            traversible = skimage.morphology.binary_dilation(
                grid[x1:x2, y1:y2],
                self.selem) != True

            traversible[self.collision_map[gx1:gx2, gy1:gy2]
                        [x1:x2, y1:y2] == 1] = 0
            traversible[self.visited_vis[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1


            traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                        int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

            traversible = add_boundary(traversible)

            planner = FMMPlanner(traversible)
            planner.set_multi_goal(goal)

            state = [start[0] - x1 + 1, start[1] - y1 + 1]
            
            # assume replan true suggests failure in planning
            stg_x, stg_y, distance, stop, replan = planner.get_short_term_goal(state)
            

        #If we fail to plan a path to the goal, make goal larger
        if self.found_goal == 1 and distance > self.args.magnify_goal_when_hard:
            radius = 2
            step = 0

            while distance > 100:
                step += 1
                if step > 8 or (is_toilet and step > 2):
                    break
                selem = skimage.morphology.disk(radius)
                goal = skimage.morphology.binary_dilation(
                    goal, selem) != True
                goal = 1 - goal * 1.
                planner.set_multi_goal(goal)

                # assume replan true suggests failure in planning
                stg_x, stg_y, distance, stop, replan = planner.get_short_term_goal(
                    state)

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1
        self.stg = (stg_x, stg_y)
        return (stg_x, stg_y), stop

    def crop_and_resize(self, image, output_size=(256, 256)):
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 二值化处理
        _, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)  # 假设白色背景是255，设置阈值为240
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 如果没有找到任何轮廓，则可能是图像完全是白色，直接返回
        if not contours:
            print("Warning: No contours found, the image might be completely white.")
            resized_image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
            return resized_image
        
        # 找到最大的轮廓，假设这是我们要保留的主要图形
        largest_contour = max(contours, key=cv2.contourArea)
        # 获取包围矩形
        x, y, w, h = cv2.boundingRect(largest_contour)
        # 裁剪图像
        cropped_image = image[y:y+h, x:x+w]
        # 确保裁剪后的图像保持原始的长宽比
        original_aspect_ratio = w / h
        target_aspect_ratio = output_size[0] / output_size[1]
        
        if original_aspect_ratio > target_aspect_ratio:
            # 宽度过大，需要在上下添加填充
            pad_top = (w - h) // 2
            pad_bottom = w - h - pad_top
            padded_image = cv2.copyMakeBorder(cropped_image, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        else:
            # 高度过大，需要在左右添加填充
            new_width = h
            pad_left = (h - w) // 2
            pad_right = h - w - pad_left
            padded_image = cv2.copyMakeBorder(cropped_image, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        # 调整大小
        resized_image = cv2.resize(padded_image, output_size, interpolation=cv2.INTER_NEAREST)
        # print(padded_image.shape)
        return resized_image

    def only_crop(self, image, output_size=(256, 256)):
        
        # 获取原始图像的尺寸
        original_height, original_width = image.shape[:2]
        
        # 计算裁剪区域的起始坐标
        start_x = (original_width) // 2
        start_y = (original_height) // 2
    
        # 裁剪图像
        cropped_image = image[start_y-output_size[1]:start_y+output_size[1], start_x-output_size[0]:start_x+output_size[0]]
        # 调整大小
        resized_image = cv2.resize(cropped_image, output_size, interpolation=cv2.INTER_NEAREST)
        # print(padded_image.shape)
        return resized_image


    def _visualize_semmap(self, inputs, f_inputs):
        """Generate semmap and save."""
        args = self.args

        dump_dir = "{}/mapdata/".format(args.dump_location)
        sem_dir = '{}/semmap/eps_{}/'.format(
            dump_dir, self.episode_no - 1)
        f_sem_dir = '{}/full_semmap/eps_{}/'.format(
            dump_dir, self.episode_no - 1)
        # mask_dir = '{}/semmap_mask/eps_{}/'.format(
        #     dump_dir, self.episode_no - 1)
        if not os.path.exists(sem_dir):
            os.makedirs(sem_dir)
        if not os.path.exists(f_sem_dir):
            os.makedirs(f_sem_dir)
        # if not os.path.exists(mask_dir):
        #     os.makedirs(mask_dir)

        map_pred = inputs['obstacle']
        exp_pred = inputs['exp_pred']

        f_map_pred = f_inputs['obstacle']
        f_exp_pred = f_inputs['exp_pred']

        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']
        # goal = inputs['goal']

        planning_window = [gx1, gx2, gy1, gy2]


        # Record Previous Step Window
        if planning_window == self.last_planning_window or self.timestep == 1:
            sem_map = inputs['sem_map_pred'] #[240,240]
            
            # self.vis_image = vu.init_vis_image(self.goal_name, self.legend)

            gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

            # sem_map += 5
            sem_map[self.collision_map[gx1:gx2, gy1:gy2] == 1] = 14
            if int(self.stg[0]) < self.local_w and int(self.stg[1]) < self.local_h:
                sem_map[int(self.stg[0]),int(self.stg[1])] = 15

            no_cat_mask = sem_map == args.num_sem_categories + 4
            map_mask = np.rint(map_pred) == 1
            exp_mask = np.rint(exp_pred) == 1
            # vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1

            sem_map[no_cat_mask] = 0
            m1 = np.logical_and(no_cat_mask, exp_mask)
            sem_map[m1] = 2

            m2 = np.logical_and(no_cat_mask, map_mask)
            sem_map[m2] = 1

            # Del Traj
            sem_map[sem_map == 3] = 2
            # Del long-time goal
            sem_map[sem_map == 4] = 0

            # Draw goal dot
            # selem = skimage.morphology.disk(4)
            # goal_mat = 1 - skimage.morphology.binary_dilation(
            #     goal, selem) != True
            
            # goal_mask = goal_mat == 1
            # sem_map[goal_mask] = 4

            color_pal = [int(x * 255.) for x in color_palette]
            sem_map_vis = Image.new("P", (sem_map.shape[1],
                                        sem_map.shape[0]))
            sem_map_vis.putpalette(color_pal)
            sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
            sem_map_vis = sem_map_vis.convert("RGB")
            sem_map_vis = np.flipud(sem_map_vis)

            sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
            sem_map_vis = cv2.resize(sem_map_vis, (256, 256),
                                    interpolation=cv2.INTER_NEAREST)
            self.last_sem_map_vis = sem_map_vis
            
            if self.timestep >= 499: # 如何设定保存全局语义地图阈值？
                # Bulid full map
                f_sem_map = f_inputs['sem_map_pred'] #[960,960]
                f_sem_map += 5
                f_sem_map[self.collision_map == 1] = 14

                f_no_cat_mask = f_sem_map == args.num_sem_categories + 4
                f_map_mask = np.rint(f_map_pred) == 1
                f_exp_mask = np.rint(f_exp_pred) == 1
                # vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1


                f_sem_map[f_no_cat_mask] = 0
                f_m1 = np.logical_and(f_no_cat_mask, f_exp_mask)
                f_sem_map[f_m1] = 2

                f_m2 = np.logical_and(f_no_cat_mask, f_map_mask)
                f_sem_map[f_m2] = 1

                # # Del Traj
                # f_sem_map[f_sem_map == 3] = 2
                # # Del long-time goal
                # f_sem_map[f_sem_map == 4] = 0

                # f_sem_map -= 9

                # color_pal = [int(x * 255.) for x in color_palette]
                # print(sem_map)
                # print(f_sem_map)
                
                f_sem_map_vis = Image.new("P", (f_sem_map.shape[1],
                                            f_sem_map.shape[0]))
                f_sem_map_vis.putpalette(color_pal)
                f_sem_map_vis.putdata(f_sem_map.flatten().astype(np.uint8))
                f_sem_map_vis = f_sem_map_vis.convert("RGB")
                f_sem_map_vis = np.flipud(f_sem_map_vis)
                f_sem_map_vis = f_sem_map_vis[:, :, [2, 1, 0]]
                # f_sem_map_vis = self.crop_and_resize(f_sem_map_vis)
                f_sem_map_vis = self.only_crop(f_sem_map_vis)
                # f_sem_map_vis = cv2.resize(f_sem_map_vis, (256, 256),
                #                         interpolation=cv2.INTER_NEAREST)
                fn = '{}/full_semmap/eps_{}/full_map_{}.jpg'.format(
                        dump_dir, self.episode_no - 1,
                        self.episode_no - 1)
                cv2.imwrite(fn, f_sem_map_vis, [cv2.IMWRITE_JPEG_QUALITY, 100])

                # exit()


            # Record mask every local_steps 20 (begin from 1st step)
            # if self.timestep % args.num_local_steps == 1:
            #     self.count_masks += 1
            #     mask_img = np.zeros_like(sem_map_vis)                
            #     lower_white = np.array([252, 252, 252])  
            #     upper_white = np.array([255, 255, 255])
            #     mask = cv2.inRange(sem_map_vis, lower_white, upper_white)
            #     white_only = cv2.bitwise_and(sem_map_vis, sem_map_vis, mask=mask)
            #     mask_img[mask != 0] = white_only[mask != 0]
                
            #     fn = '{}/semmap_mask/eps_{}/{}.jpg'.format(
            #         dump_dir, self.episode_no - 1,
            #         self.count_masks)
            #     cv2.imwrite(fn, mask_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        else: #(self.count_windows == 0 and planning_window != self.last_planning_window) or (planning_window != self.last_planning_window and self.count_windows > 0):
            self.count_windows += 1
            sem_map_vis = self.last_sem_map_vis
            fn = '{}/semmap/eps_{}/{}.jpg'.format(
                    dump_dir, self.episode_no - 1,
                    self.count_windows)
            # cv2.imwrite(fn, sem_map_vis, [cv2.IMWRITE_JPEG_QUALITY, 100]) 2025.1.4 在使用全局语义图过程中，我们暂时不画出来

        
        self.last_planning_window = planning_window



    def _visualize(self, inputs, f_inputs):
        """Generate visualization and save."""

        args = self.args
        dump_dir = "{}/dump/{}/".format(args.dump_location,
                                        args.exp_name)
        ep_dir = '{}/episodes/thread_{}/eps_{}/'.format(
            dump_dir, self.rank, self.episode_no - 1)
        if args.only_explore == 0 and args.visualize == 2:
            if not os.path.exists(ep_dir):
                os.makedirs(ep_dir)

        map_pred = inputs['obstacle']
        exp_pred = inputs['exp_pred']

        f_map_pred = f_inputs['obstacle']
        f_exp_pred = f_inputs['exp_pred']

        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']
        goal = inputs['goal']

        sem_map = inputs['sem_map_pred'] #[240,240]
        
        self.vis_image = vu.init_vis_image(self.goal_name, self.legend)

        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        
        if args.only_explore == 0:
            sem_map += 0
        else:
            sem_map += 5
        
        sem_map[self.collision_map[gx1:gx2, gy1:gy2] == 1] = 14
        # if int(self.stg[0]) < self.local_w and int(self.stg[1]) < self.local_h:
        #     sem_map[int(self.stg[0]),int(self.stg[1])] = 15

        no_cat_mask = sem_map == args.num_sem_categories + 4
        map_mask = np.rint(map_pred) == 1
        exp_mask = np.rint(exp_pred) == 1
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1

        sem_map[vis_mask] = 3

        # Draw goal dot
        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal, selem) != True
        
        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4

        color_pal = [int(x * 255.) for x in color_palette]
        sem_map_vis = Image.new("P", (sem_map.shape[1],
                                      sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                 interpolation=cv2.INTER_NEAREST)
        self.vis_image[50:530, 15:655] = self.rgb_vis
        self.vis_image[50:530, 670:1150] = sem_map_vis
        
        if self.diffusion_output is not None:
            self.vis_image[50:530, 1165:] = self.diffusion_output
            self.vis_image[50:530, 1164+480] = [100,100,100]
        # right_panel = self.vis_image[:, -250:] 
                                   
        # my_cm = matplotlib.cm.get_cmap('Purples')
        # data = self.agent_states.target_pred
        # if data is not None:
        #     normed_data = (data - np.min(data)) / (np.max(data) - np.min(data))
                                   
        #     mapped_data = my_cm(normed_data)[::-1, :, [2, 1, 0]] * 255
            
        #     white_idx = np.where(np.sum(sem_map_vis, axis=2) == 255 * 3) 
        #     mapped_data_vis = cv2.resize(mapped_data, (480, 480),
        #                          interpolation=cv2.INTER_NEAREST)
        #     self.vis_image[50:530, 670:1150][white_idx] = mapped_data_vis[white_idx]


        #     data = self.agent_states.value
        #     normed_data = (data - np.min(data)) / (np.max(data) - np.min(data))
        #     mapped_data = my_cm(normed_data)[::-1, :, [2, 1, 0]] * 255
        #     mapped_data_vis = cv2.resize(mapped_data, (240, 240),
        #                      interpolation=cv2.INTER_NEAREST)
        #     right_panel[290:530, :240] = mapped_data_vis

        #     data = self.agent_states.dd_wt
        #     normed_data = (data - np.min(data)) / (np.max(data) - np.min(data))
        #     mapped_data = my_cm(normed_data)[::-1, :, [2, 1, 0]] * 255
        #     mapped_data_vis = cv2.resize(mapped_data, (240, 240),
        #                      interpolation=cv2.INTER_NEAREST)
        #     right_panel[50:290, :240] = mapped_data_vis
            
        #     border_color = [100] * 3
        #     right_panel[49, :240] = border_color
        #     right_panel[530, :240] = border_color
        #     right_panel[49:531, 240] = border_color
            
                            
        pos = (
            (start_x * 100. / args.map_resolution - gy1)
            * 480 / map_pred.shape[0],
            (map_pred.shape[1] - start_y * 100. / args.map_resolution + gx1)
            * 480 / map_pred.shape[1],
            np.deg2rad(-start_o)
        )

        # Draw agent as an arrow
        agent_arrow = vu.get_contour_points(pos, origin=(670, 50))
        color = (int(color_palette[11] * 255),
                 int(color_palette[10] * 255),
                 int(color_palette[9] * 255))
        cv2.drawContours(self.vis_image, [agent_arrow], 0, color, -1) 
        
        if args.visualize == 1:
            # Displaying the image
            cv2.imshow("Thread {}".format(self.rank), self.vis_image)
            cv2.waitKey(1)

        elif args.visualize == 2:
            # Saving the image
            if args.only_explore:
                pass
            fn = '{}/episodes/thread_{}/eps_{}/{}-{}-Vis-{}.jpg'.format(
                dump_dir, self.rank, self.episode_no - 1,
                self.rank, self.episode_no - 1, self.timestep)

            cv2.imwrite(fn, self.vis_image, [cv2.IMWRITE_JPEG_QUALITY, 100])


