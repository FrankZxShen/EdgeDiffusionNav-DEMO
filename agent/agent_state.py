from collections import deque, defaultdict
import os
import logging
import time
import json
import gym
import torch.nn as nn
import torch
import numpy as np
import skfmm
import skimage.morphology
from skimage import measure
from numpy import ma
import cv2

from agent.mapping import Semantic_Mapping
# from agent.prediction import PEANUT_Prediction_Model
from arguments import get_args


def add_boundary(mat, value=1):
    h, w = mat.shape
    new_mat = np.zeros((h + 2, w + 2)) + value
    new_mat[1:h + 1, 1:w + 1] = mat
    return new_mat

def find_frontiers(full_map_pred):
    # ------------------------------------------------------------------
    ##### Get the frontier map and filter
    # ------------------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    full_w = full_map_pred.shape[1]
    local_ex_map = np.zeros((full_w, full_w))
    local_ob_map = np.zeros((full_w, full_w))

    local_ob_map = cv2.dilate(full_map_pred[0].cpu().numpy(), kernel)

    show_ex = cv2.inRange(full_map_pred[1].cpu().numpy(),0.1,1)
    
    kernel = np.ones((5, 5), dtype=np.uint8)
    free_map = cv2.morphologyEx(show_ex, cv2.MORPH_CLOSE, kernel)

    contours,_=cv2.findContours(free_map, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    if len(contours)>0:
        contour = max(contours, key = cv2.contourArea)
        cv2.drawContours(local_ex_map,contour,-1,1,1)

    # clear the boundary
    local_ex_map[0:2, 0:full_w]=0.0
    local_ex_map[full_w-2:full_w, 0:full_w-1]=0.0
    local_ex_map[0:full_w, 0:2]=0.0
    local_ex_map[0:full_w, full_w-2:full_w]=0.0

    target_edge = local_ex_map-local_ob_map
    # print("local_ob_map ", self.local_ob_map[200])
    # print("full_map ", self.full_map[0].cpu().numpy()[200])

    target_edge[target_edge>0.8]=1.0
    target_edge[target_edge!=1.0]=0.0

    wall_edge = local_ex_map - target_edge

    # contours, hierarchy = cv2.findContours(cv2.inRange(wall_edge,0.1,1), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # if len(contours)>0:
    #     dst = np.zeros(wall_edge.shape)
    #     cv2.drawContours(dst, contours, -1, 1, 1)

    # edges = cv2.Canny(cv2.inRange(wall_edge,0.1,1), 30, 90)

    # Wall_lines = cv2.HoughLinesP(cv2.inRange(wall_edge,0.1,1), 1, np.pi / 180, threshold=30, minLineLength=10, maxLineGap=10)

    # original_image_color = cv2.cvtColor(cv2.inRange(wall_edge,0.1,1), cv2.COLOR_GRAY2BGR)
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(original_image_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

    
    img_label, num = measure.label(target_edge, connectivity=2, return_num=True)#输出二值图像中所有的连通域
    props = measure.regionprops(img_label)#输出连通域的属性，包括面积等

    Goal_edge = np.zeros((img_label.shape[0], img_label.shape[1]))
    Goal_point = []
    Goal_area_list = []
    dict_cost = {}
    for i in range(1, len(props)):
        if props[i].area > 4:
            dict_cost[i] = props[i].area

    if dict_cost:
        dict_cost = sorted(dict_cost.items(), key=lambda x: x[1], reverse=True)

        for i, (key, value) in enumerate(dict_cost):
            Goal_edge[img_label == key + 1] = 1
            Goal_point.append([int(props[key].centroid[0]), int(props[key].centroid[1])])
            Goal_area_list.append(value)
            if i == 3:
                break
        # frontiers = cv2.HoughLinesP(cv2.inRange(Goal_edge,0.1,1), 1, np.pi / 180, threshold=10, minLineLength=10, maxLineGap=10)

        # original_image_color = cv2.cvtColor(cv2.inRange(Goal_edge,0.1,1), cv2.COLOR_GRAY2BGR)
        # if frontiers is not None:
        #     for frontier in frontiers:
        #         x1, y1, x2, y2 = frontier[0]
        #         cv2.line(original_image_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return Goal_area_list, Goal_edge, Goal_point

class Agent_State:
    """
    Class containing functions updating map and prediction.
    """

    def __init__(self,args):
        self.args = args
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
        self.device = args.device = torch.device("cuda:" + str(args.sem_gpu_id) if args.cuda else "cpu")

        self.nc = 4 + args.num_sem_categories  # num channels in map
        
        self.map_size = args.map_size_cm // args.map_resolution
        self.full_w, self.full_h = self.map_size, self.map_size
        self.local_w = int(self.full_w / args.global_downscaling)
        self.local_h = int(self.full_h / args.global_downscaling)

        # Initializing full and local map
        # Channels:
        # 0: obstacle map
        # 1: exploration map
        # 2: agent's current location
        # 3: all past locations
        # Rest: semantic categories
        self.full_map = torch.zeros(self.nc, self.full_w, self.full_h).float().to(
            self.device)

        self.local_map = torch.zeros(self.nc, self.local_w,
                                self.local_h).float().to(self.device)

        # Initial full and local pose
        self.full_pose = torch.zeros( 3).float().to(self.device)
        self.local_pose = torch.zeros( 3).float().to(self.device)

        # Origin of local map
        self.origins = np.zeros((3))

        # Local Map Boundaries
        self.lmb = np.zeros(( 4)).astype(int)

        # Planner pose inputs has 7 dimensions
        # 1-3 store continuous global agent location
        # 4-7 store local map boundaries
        self.planner_pose_inputs = np.zeros(( 7))

        # Semantic Mapping
        self.sem_map_module = Semantic_Mapping(args).to(self.device)
        self.sem_map_module.eval()

        # Preset corner goals (for Stubborn data collection policy)
        self.global_goal_presets = [(0.1,0.1),(0.9,0.1),(0.9,0.9),(0.1,0.9)]
        self.global_goal_preset_id = 0

        # Unseen Target Prediction
        # self.prediction_model = PEANUT_Prediction_Model(args) if args.only_explore == 0 else None 
        self.prediction_model = None 
        self.selem = skimage.morphology.disk(args.col_rad)
        self.selem_idx = np.where(skimage.morphology.disk(args.col_rad + 1) > 0)
        self.target_pred = None
        self.value = None
        self.dd_wt = None
        self.last_global_goal = None

        # SemMaps Predition

        

    def reset(self):
        """Reset per episode."""
        self.l_step = 0
        self.step = 0
        self.goal_cat = -1
        self.found_goal = False
        self.init_map_and_pose()
        self.target_pred = None
        self.value = None
        self.dd_wt = None
        self.last_global_goal = None

        
    def init_with_obs(self, obs, infos):
        """Initialize from initial observation."""

        self.l_step = 0
        self.step = 0

        self.poses = torch.from_numpy(np.asarray(
            infos['sensor_pose'])
        ).float().to(self.device)
        _, self.local_map, _, self.local_pose = \
            self.sem_map_module(obs, self.poses, self.local_map, self.local_pose,self)

        # Compute Global policy input
        self.locs = self.local_pose.cpu().numpy()

        r, c = self.locs[1], self.locs[0]
        loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                        int(c * 100.0 / self.args.map_resolution)]

        self.local_map[2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.
        
        rgoal = [0.1, 0.1]
        self.global_goals = [[int(rgoal[0] * self.local_w), int(rgoal[1] * self.local_h)]]
        self.global_goals = [[min(x, int(self.local_w - 1)), min(y, int(self.local_h - 1))]
                        for x, y in self.global_goals]

        self.goal_maps = np.zeros((self.local_w, self.local_h))
        self.goal_maps[self.global_goals[0][0], self.global_goals[0][1]] = 1


        p_input = {}

        p_input['obstacle'] = self.local_map[0, :, :].cpu().numpy()
        p_input['exp_pred'] = self.local_map[1, :, :].cpu().numpy()
        p_input['pose_pred'] = self.planner_pose_inputs
        p_input['goal'] = self.goal_maps  # global_goals[e]
        p_input['new_goal'] = 1
        p_input['found_goal'] = 0
        if self.args.visualize:
            vlm = torch.clone(self.local_map[4:, :, :])
            vlm[-1] = 1e-5
            p_input['sem_map_pred'] = vlm.argmax(0).cpu().numpy()


        self.planner_inputs = p_input

        torch.set_grad_enabled(False)


    def get_local_map_boundaries(self, agent_loc, local_sizes, full_sizes):
        """Get boundaries of local map with respect to full map."""

        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if self.args.global_downscaling > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx1, gy1 = gx1 - gx1%self.args.grid_resolution, gy1 - gy1%self.args.grid_resolution
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]


    def init_map_and_pose(self):
        """Initialize maps and pose variables."""

        args = self.args
        self.full_map.fill_(0.)
        self.full_pose.fill_(0.)

        self.full_pose[:2] = self.args.map_size_cm / 100.0 / 2.0

        locs = self.full_pose.cpu().numpy()
        self.planner_pose_inputs[:3] = locs

        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        self.full_map[2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

        self.lmb = self.get_local_map_boundaries((loc_r, loc_c),
                                                 (self.local_w, self.local_h),
                                                 (self.full_w, self.full_h))
        # print(loc_r)
        # print(loc_c)
        # print("self.lmb:",self.lmb[0])
        # print("self.lmb:",self.lmb[1])
        # print("self.lmb:",self.lmb[2])
        # print("self.lmb:",self.lmb[3])
        # print(self.full_map[4:, :, :].shape) #[10,960,960]
        # exit()

        self.planner_pose_inputs[3:] = self.lmb
        self.origins = np.array([self.lmb[2] * args.map_resolution / 100.0,
                        self.lmb[0] * args.map_resolution / 100.0, 0.])
        self.local_map = self.full_map[:,
                         self.lmb[0]:self.lmb[1],
                         self.lmb[2]:self.lmb[3]]
        self.local_pose = self.full_pose - \
                          torch.from_numpy(self.origins).to(self.device).float()


    def update_state(self, obs, infos):
        """Update agent state, including semantic map, target prediction, and long-term goal."""

        args = self.args

        self.goal_cat = infos['goal_cat_id']
        self.poses = torch.from_numpy(np.asarray(
            infos['sensor_pose'] )
        ).float().to(self.device)
        
        self.update_local_map(obs)

        if self.l_step == args.num_local_steps - 1:
            self.l_step = 0

            self.update_full_map()

            # If we don't want to activate prediction yet
            if self.step < args.switch_step:
                # Set goal to corner (like Stubborn)
                preset = self.global_goal_presets[self.global_goal_preset_id]
                self.global_goals = [[int(preset[0] * self.local_w), int(preset[1] * self.local_h)]]
                self.global_goals = [[min(x, int(self.local_w - 1)),
                                 min(y, int(self.local_h - 1))]
                                for x, y in self.global_goals]
                
        # Activating prediction 
        if (self.step % args.update_goal_freq == args.update_goal_freq - 1 or 
            self.step == 0 or
            self.dist_to_goal < args.goal_reached_dist) and self.step >= args.switch_step:
            
            #暂时注释为为pass，后续有了diffusion的图片需要改
            # self.update_prediction()
            # self.update_global_goal()
            self.update_global_goal_only_fbe()
       
        self.update_goal_map(infos)
                    
        # ------------------------------------------------------------------
        # Assemble planner inputs
        p_input = {}
        p_input['obstacle'] = self.local_map[0, :, :].cpu().numpy()
        p_input['exp_pred'] = self.local_map[1, :, :].cpu().numpy()
        p_input['pose_pred'] = self.planner_pose_inputs
        p_input['goal'] = self.goal_map  
        p_input['found_goal'] = self.found_goal
        p_input['goal_name'] = infos['goal_name']

        # full_map
        f_input = {}
        f_input['obstacle'] = self.full_map[0, :, :].cpu().numpy()
        f_input['exp_pred'] = self.full_map[1, :, :].cpu().numpy()
        f_input['pose_pred'] = self.planner_pose_inputs
        f_input['lmb'] = self.lmb
        f_input['goal'] = self.goal_map  
        f_input['found_goal'] = self.found_goal
        f_input['goal_name'] = infos['goal_name']

        if args.visualize:
            vlm = torch.clone(self.local_map[4:, :, :])
            vlm[-1] = 1e-5
            p_input['sem_map_pred'] = vlm.argmax(0).cpu().numpy() # [480,480]

            fvlm = torch.clone(self.full_map[4:, :, :])
            fvlm[-1] = 1e-5
            f_input['sem_map_pred'] = fvlm.argmax(0).cpu().numpy() # [960,960]

        self.inc_step()
        return p_input, f_input


    def update_local_map(self, obs):
        """Update the agent's local map."""

        args = self.args

        _, self.local_map, _, self.local_pose = \
            self.sem_map_module(obs, self.poses, self.local_map, self.local_pose,self)
        
        locs = self.local_pose.cpu().numpy()
        self.planner_pose_inputs[:3] = locs + self.origins
        self.local_map[2, :, :].fill_(0.)  # Resetting current location channel

        r, c = locs[1], locs[0]
        loc_r = int(r * 100.0 / args.map_resolution)
        loc_c = int(c * 100.0 / args.map_resolution)
        
        # Recording agent trajectory
        traj_rad = 2
        self.local_map[2:4, loc_r - traj_rad:loc_r + traj_rad + 1, loc_c - traj_rad:loc_c + traj_rad + 1] = 1.

        # Explored under the agent
        to_fill = (self.selem_idx[0] - (args.col_rad+1) + loc_r, self.selem_idx[1] - (args.col_rad+1) + loc_c)
        self.local_map[1][to_fill] = 1.
        
        # Ensure goal is marked as explored once we get close enough
        self.dist_to_goal = np.sqrt((loc_r - (self.global_goals[0][0]))**2 + (loc_c - (self.global_goals[0][1]))**2) * args.map_resolution
        if self.dist_to_goal < args.goal_reached_dist:
            to_fill = (self.selem_idx[0] - (args.col_rad+1) + self.global_goals[0][0], 
                       self.selem_idx[1] - (args.col_rad+1) + self.global_goals[0][1])
            self.local_map[1][to_fill] = 1.

        self.loc_r = loc_r
        self.loc_c = loc_c


    def update_full_map(self):
        """Update the agent's full (global) map."""

        args = self.args 

        self.full_map[:, self.lmb[0]:self.lmb[1], self.lmb[2]:self.lmb[3]] = \
                self.local_map
        res = self.args.grid_resolution

        self.full_pose = self.local_pose + \
                            torch.from_numpy(self.origins).to(self.device).float()

        locs = self.full_pose.cpu().numpy()
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        self.lmb = self.get_local_map_boundaries((loc_r, loc_c),
                                                    (self.local_w, self.local_h),
                                                    (self.full_w, self.full_h))

        self.planner_pose_inputs[3:] = self.lmb
        self.origins = np.array([self.lmb[2] * args.map_resolution / 100.0,
                        self.lmb[0] * args.map_resolution / 100.0, 0.])

        self.local_map = self.full_map[:,
                            self.lmb[0]:self.lmb[1],
                            self.lmb[2]:self.lmb[3]]
        self.local_pose = self.full_pose - \
                            torch.from_numpy(self.origins).to(self.device).float()


        locs = self.local_pose.cpu().numpy()
        r, c = locs[1], locs[0]
        self.loc_r = int(r * 100.0 / args.map_resolution)
        self.loc_c = int(c * 100.0 / args.map_resolution)
                        

    def next_preset_goal(self):
        self.global_goal_preset_id = (self.global_goal_preset_id + 1) % len(self.global_goal_presets)


    def update_prediction(self):
        """Update unseen target prediction."""

        args = self.args

        self.full_map[:, self.lmb[0]:self.lmb[1], self.lmb[2]:self.lmb[3]] = \
                    self.local_map
            
        # Get prediction aligned with agent's global map
        if self.full_w == args.prediction_window and self.full_h == args.prediction_window:
            object_preds = self.prediction_model.get_prediction(self.full_map.cpu().numpy())
        else:
            x1 = self.full_w // 2 - args.prediction_window // 2
            x2 = x1 + args.prediction_window
            y1 = self.full_h // 2 - args.prediction_window // 2
            y2 = y1 + args.prediction_window
            object_preds = self.prediction_model.get_prediction(self.full_map[:, x1:x2, y1:y2].cpu().numpy())
            temp = np.zeros((object_preds.shape[0], self.full_w, self.full_h))
            temp[:, x1:x2, y1:y2] = object_preds
            object_preds = temp
            
        target = self.goal_cat
            
        # Extract the prediction in the local map bounds
        target_pred = object_preds[target,
                                    self.lmb[0]:self.lmb[1],
                                    self.lmb[2]:self.lmb[3]]
        target_pred *= self.local_map[1].cpu().numpy() < 0.5  # unexplored regions only
        self.target_pred = target_pred
    
    
    def update_global_goal_only_fbe(self):
        """Update long-term goal based on current location. ONLY FBE"""
        
        full_Frontier_list, full_target_edge_map, full_target_point_map = find_frontiers(self.local_map)
        # print(full_Frontier_list)
        # print(full_target_edge_map)
        # print(full_target_point_map)
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = self.planner_pose_inputs
        # # 计算每个点与 start 的欧几里得距离
        # distances = np.linalg.norm(full_target_point_map - np.array((start_x, start_y)), axis=1)
        # # 找到最小距离的索引
        # nearest_index = np.argmin(distances)
        if len(full_target_point_map) > 0:
            index = np.random.randint(0,len(full_target_point_map))
            self.global_goals = [full_target_point_map[index]]
        else:
            actions = np.random.rand(1, 2).squeeze()*(480 - 1)
            self.global_goals = [int(actions[0]), int(actions[1])]


    def update_global_goal(self):
        """Update long-term goal based on current location."""

        args = self.args

        # Weight value based on inverse geodesic distance
        trav = skimage.morphology.binary_dilation(np.rint(self.full_map[0].cpu().numpy()), self.selem) != True
        gx1, gx2, gy1, gy2 = int(self.lmb[0]), int(self.lmb[1]), int(self.lmb[2]), int(self.lmb[3])
        
        trav[self.helper.collision_map == 1] = 0     
        trav[self.helper.visited_vis == 1] = 1
        
        traversible_ma = ma.masked_values(trav * 1, 0)
        traversible_ma[np.clip(self.loc_r + self.lmb[0], a_min=0, a_max=self.full_w-1), 
                        np.clip(self.loc_c + self.lmb[2], a_min=0, a_max=self.full_h-1)] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd = ma.filled(dd, np.max(dd) + 1)
        dd[np.where(dd == np.max(dd))] = np.inf  # not traversible

        temperature = args.dist_weight_temperature / args.map_resolution
        dd_wt = np.exp(-dd / temperature)[self.lmb[0]:self.lmb[1], self.lmb[2]:self.lmb[3]]
        
        if np.sum(dd_wt) < 10 and self.dd_wt is not None:  # stuck inside obstacle, use last dd_wt
            dd_wt = self.dd_wt
            
        if args.dist_weight_temperature == -1:  # no weighting
            value = self.target_pred
        elif args.dist_weight_temperature == 0:  # frontier-based exploration
            dd[np.where(dd < 60)] = np.inf
            value = np.exp(-dd / 100.)[self.lmb[0]:self.lmb[1], self.lmb[2]:self.lmb[3]]
        else:
            value = self.target_pred * dd_wt

        self.dd_wt = dd_wt
        self.value = value
        
        new_global_goal = [np.unravel_index(value.argmax(), value.shape)]
        if new_global_goal != self.last_global_goal:  # avoid repeating the last goal
            self.last_global_goal = self.global_goals
            self.global_goals = new_global_goal


    def update_goal_map(self, infos):
        """Produce goal map to send to planner."""
        
        args = self.args

        # self.global_goals[0] = [24,400]

        # print("self.global_goals:",self.global_goals[0])

        self.found_goal = 0
        self.goal_map = np.zeros((self.local_w, self.local_h))
        self.goal_map[self.global_goals[0][0], self.global_goals[0][1]] = 1

        # Update long-term goal if target object is found
        if self.args.only_explore == 0:
            cn = self.goal_cat + 4
            if self.local_map[cn, :, :].sum() != 0.:
                cat_semantic_map = self.local_map[cn, :, :].cpu().numpy()
                cat_semantic_scores = cat_semantic_map
                cat_semantic_scores[cat_semantic_scores > 0] = 1.
                temp_goal = cat_semantic_scores

                # Erosion to remove noisy pixels
                if ('tv' not in infos['goal_name']):  # don't erode TV, too thin
                    for _ in range(self.args.goal_erode):
                        temp_goal = skimage.morphology.binary_erosion(temp_goal.astype(bool)).astype(float)
                    temp_goal = skimage.morphology.binary_dilation(temp_goal.astype(bool)).astype(float)
                    
                temp_goal *= (torch.sum(self.local_map[4:10], dim=0) - self.local_map[cn]).cpu().numpy() == 0

                if temp_goal.sum() != 0.:
                    self.goal_map = temp_goal
                    self.found_goal = 1


    def inc_step(self):
        """Increase step counters."""
        args = self.args
        self.l_step += 1
        self.step += 1
        self.l_step = self.step % args.num_local_steps
