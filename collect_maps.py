import argparse
import os
import random
import habitat
import torch
import sys
import cv2
from arguments import get_args
from habitat.core.env import Env
from constants import hm3d_names
import numpy as np
import matplotlib.pyplot as plt
from constants import color_palette, object_category
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

from agent.peanut_agent import PEANUT_Agent

def shuffle_episodes(env, shuffle_interval):
    ranges = np.arange(0, len(env.episodes), shuffle_interval)
    np.random.shuffle(ranges)
    new_episodes = []
    for r in ranges:
        new_episodes += env.episodes[r:r + shuffle_interval]
    env.episodes = new_episodes

def d240(x):
    if x < 240:
        x = x + 2*(240-x)
    elif x >= 240:
        x = x - 2*(x-240)
    return x

def _visualize(args, inputs, agents_seg_list, step_i):

    map_pred = inputs['obstacle']
    exp_pred = inputs['exp_pred']
    sem_map = inputs['sem_map_pred'] #[960,960]

    no_cat_mask = sem_map == args.num_sem_categories + 4
    map_mask = np.rint(map_pred) == 1
    exp_mask = np.rint(exp_pred) == 1

    sem_map[no_cat_mask] = 0
    m1 = np.logical_and(no_cat_mask, exp_mask)
    sem_map[m1] = 2

    m2 = np.logical_and(no_cat_mask, map_mask)
    sem_map[m2] = 1

    sem_map -= 2



    color_pal = [int(x * 255.) for x in color_palette]
    sem_map_vis = Image.new("P", (sem_map.shape[1],
                                    sem_map.shape[0]))
    sem_map_vis.putpalette(color_pal)
    sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
    sem_map_vis = sem_map_vis.convert("RGB")
    sem_map_vis = np.flipud(sem_map_vis)

    sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
    sem_map_vis = cv2.resize(sem_map_vis, (960, 960),
                                interpolation=cv2.INTER_NEAREST)
    # sem_map_vis = cv2.resize(sem_map_vis, (256, 256),
    #                             interpolation=cv2.INTER_NEAREST)
    color_blue = (255,0,0)
    for key, value in agents_seg_list.items():
        # 将每个 value 转换成适合 cv2.polylines 使用的格式（一个 numpy 数组）
        for array in value:
            pts = array.reshape((-1, 1, 2))*2
            for i in pts:
                for j in i:
                    # j[0] = d240(j[0])
                    j[1] = d240(j[1])
            # 绘制多边形
            # cv2.polylines(sem_map_vis, [pts], isClosed=True, color=color_blue, thickness=2)
            
            # 标注key值，文本位置选在多边形的第一个坐标处
            text_position = (pts[0][0][0], pts[0][0][1])
            # moments = cv2.moments(pts)
            # cX = int(moments["m10"] / moments["m00"])
            # cY = int(moments["m01"] / moments["m00"])
            # cv2.putText(sem_map_vis, key, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            

    fn = 'a-Vis3-{}.jpg'.format(step_i)
    cv2.imwrite(fn, sem_map_vis, [cv2.IMWRITE_JPEG_QUALITY, 100])

def Objects_Extract(full_map_pred, use_sam=True):

    semantic_map = full_map_pred[4:]

    dst = np.zeros(semantic_map[0, :, :].shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7))

    Object_list = {}
    Object = []
    for i in range(len(semantic_map)):
        if semantic_map[i, :, :].sum() != 0:
            Single_object_list = []
            se_object_map = semantic_map[i, :, :].cpu().numpy()
            se_object_map[se_object_map>0.1] = 1
            se_object_map = cv2.morphologyEx(se_object_map, cv2.MORPH_CLOSE, kernel)
            contours, hierarchy = cv2.findContours(cv2.inRange(se_object_map,0.1,1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            
            # print("contours:",contours)
            # color_blue = (255,0,0)
            # for value in contours:
            #     # 将每个 value 转换成适合 cv2.polylines 使用的格式（一个 numpy 数组）
            #     pts = value.reshape((-1, 1, 2))
            #     # for i in pts:
            #     #     for j in i:
            #     #         j[0] = d240(j[0])
            #     #         j[1] = d240(j[1])
            #     # 绘制多边形
            #     cv2.polylines(se_object_map, [pts], isClosed=True, color=color_blue, thickness=2)
            
            # fn = 'B-Vis-1.jpg'
            # cv2.imwrite(fn, se_object_map, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    
            for cnt in contours:
                if len(cnt) >= 5:
                    epsilon = 0.05 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    Single_object_list.append(approx)
                    cv2.polylines(dst, [approx], True, 1)
            if len(Single_object_list) > 0:
                if use_sam:
                    Object_list[object_category[i]] = Single_object_list
                else:
                    pass
            Object.append(object_category[i])
    return Object_list, Object

def main():

    args_2 = get_args()
    args_2.only_explore = 1  
    args_2.switch_step = 999
    args_2.global_downscaling = 4
    
    # config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config_paths = 'configs/challenge_objectnav2022.local.rgbd.yaml'
    config = habitat.get_config(config_paths=config_paths)
    config.defrost()
    config.SIMULATOR.SCENE_DATASET = 'data/scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json'
    config.SEED = 100
    # config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args_2.sem_gpu_id
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = 50
    config.DATASET.SPLIT = 'train'
    # config.DATASET.SPLIT = 'val'
    config.freeze()
 
    nav_agent = PEANUT_Agent(args=args_2,task_config=config)
    hab_env = Env(config=config)
    print(hab_env.episode_iterator._max_rep_step)
    print(len(hab_env.episodes), 'episodes in dataset')
    
    num_episodes = 4000
    start = args_2.start_ep
    end = args_2.end_ep if args_2.end_ep > 0 else num_episodes
    
    save_steps = list(range(25, 525, 25))
    succs, spls, dtgs, epls = [], [], [], []
    
    count_episodes = 0
    while count_episodes < min(num_episodes, end):
        observations = hab_env.reset()
        observations['objectgoal'] = [0]
        nav_agent.reset()
        print(count_episodes, hab_env._current_episode.scene_id)
        sys.stdout.flush()
        
        if count_episodes >= start and count_episodes < end:

            step_i = 0
            seq_i = 0
            full_map_seq = np.zeros((len(save_steps), 4 + args_2.num_sem_categories, 
                                     nav_agent.agent_states.full_w, nav_agent.agent_states.full_h), dtype=np.uint8)
            while not hab_env.episode_over:
                sys.stdout.flush()
                action = nav_agent.act(observations)
                observations = hab_env.step(action)
                observations['objectgoal'] = [0]
                          
                if step_i % 100 == 0:
                    print('episode %d, step %d' % (count_episodes, step_i))
                    sys.stdout.flush()

                step_i += 1
                # if step_i >= 450:
                #     exit()
                # full_map = nav_agent.agent_states.full_map
                full_map1 = [nav_agent.agent_states.local_map]
                full_map2 = torch.cat([fm.unsqueeze(0) for fm in full_map1], dim=0)
                full_map_pred, object_list = torch.max(full_map2, 0)
                agents_seg_list, object_list = Objects_Extract(full_map_pred)
                # print(agents_seg_list)
                # print("obj_list:",object_list)
                if step_i in save_steps:
                    full_map = nav_agent.agent_states.full_map.cpu().numpy() * 255
                    # full_map = nav_agent.agent_states.local_map.cpu().numpy() * 255

                    p_input = {}
                    p_input['obstacle'] = full_map[0, :, :]
                    p_input['exp_pred'] = full_map[1, :, :]
                    p_input['sem_map_pred'] = full_map[4: ,: , :].argmax(0)

                    # full_map_seq[seq_i] = full_map.astype(np.uint8)
                    seq_i += 1
                    # print(full_map.shape) #[14,960,960]
                    # print(full_map[4: ,: , :].argmax(0).shape)

                    # _visualize(args_2, p_input, agents_seg_list, step_i)
                    
                    # print(full_map_seq.shape) #[20,14,960,960]

                    # if step_i == 100:
                    #     exit()

                # exit()
                
            # if np.sum(full_map_seq[:, 4:]) > 0 and np.sum(full_map_seq[:, 1]) > 4000:
                # _visualize(p_input, agents_seg_list, step_i)
                # exit()
                # np.savez_compressed('./data/saved_maps/%s_80/f%05d.npz' % (config.DATASET.SPLIT, count_episodes), maps=full_map_seq)

        count_episodes += 1
        

if __name__ == "__main__":
    main()
