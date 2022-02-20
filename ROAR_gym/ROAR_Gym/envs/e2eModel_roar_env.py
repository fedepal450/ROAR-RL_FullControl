try:
    from ROAR_Gym.envs.roar_env import ROAREnv
except:
    from ROAR_Gym.ROAR_Gym.envs.roar_env import ROAREnv

from ROAR.utilities_module.vehicle_models import VehicleControl
from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.vehicle_models import Vehicle
from typing import Tuple
import numpy as np
from typing import List, Any
import gym
import math
from collections import OrderedDict
from gym.spaces import Discrete, Box
import cv2
import wandb

mode='baseline'
if mode=='no_map':
    FRAME_STACK = 1
else:
    FRAME_STACK = 4

if mode=='baseline':
    CONFIG = {
        "x_res": 84,
        "y_res": 84
    }
else:
    CONFIG = {
        # max values are 280x280
        # original values are 80x80
        "x_res": 80,
        "y_res": 80
    }



class ROARppoEnvE2E(ROAREnv):
    def __init__(self, params):
        super().__init__(params)
        #self.action_space = Discrete(len(DISCRETE_ACTIONS))
        low=np.array([-2.5, -5.0, 1.0])
        high=np.array([-0.5, 5.0, 3.0])
        # low=np.array([100, 0, -1])
        # high=np.array([1, 0.12, 0.5])
        self.mode=mode
        if self.mode=='baseline':
            self.action_space = Box(low=low, high=high, dtype=np.float32)
        else:
            self.action_space = Box(low=np.tile(low,(FRAME_STACK)), high=np.tile(high,(FRAME_STACK)), dtype=np.float32)

        if self.mode=='no_map':
            self.observation_space = Box(low=np.tile([-1],(13)), high=np.tile([1],(13)), dtype=np.float32)
        elif self.mode=='combine':
            self.observation_space = Box(-10, 1, shape=(FRAME_STACK,3, CONFIG["x_res"], CONFIG["y_res"]), dtype=np.float32)
        elif self.mode=='baseline':
            self.observation_space = Box(-10, 1, shape=(FRAME_STACK,3, CONFIG["x_res"], CONFIG["y_res"]), dtype=np.float32)
        else:
            self.observation_space = Box(-10, 1, shape=(FRAME_STACK, CONFIG["x_res"], CONFIG["y_res"]), dtype=np.float32)
        self.prev_speed = 0
        self.prev_cross_reward = 0
        self.crash_check = False
        self.ep_rewards = 0
        self.frame_reward = 0
        self.highscore = -1000
        self.highest_chkpt = 0
        self.speeds = []
        self.prev_int_counter = 0
        self.steps=0
        self.largest_steps=0
        self.highspeed=0
        self.complete_loop=False
        self.his_checkpoint=[]
        self.his_score=[]

    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        obs = []
        rewards = []
        self.steps+=1
        for i in range(1):
            # throttle=(action[i*3+0]+0.5)/2+1
            throttle=0.8#(action[i*3+2]-1)/2==0
            # target_steering=action[i*3+1]/10
            # cur_steering=self.agent.vehicle.control.steering
            # steering=max(min(cur_steering+0.1,target_steering),cur_steering-0.1)
            # steering=np.sign(action[i*3+1])*max(0,(abs(action[i*3+1])-0.5))/4
            steering=action[i*3+1]/5
            braking=0#(action[i*3+2]-1)/2
            # throttle=min(max(action[i*3+0],0),1)
            # steering=min(max(action[i*3+1],-1),1)
            # braking=min(max(action[i*3+2],0),1)
            self.agent.kwargs["control"] = VehicleControl(throttle=throttle,
                                                          steering=steering,
                                                          braking=braking)
            ob, reward, is_done, info = super(ROARppoEnvE2E, self).step(action)
            obs.append(ob)
            rewards.append(reward)
            if is_done:
                break
        self.render()
        self.frame_reward = sum(rewards)
        self.ep_rewards += sum(rewards)
        if is_done:
            self.wandb_logger()
            self.crash_check = False
            self.update_highscore()
        return np.array(obs), self.frame_reward, self._terminal(), self._get_info()

    def _get_info(self) -> dict:
        info_dict = OrderedDict()
        info_dict["Current HIGHSCORE"] = self.highscore
        info_dict["Furthest Checkpoint"] = self.highest_chkpt*self.agent.interval
        info_dict["episode reward"] = self.ep_rewards
        info_dict["checkpoints"] = self.agent.int_counter*self.agent.interval
        info_dict["reward"] = self.frame_reward
        info_dict["largest_steps"] = self.largest_steps
        info_dict["highest_speed"] = self.highspeed
        info_dict["complete_state"]=self.complete_loop
        info_dict["avg10_checkpoints"]=np.average(self.his_checkpoint)
        info_dict["avg10_score"]=np.average(self.his_score)
        # info_dict["throttle"] = action[0]
        # info_dict["steering"] = action[1]
        # info_dict["braking"] = action[2]
        return info_dict

    def update_highscore(self):
        if self.ep_rewards > self.highscore:
            self.highscore = self.ep_rewards
        if self.agent.int_counter > self.highest_chkpt:
            self.highest_chkpt = self.agent.int_counter
        if self.agent.vehicle.get_speed(self.agent.vehicle)>self.highspeed:
            self.highspeed=self.agent.vehicle.get_speed(self.agent.vehicle)
        return

    def _terminal(self) -> bool:
        if self.carla_runner.get_num_collision() > self.max_collision_allowed:
            # crash_rep = open("crash_spot.txt", "a")
            # loc = np.array([self.agent.vehicle.transform.location.x, self.agent.vehicle.transform.location.y, self.agent.vehicle.transform.location.z])
            # np.savetxt(crash_rep, loc, delimiter=',')
            # crash_rep.close()
            return True
        elif self.agent.finish_loop:
            self.complete_loop=True
            return True
        else:
            return False

    def get_reward(self) -> float:
        # prep for reward computation
        # reward = -0.1*(1-self.agent.vehicle.control.throttle+10*self.agent.vehicle.control.braking+abs(self.agent.vehicle.control.steering))*400/8
        reward=-1
        curr_dist_to_strip = self.agent.curr_dist_to_strip

        if self.crash_check:
            return 0
        # reward computation
        # current_speed = self.agent.bbox_list[self.agent.int_counter%len(self.agent.bbox_list)].get_directional_velocity(self.agent.vehicle.velocity.x,self.agent.vehicle.velocity.y)
        # current_speed = self.agent.vehicle.get_speed()
        # self.speeds.append(current_speed)

        if self.agent.cross_reward > self.prev_cross_reward:
            # num_crossed = self.agent.int_counter - self.prev_int_counter
            #speed reward
            # reward+= np.average(self.speeds) * num_crossed/8
            # self.speeds=[]
            # self.prev_int_counter =self.agent.int_counter
            #crossing reward
            reward += (self.agent.cross_reward - self.prev_cross_reward)*self.agent.interval*4

        if self.carla_runner.get_num_collision() > 0:
            reward -= 200#0# /(min(total_num_cross,10))
            self.crash_check = True

        # log prev info for next reward computation
        self.prev_speed = Vehicle.get_speed(self.agent.vehicle)
        self.prev_cross_reward = self.agent.cross_reward
        return reward

    def _get_obs(self) -> np.ndarray:
        if mode=='baseline':
            # vehicle_state=self.agent.vehicle.to_array() #12
            # line_location=self.agent.bbox.to_array(vehicle_state[3],vehicle_state[5]) #4
            # v_speed=np.sqrt(np.square(vehicle_state[0])+np.square(vehicle_state[1]))/150
            # v_height=vehicle_state[4]/100
            # v_roll,v_pitch,v_yaw=vehicle_state[[6,7,8]]/180
            # v_throttle,v_steering,v_braking=vehicle_state[[9,10,11]]
            # x_dis,y_dis,xy_dis=line_location[:3]/40
            # l_yaw,vtol_yaw=line_location[3:]
            # data=np.array([v_speed,v_height,v_roll,v_pitch,v_yaw,v_throttle,v_steering,v_braking,x_dis,y_dis,xy_dis,l_yaw,vtol_yaw])
            # l=len(self.agent.bbox_list)
            index_from=(self.agent.int_counter%len(self.agent.bbox_list))
            if index_from+10<=len(self.agent.bbox_list):
                # print(index_from,len(self.agent.bbox_list),index_from+10-len(self.agent.bbox_list))
                next_bbox_list=self.agent.bbox_list[index_from:index_from+10]
            else:
                # print(index_from,len(self.agent.bbox_list),index_from+10-len(self.agent.bbox_list))
                next_bbox_list=self.agent.bbox_list[index_from:]+self.agent.bbox_list[:index_from+10-len(self.agent.bbox_list)]
            assert(len(next_bbox_list)==10)
            map_list = self.agent.occupancy_map.get_map_baseline(transform_list=self.agent.vt_queue,
                                                    view_size=(CONFIG["x_res"], CONFIG["y_res"]),
                                                    bbox_list=self.agent.frame_queue,
                                                                 next_bbox_list=next_bbox_list
                                                    )
            # data = cv2.resize(occu_map, (CONFIG["x_res"], CONFIG["y_res"]), interpolation=cv2.INTER_AREA)
            #cv2.imshow("Occupancy Grid Map", cv2.resize(np.float32(data), dsize=(500, 500)))

            # data_view=np.sum(data,axis=2)
            cv2.imshow("data", np.hstack(np.hstack(map_list))) # uncomment to show occu map
            cv2.waitKey(1)
            # yaw_angle=self.agent.vehicle.transform.rotation.yaw
            # velocity=self.agent.vehicle.get_speed(self.agent.vehicle)
            # data[0,0,2]=velocity
            # map_input=map_list[:,0]
            # map_input*=255
            # waypoint=np.sum(map_list[:,1:3],axis=1)
            # waypoint*=255
            # data_input=np.zeros_like(map_list)
            # data_input[0,:13]=data
            #print(map_list[:,:-1].shape)
            return map_list[:,:-1]

        else:
            data = self.agent.occupancy_map.get_map(transform=self.agent.vehicle.transform,
                                                    view_size=(CONFIG["x_res"], CONFIG["y_res"]),
                                                    arbitrary_locations=self.agent.bbox.get_visualize_locs(),
                                                    arbitrary_point_value=self.agent.bbox.get_value(),
                                                    vehicle_velocity=self.agent.vehicle.velocity,
                                                    # rotate=self.agent.bbox.get_yaw()
                                                    )
            # data = cv2.resize(occu_map, (CONFIG["x_res"], CONFIG["y_res"]), interpolation=cv2.INTER_AREA)
            #cv2.imshow("Occupancy Grid Map", cv2.resize(np.float32(data), dsize=(500, 500)))

            # data_view=np.sum(data,axis=2)
            cv2.imshow("data", data) # uncomment to show occu map
            cv2.waitKey(1)
            # yaw_angle=self.agent.vehicle.transform.rotation.yaw
            # velocity=self.agent.vehicle.get_speed(self.agent.vehicle)
            # data[0,0,2]=velocity
            data_input=data.copy()
            data_input[data_input==1]=-10
            return data_input  # height x width x 3 array
    #3location 3 rotation 3velocity 20 waypoline locations 20 wayline rewards

    def reset(self) -> Any:
        if len(self.his_checkpoint)>=10:
            self.his_checkpoint=self.his_checkpoint[-10:]
            self.his_score=self.his_score[-10:]
        if self.agent:
            self.his_checkpoint.append(self.agent.int_counter*self.agent.interval)
            self.his_score.append(self.ep_rewards)
        self.ep_rewards = 0
        if self.steps>self.largest_steps and not self.complete_loop:
            self.largest_steps=self.steps
        elif self.complete_loop and self.agent.finish_loop and self.steps<self.largest_steps:
            self.largest_steps=self.steps
        super(ROARppoEnvE2E, self).reset()
        self.steps=0
        return self._get_obs()

    def wandb_logger(self):
        wandb.log({
            "episode reward": self.ep_rewards,
            "global_step":self.steps,
            "Furthest Checkpoint" : self.highest_chkpt*self.agent.interval,
            "Current HIGHSCORE" : self.highscore,
            "largest_steps" : self.largest_steps,
            "highest_speed" : self.highspeed,
            "avg10_checkpoints":np.average(self.his_checkpoint),
            "avg10_score":np.average(self.his_score)

        })
        return