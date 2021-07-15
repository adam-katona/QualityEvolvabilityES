#Copyright (c) 2019 Uber Technologies, Inc.
#
#Licensed under the Uber Non-Commercial License (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at the root directory of this project. 
#
#See the License for the specific language governing permissions and
#limitations under the License.
import gym
import numpy as np

from .bullet import BulletEnv


class AntEnv(BulletEnv):
    def __init__(self, seed=None,environment_mode=None):
        super().__init__("AntBulletEnv-v0", seed)
        self.novelty_space = gym.spaces.Box(shape=(2,), low=-100, high=100)
        self.coord = "x"
        self.direction = "+"
        self.environment_mode = environment_mode


    def render(self, mode="human"):
        # TODO we might want to follow the robot with the camera, or zoom out or something
        # to zoom out: env.env.env._cam_dist = 10
        return self.env.render(mode="rgb_array")

    def add_trap_boxes(self):
    
        # From novelty seeking agents paper, this is the trap geometry they used:
        #<geom name="frontwall" type="box" pos="5.5 0 0" size=".1 1.5 2"/>
        #<geom name="topwall" type="box" pos="4 1.5 0" size="1.5 .1 2"/>
        #<geom name="bottomwall" type="box" pos="4 -1.5 0" size="1.5 .1 2"/>

        #Let us use the same 
    
        # get the bullet client
        p = self.env.env.robot._p
        box_1_id = p.createCollisionShape(p.GEOM_BOX,halfExtents=[0.1, 1.5, 2])
        box_2_id = p.createCollisionShape(p.GEOM_BOX,halfExtents=[1.5, 0.1, 2])
        box_3_id = p.createCollisionShape(p.GEOM_BOX,halfExtents=[1.5, 0.1, 2])
        
        p.createMultiBody(baseMass=0,
                    baseCollisionShapeIndex=box_1_id,
                    basePosition=[5.5, 0, 0])
        p.createMultiBody(baseMass=0,
                    baseCollisionShapeIndex=box_2_id,
                    basePosition=[4, 1.5, 0])
        p.createMultiBody(baseMass=0,
                    baseCollisionShapeIndex=box_3_id,
                    basePosition=[4, -1.5, 0])



    def add_large_trap(self):

        p = self.env.env.robot._p
        box_1_id = p.createCollisionShape(p.GEOM_BOX,halfExtents=[0.1, 2, 2])
        box_2_id = p.createCollisionShape(p.GEOM_BOX,halfExtents=[2, 0.1, 2])
        box_3_id = p.createCollisionShape(p.GEOM_BOX,halfExtents=[2, 0.1, 2])

        p.createMultiBody(baseMass=0,
                    baseCollisionShapeIndex=box_1_id,
                    basePosition=[6, 0, 0])  # this is kind of an error, this should be 6.5, well works this way too
        p.createMultiBody(baseMass=0,
                    baseCollisionShapeIndex=box_2_id,
                    basePosition=[4.5, 2, 0])
        p.createMultiBody(baseMass=0,
                    baseCollisionShapeIndex=box_3_id,
                    basePosition=[4.5, -2, 0])


    # override reset to handle the reset error because the added shapes.
    def reset(self):
        if self.environment_mode == "DECEPTIVE":
            # recreate the env from scratch, this is necessary becuase env.reset() fails when we add collision boxes
            self.env = gym.make("AntBulletEnv-v0")
            if self._seed is not None:
                self.env.seed(self._seed)
            unused_image = self.env.render(mode="rgb_array") 
            obs = self.env.reset()
            #self.add_trap_boxes()
            self.add_large_trap()
        else:
            if self._seed is not None:
                self.env.seed(self._seed)
            obs = self.env.reset()

        return obs

    def step(self, action):
        state, _reward, done, info = self.env.step(action)
        x, y, _z = self.env.unwrapped.robot.body_xyz
        # dist = (x ** 2 + y ** 2) ** 0.5
        if done:
            reward = x if self.coord == "x" else y
            if self.direction == "-":
                reward = -reward
        else:
            reward = 0.0
        return state, reward, done, info

    def novelty(self):
        return self.env.unwrapped.robot.body_xyz[:2]

    def set_target(self, coord, direction):
        self.coord = coord
        self.direction = direction



# every reset, a random direction is determined
# The direction is part of the observation, and the fitness is the distance moved in that direction.
# The policy must learn to go in a specified direction (which is contant for an episode)
class DirectionalAnt(BulletEnv):
    def __init__(self, seed=None):
        super().__init__("AntBulletEnv-v0", seed)
        self.novelty_space = gym.spaces.Box(shape=(2,), low=-100, high=100)
        
        self.observation_space = gym.spaces.Box(shape=(self.env.observation_space.shape[0] + 2,),low=self.env.observation_space.low[0],high=self.env.observation_space.high[0])

        # We have 8 possible directions
        # There is no [0,0]
        self.all_directions = [ [1,0],
                                [1,1],
                                [0,1],
                                [-1,1],
                                [-1,0],
                                [-1,-1],
                                [0,-1],
                                [1,-1] ]

        self.current_episode_direction = None

    def step(self, action):
        state, _reward, done, info = self.env.step(action)
        x, y, _z = self.env.unwrapped.robot.body_xyz
        # dist = (x ** 2 + y ** 2) ** 0.5
        if done:
            
            direction = np.array(self.current_episode_direction)
            length = np.sqrt((direction ** 2).sum())
            direction = direction / length  # make sure it is unit length

            # take the distance traveled in the selected direction
            final_position = np.array([x,y])
            reward = np.dot(final_position,direction)
        else:
            reward = 0.0 
            
        # add the desired direction to the obervations
        state = np.concatenate([state,np.array(self.current_episode_direction)])  

        return state, reward, done, info


    def reset(self,fixed_dir_index=None):
        # if fixed_dir is None, it means we choose one randomly
        if fixed_dir_index is not None:
            self.current_episode_direction = self.all_directions[fixed_dir_index]
        else:
            self.current_episode_direction = self.all_directions[np.random.choice(len(self.all_directions))]
        state = super(DirectionalAnt, self).reset()
        state = np.concatenate([state,np.array(self.current_episode_direction)])  
        return state
        
    def novelty(self):
        return self.env.unwrapped.robot.body_xyz[:2]