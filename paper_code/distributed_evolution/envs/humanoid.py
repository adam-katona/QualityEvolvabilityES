import gym
import numpy as np
import pybullet_envs  # pylint: disable=unused-import
import pybullet

# to avoid generating GB-s of warning in the stdout logfiles
gym.logger.set_level(gym.logger.ERROR)

class NoveltyHumanoid(gym.Env):

    #################################
    # The official gym API functions
    #################################

    def __init__(self,environment_mode,use_built_in_rewards,seed=None):

        # environment_mode can be:
        # NORMAL        # objective move forward, normal env
        # DECEPTIVE     # objective move forward, there is a trap to make the problem deceptive
        # DIRECTIONAL   # objective go in randomly selected direction
        self.environment_mode = environment_mode
        self._seed = seed
        self.use_built_in_rewards = use_built_in_rewards

        # used by DIRECTIONAL
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

        obs = self.reset()

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        if self.environment_mode == "DIRECTIONAL":
            self.observation_space = gym.spaces.Box(shape=(self.env.observation_space.shape[0] + 2,),low=self.env.observation_space.low[0],high=self.env.observation_space.high[0])
            
            
        self.novelty_space = gym.spaces.Box(shape=(2,), low=-1000, high=1000)

        self.episode_rewards = {
            "alive" : 0.0,
            "progress" : 0.0,
            "electricity" : 0.0,
            "joint_at_limit" : 0.0,
            "feet_collision" : 0.0
        }

        

    def reset(self):
        
        # reset accumulated rewards
        self.episode_rewards = {
            "alive" : 0.0,
            "progress" : 0.0,
            "electricity" : 0.0,
            "joint_at_limit" : 0.0,
            "feet_collision" : 0.0
        }

        # recreate the env from scratch, this is necessary becuase env.reset() fails when we add collision boxes
        self.env = gym.make("HumanoidBulletEnv-v0")

        # render() needs to be called before reset() for some reason...
        # reset() initializes a bunch of stuff like bullet_client  (WTF is this lazy init??)
        unused_image = self.env.render(mode="rgb_array") 
        obs = self.env.reset()

        if self.environment_mode == "DIRECTIONAL":
            self.current_episode_direction = self.all_directions[np.random.choice(len(self.all_directions))]
            obs = np.concatenate([obs,np.array(self.current_episode_direction)])  

        if self.environment_mode == "DECEPTIVE":
            self.add_trap_boxes()

        return obs



    def render(self, mode="human"):
        # TODO we might want to follow the robot with the camera, or zoom out or something
        # to zoom out: env.env.env._cam_dist = 10
        return self.env.render(mode="rgb_array")

    def step(self,action):
        obs, _reward, done, info = self.env.step(action) 
        robot_position = self.env.env.robot.robot_body.get_position()[0:2]

        # HANDLEING OBSERVATION
        if self.environment_mode == "DIRECTIONAL":
           obs = np.concatenate([obs,np.array(self.current_episode_direction)])   


        # HANDLING REWARDS
        # In Ant and Chetah, the env reward was ignored and instead custom reward was calculated on the last step 
        # Humanoid have all kinds of extra fitness terms, (alive, electricity cost, collision penalty)
        # Do we want to keep these?
        # The way rollout is implemented is it summs up all the rewards for all steps, it would work with the built in reward as well.
        
        all_rewards = self.env.env.rewards
        # self._alive, progress, electricity_cost, joints_at_limit_cost, feet_collision_cost

        self.episode_rewards["alive"] += all_rewards[0]
        self.episode_rewards["progress"] += all_rewards[1]
        self.episode_rewards["electricity"] += all_rewards[2]
        self.episode_rewards["joint_at_limit"] += all_rewards[3]
        self.episode_rewards["feet_collision"] += all_rewards[4]
       
        if done:
            # task fitness
            if self.environment_mode == "DIRECTIONAL":
                direction = np.array(self.current_episode_direction)
                length = np.sqrt((direction ** 2).sum())
                direction = direction / length  # make sure it is unit length

                final_position = np.array(robot_position)
                task_fitness = np.dot(final_position,direction)
            else:
                # progress is measured by moving forward
                task_fitness = robot_position[0]  # x coordinate of the position


            # in the original env, progrees is measured by change in petential which is calculated like:
            # -self.walk_target_dist / self.scene.dt
            # For example initial potential is -1000/0.0165 = -60606
            # Potential after walking 10 meters: -990/0.0165 = -60000
            # Fitness from walking = 606
            # I imagine they devide with dt, because they multiply the alive stuff with dt, so if dt changes, their ratio stays the same
            # To keep the ratios the same we will implemnt our fitnes with the same scale
            progress_scale = 1/self.env.env.scene.dt
            #task_fitness = task_fitness * progress_scale

            # other fitness terms
            # So we want to keep the alive term? Otherwise it might learn to lounge in the end. So let us keep it.
            other_fitnesses = (self.episode_rewards["alive"] + 
                               self.episode_rewards["electricity"] + 
                               self.episode_rewards["joint_at_limit"] + 
                               self.episode_rewards["feet_collision"])

            if self.use_built_in_rewards is True:
                task_fitness = task_fitness * progress_scale
                reward = task_fitness + other_fitnesses
            else:
                reward = task_fitness
        else:
            reward = 0

        return obs, reward, done, info

    def seed(self, seed=None):
        self._seed = seed
        self.env.seed(self._seed)


    #################################
    # Extra API functions
    #################################

    def novelty(self):
        return self.env.env.robot.robot_body.get_position()[0:2]
    

    #################################
    # Internal functions
    #################################


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


