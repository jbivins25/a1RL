import gym
from gym import error, spaces, utils
from gym.utils import seeding

#Use PyBullet to simulate the A1 robot
import pybullet as p
import time
import pybullet_data as pd
import numpy as np

class Robot():
    def __init__(self,client):
        self.robot = p.loadURDF("a1/a1.urdf",[0,0,0.5],physicsClientId=client)

        A1_DEFAULT_ABDUCTION_ANGLE = 0
        A1_DEFAULT_HIP_ANGLE = 0.9
        A1_DEFAULT_KNEE_ANGLE = -1.8
        NUM_LEGS = 4
        self.INIT_MOTOR_ANGLES = np.array([
            A1_DEFAULT_ABDUCTION_ANGLE,
            A1_DEFAULT_HIP_ANGLE,
            A1_DEFAULT_KNEE_ANGLE
        ] * NUM_LEGS)

        MOTOR_NAMES = [
            "FR_hip_joint",
            "FR_upper_joint",
            "FR_lower_joint",
            "FL_hip_joint",
            "FL_upper_joint",
            "FL_lower_joint",
            "RR_hip_joint",
            "RR_upper_joint",
            "RR_lower_joint",
            "RL_hip_joint",
            "RL_upper_joint",
            "RL_lower_joint",
        ]
        self.motor_ids = []

        for j in range (p.getNumJoints(self.robot)):
            joint_info = p.getJointInfo(self.robot,j,physicsClientId=client)
            name = joint_info[1].decode('utf-8')
            print("joint_info[1]=",name)
            if name in MOTOR_NAMES:
                self.motor_ids.append(j)

        for index in range (12):
            joint_id = self.motor_ids[index]
            p.setJointMotorControl2(self.robot, joint_id, p.POSITION_CONTROL, self.INIT_MOTOR_ANGLES[index],physicsClientId=client)
            
    def getID(self):
        return self.robot
        
    def getMotorID(self):
        return self.motor_ids

class A1Env(gym.Env):

  def __init__(self):
    self.client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())
    p.setGravity(0,0,-9.8)
    self._elapsed_steps = 0

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
    p.loadURDF("plane.urdf")
    
    # Observation Space
    # Looks at x,y coordinates of agent
    # Looks at the base orientation of the agent for falling
    # Looks at the angle for each joint
    # All of these are arbitrarily guessed
    self.observation_space = gym.spaces.box.Box(
        low = np.array([-30,-30,-360,-180,-180,-180,-180,-180,-180,-180,-180,-180,-180,-180,-180]),
        high = np.array([30,30,360,180,180,180,180,180,180,180,180,180,180,180,180])
    )
    
    # Action Space
    # Continous action space, each dimension is correlated to one joint angle
    # Defining the lower limit for changing each angle as -5 degrees
    # Defining the upper limit for changing each angle as 5 degrees
    self.action_space = gym.spaces.box.Box(
        low = np.array([-5.,-5.,-5.,-5.,-5.,-5.,-5.,-5.,-5.,-5.,-5.,-5.]),
        high = np.array([5.,5.,5.,5.,5.,5.,5.,5.,5.,5.,5.,5.])
    )
        
    # Seed for identical demonstrations
    self.np_random, _ = gym.utils.seeding.np_random()
    
    
    self.a1Bot = Robot(self.client)
    self.startState = p.saveState()
    print("In init")
    

  def step(self, action):
    assert( self.action_space.contains(action), "%r (%s) invalid" % (action, type(action) ) )
    
    robotID = self.a1Bot.getID()
    motor_ids = self.a1Bot.getMotorID()
    for index in range(12):
        joint_id = motor_ids[index]
        p.setJointMotorControl2(robotID, joint_id, p.POSITION_CONTROL, action[index])
    observation = self._get_obs()
    reward = 0
    if observation[2] > 90 and observation[2] < 270:
        done = True
    else:
        done = False
    self._elapsed_steps += 1
    if self._elapsed_steps >= 500:
        done = True
    return observation, reward, done, {}

  # This is the function called to return the observation
  # This gets:
  # The x,y coordinates (technically x,z) of the robot base
  # The orienatation of the base
  # The position of each joint
  def _get_obs(self):
    robotID = self.a1Bot.getID()
    robotPandO = p.getBasePositionAndOrientation(robotID)
    x = robotPandO[0][0]
    y = robotPandO[0][2]
    orientation = (p.getEulerFromQuaternion(robotPandO[1]))[0]
    motorIDs = self.a1Bot.getMotorID()
    jointInfo = p.getJointStates(robotID,motorIDs)
    j0, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11 = [jointInfo[i][0] for i in range(12)]
    return np.array([x,y,orientation,j0,j1,j2,j3,j4,j5,j6,j7,j8,j9,j10,j11],dtype= np.float32)

  def reset(self):
    print("In reset")
    p.resetSimulation(self.client)
    p.setGravity(0,0,-9.8)
    p.loadURDF("plane.urdf")
    self.robot = Robot(self.client)
    observation = self._get_obs()
    return (observation,{})
    
  def close(self):
    p.disconnect(self.client)
    
  def seed(self, seed = None):
    self.np_random, seed = gym.utils.seeding.np_random(seed)
    return [seed]
