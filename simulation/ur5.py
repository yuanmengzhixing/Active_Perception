import numpy as np
import math
import time

class UR5():

    def __init__(self):
        self.clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5) 
        self.RAD2DEG = 180 / math.pi   # 常数，弧度转度数
        self.tstep = 0.005             # 定义仿真步长
        self.cubeName = 'UR5_ikTarget'
        self.targetPosition=np.zeros(3,dtype=np.float)#目标位置
        self.targetQuaternion=np.zeros(4)

    def connect(self):  #连接v-rep
        print('Program started') # 关闭潜在的连接 
        vrep.simxFinish(-1) # 每隔0.2s检测一次，直到连接上V-rep 
        while True:
            self.clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5) 
            if self.clientID > -1: 
                break 
            else: 
                time.sleep(0.2) 
                print("Failed connecting to remote API server!") 
        print("Connection success!")
        vrep.simxSetFloatingParameter(self.clientID, vrep.sim_floatparam_simulation_time_step, self.tstep, vrep.simx_opmode_oneshot) # 保持API端与V-rep端相同步长
        vrep.simxSynchronous(self.clientID, True) # 然后打开同步模式 
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot) 

    def disconnect(self):
        vrep.simxStopSimulation(self.clientID,vrep.simx_opmode_oneshot)
        vrep.simxFinish(self.clientID)
        print ('Program ended!')
        
    def get_clientID(self):
        return self.clientID

    def ur5moveto(self,x,y,z):
        vrep.simxSynchronousTrigger(self.clientID) # 让仿真走一步 
        self.targetQuaternion[0]=0.707
        self.targetQuaternion[1]=0
        self.targetQuaternion[2]=0.707
        self.targetQuaternion[3]=0      #四元数
        self.targetPosition[0]=x;
        self.targetPosition[1]=y;
        self.targetPosition[2]=z;
        vrep.simxPauseCommunication(self.clientID, True)    #开启仿真
        vrep.simxSetIntegerSignal(self.clientID, 'ICECUBE_0', 21, vrep.simx_opmode_oneshot)
        for i in range(3):
            vrep.simxSetFloatSignal(self.clientID, 'ICECUBE_'+str(i+1),self.targetPosition[i],vrep.simx_opmode_oneshot)
        for i in range(4):
            vrep.simxSetFloatSignal(self.clientID, 'ICECUBE_'+str(i+4),self.targetQuaternion[i], vrep.simx_opmode_oneshot)
        vrep.simxPauseCommunication(self.clientID, False)
        vrep.simxSynchronousTrigger(self.clientID) # 进行下一步 
        vrep.simxGetPingTime(self.clientID) # 使得该仿真步走完

    def grasp(self):
        vrep.simxSetIntegerSignal(self.clientID,'RG2CMD',1,vrep.simx_opmode_blocking);

    def lose(self):
        vrep.simxSetIntegerSignal(self.clientID,'RG2CMD',0,vrep.simx_opmode_blocking);


