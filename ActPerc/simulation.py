#import sys
#sys.path.append(r'/home/lukai/testVrep/ActPerc')
#from Camera import Camera
#from UR5 import UR5
import os
import time
import numpy.random as random
import numpy as np
import math
import PIL.Image as Image
import array

try:
    import vrep
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')
    print ('')

# Make sure to have the server side running in V-REP: 
# in a child script of a V-REP scene, add following command
# to be executed just once, at simulation start:
#
# simRemoteApi.start(19999)
#
# then start simulation, and run this program.


class Camera():

    def __init__(self, clientID):
        """
            Initialize the Camera in simulation
        """
        # self.RAD2EDG = 180 / math.pi
        self.Save_IMG = True
        self.Save_PATH_COLOR = r'./color/'
        self.Save_PATH_DEPTH = r'./depth/'
        self.Dis_NEAR = 0.01
        self.Dis_FAR = 10
        self.INT16 = 65535
        self.Img_WIDTH = 512
        self.Img_HEIGHT = 424
        self.Camera_NAME = r'kinect'
        self.Camera_RGB_NAME = r'kinect_rgb'
        self.Camera_DEPTH_NAME = r'kinect_depth'
        self.clientID = clientID
        self._setup_sim_camera()

    def _euler2rotm(self,theta):
        """
            -- Get rotation matrix from euler angles
        """
        R_x = np.array([[1,         0,                  0                   ],
                        [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                        [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                        ])
        R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                        [0,                     1,      0                   ],
                        [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                        ])         
        R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                        [math.sin(theta[2]),    math.cos(theta[2]),     0],
                        [0,                     0,                      1]
                        ])            
        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R

    def _setup_sim_camera(self):
        """
            -- Get some param and handles from the simulation scene
            and set necessary parameter for camera
        """
        # Get handle to camera
        _, self.cam_handle = vrep.simxGetObjectHandle(self.clientID, self.Camera_NAME, vrep.simx_opmode_oneshot_wait)
        _, self.kinectRGB_handle = vrep.simxGetObjectHandle(self.clientID, self.Camera_RGB_NAME, vrep.simx_opmode_oneshot_wait)
        _, self.kinectDepth_handle = vrep.simxGetObjectHandle(self.clientID, self.Camera_DEPTH_NAME, vrep.simx_opmode_oneshot_wait)
        # Get camera pose and intrinsics in simulation
        _, cam_position = vrep.simxGetObjectPosition(self.clientID, self.cam_handle, -1, vrep.simx_opmode_oneshot_wait)
        _, cam_orientation = vrep.simxGetObjectOrientation(self.clientID, self.cam_handle, -1, vrep.simx_opmode_oneshot_wait)

        self.cam_trans = np.eye(4,4)
        self.cam_trans[0:3,3] = np.asarray(cam_position)
        self.cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        self.cam_rotm = np.eye(4,4)
        self.cam_rotm[0:3,0:3] = np.linalg.inv(self._euler2rotm(cam_orientation))
        self.cam_pose = np.dot(self.cam_trans, self.cam_rotm) # Compute rigid transformation representating camera pose

    def get_camera_data(self):
        """
            -- Read images data from vrep and convert into np array
        """
        # Get color image from simulation
        res, resolution, raw_image = vrep.simxGetVisionSensorImage(self.clientID, self.kinectRGB_handle, 0, vrep.simx_opmode_oneshot_wait)
        self._error_catch(res)
        color_img = np.array(raw_image, dtype=np.uint8)
        color_img.shape = (resolution[1], resolution[0], 3)
        color_img = color_img.astype(np.float)/255
        color_img[color_img < 0] += 1
        color_img *= 255
        color_img = np.fliplr(color_img)
        color_img = color_img.astype(np.uint8)

        # Get depth image from simulation
        res, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.clientID, self.kinectDepth_handle, vrep.simx_opmode_oneshot_wait)
        self._error_catch(res)
        depth_img = np.array(depth_buffer)
        depth_img.shape = (resolution[1], resolution[0])
        depth_img = np.fliplr(depth_img)
        # zNear = 0.01
        # zFar = 10
        # depth_img = depth_img * (zFar - zNear) + zNear
        depth_img = (depth_img - self.Dis_NEAR)/self.Dis_FAR * self.INT16
        return depth_img, color_img

    def save_image(self, cur_depth, cur_color, currCmdTime, Save_IMG = True):
        """
            -- Save Color&Depth images
        """
        if Save_IMG:
            img = Image.fromarray(cur_color.astype('uint8')).convert('RGB')
            img_path = self.Save_PATH_COLOR + str(currCmdTime) + '_Rgb.png'
            img.save(img_path)
            depth_img = Image.fromarray(cur_depth.astype(np.uint32),mode='I')
            depth_path = self.Save_PATH_DEPTH + str(currCmdTime) + '_Depth.png'
            depth_img.save(depth_path)
            print("Succeed to save current images")
            return depth_path, img_path
        else:
            print("Skip saving images this time")

    def _error_catch(self, res):
        """
            -- Deal with error unexcepted
        """
        if res == vrep.simx_return_ok:
            print ("--- Image Exist!!!")
        elif res == vrep.simx_return_novalue_flag:
            print ("--- No image yet")
        else:
            print ("--- Error Raise")


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


Simu_STEP = 2
Lua_PATH = r'../model/infer.lua'
Save_PATH_RES = r'./affordance/'

def main():
    ur5 = UR5()
    ur5.connect();
    clientID = ur5.get_clientID()
    camera = Camera(clientID)

    lastCmdTime = vrep.simxGetLastCmdTime(clientID)
    vrep.simxSynchronousTrigger(clientID)

    count = 0
    while vrep.simxGetConnectionId(clientID) != -1 and count < Simu_STEP:
        currCmdTime=vrep.simxGetLastCmdTime(clientID)
        dt = currCmdTime - lastCmdTime

        # Camera achieve data
        cur_depth, cur_color = camera.get_camera_data()
        cur_depth_path, cur_img_path = camera.save_image(cur_depth, cur_color, currCmdTime)

        # Call affordance map function
        affordance_cmd = 'th ' + Lua_PATH + ' -imgColorPath ' + cur_img_path + ' -imgDepthPath ' + cur_depth_path + ' -resultPath ' + Save_PATH_RES + str(currCmdTime) + '_results.h5'
        try:
            os.system(affordance_cmd)
            print('-- Successfully create affordance map!')
        except:
            print('!!!!!!!!!!!!!!!!!!!!!!!!  Error occurred during creating affordance map')
        
        # Move ur5 
        location = random.randint(100, 200, (1, 3)) 
        location = location[0] / 400
        print("UR5 Move to %s" %(location))
        ur5.ur5moveto(location[0], location[1], location[2])
        count += 1

        lastCmdTime=currCmdTime
        vrep.simxSynchronousTrigger(clientID)
        vrep.simxGetPingTime(clientID)

    ur5.disconnect()


if __name__ == '__main__':


    main()
