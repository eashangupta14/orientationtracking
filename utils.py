import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import transforms3d.quaternions as quaternions
import transforms3d._gohlketransforms as gotrans
import transforms3d.euler as euler
import jax
import jax.numpy as jnp
from tqdm import tqdm
class dataload():
    'Class to load data'
    def __init__(self,data_type,data_num, folder_loc = None):
        self.data_type = data_type
        self.data_num = data_num
        if folder_loc:
            self.folder_loc = folder_loc
        else:
            self.folder_loc = f'../data/{self.data_type}set/'
        self.sensor = [('cam','cam'), ('imu','imuRaw'), ('vicon','viconRot')]
        self.data = {}

    def __read_data(self,sensor,loc):
        d = []
        fname = f'{self.folder_loc}{sensor}/{loc}{self.data_num}.p'
        if os.path.isfile(fname):
            with open(fname, 'rb') as f:
                if sys.version_info[0] < 3:
                    d = pickle.load(f)
                else:
                    d = pickle.load(f, encoding='latin1')  # needed for python 3    
            self.data[sensor] = d
            if sensor == 'imu':
                self.data['imu']['vals'] = self.data['imu']['vals'].astype(int)
        else:
            print(f"{fname} file doesn't exist")
    
    def load(self):
        # Loads cam, imu, and vcon data
        for sensor,loc in self.sensor:
            self.__read_data(sensor,loc)
    
    def print_data_stats(self):
        # Print Data Stats
        if self.data:
            for k in self.data.keys():
                print(k)
                for j in self.data[k].keys():
                    print(type(self.data[k][j]), j, self.data[k][j].shape)
                    print(type(self.data[k][j][0,0]))
        else:
            print('Load Data First')
    
    def __plotimu(self,calibration = False, plot_calib = False, plot_calib_only = False, loc = None, user = False):
        if 'imu' in self.data.keys():
            labels = ['A_x', 'A_Y','A_Z','W_Z','W_X','W_Y']
            plt.figure(figsize=(10, 6))  # Adjust figure size as needed
            # Plot each row of the data
            keys = list(self.data['imu'].keys())
            if not plot_calib_only:
                
                for i in range(self.data['imu'][keys[0]].shape[0]):
                    plt.plot(self.data['imu'][keys[1]][0], self.data['imu'][keys[0]][i], label=labels[i])

            
            if calibration:
                if user:
                    plt.title('IMU DATA vs TIME \n Select Two Points for Calibration')
                    print("Click on two points on the plot to get their indices.")
                    selected_points = plt.ginput(2, timeout=0)
                    # Convert selected points to indices
                    self.indices = [np.argmin(np.abs(self.data['imu'][keys[1]][0] - point[0])) for point in selected_points]
                else:
                    plt.title('IMU DATA vs TIME')
                    self.indices = [0,250]    
                print("Indices of selected points:", self.indices)
            else:
                plt.title('Calibrated IMU DATA vs TIME')
            if plot_calib and 'imu_calib' in self.data.keys():
                keys_calib = list(self.data['imu_calib'].keys())
                for i in range(self.data['imu_calib'][keys_calib[0]].shape[0]):
                    plt.plot(self.data['imu'][keys[1]][0], self.data['imu_calib'][keys_calib[0]][i], label=f'{labels[i]}_cal')
            ## Plot the vector on the x-axis
            plt.xlabel('Time')
            plt.ylabel('IMU DATA')
            plt.legend(loc='upper right', fontsize="7")
            plt.grid(True)
            if loc is not None:
                plt.savefig(f'{loc}calibrated_imu.png', dpi= 199)
            plt.show()
            
        else:
            print('IMU DATA NOT FOUND')
    
    def plot(self,sensor):
        "Plots sensor data"
        if sensor == 'imu':
            self.__plotimu()
    
    def calibrate_imu(self, root = None, user = False):
        # Calibrates IMU Value
        if 'imu' in self.data.keys():
            self.__plotimu(True, user = user)
            bias = np.mean(self.data['imu']['vals'][:,self.indices[0]:self.indices[1]], axis=1)
            V_ref = np.array([3300, 3300, 3300, 3300, 3300, 3300])
            sensitivity = np.array([300,300,300,3.33*180/np.pi,3.33*180/np.pi,3.33*180/np.pi])
            scale_factor = V_ref/sensitivity
            scale_factor = scale_factor/1023
            bias = bias - np.array([0,0,1/scale_factor[2],0,0,0])
            self.data['imu_calib'] = {}
            self.data['imu_calib']['vals'] = (self.data['imu']['vals'] - bias[:, np.newaxis])*scale_factor[:, np.newaxis]
            self.__plotimu(plot_calib=True, plot_calib_only=True, loc = root)

            self.data['imu_calib']['vals'][0,:] = self.data['imu_calib']['vals'][0,:]*9.81
            self.data['imu_calib']['vals'][1,:] = self.data['imu_calib']['vals'][1,:]*9.81
            self.data['imu_calib']['vals'][2,:] = self.data['imu_calib']['vals'][2,:]*9.81*(-1) 
            #print(self.data['imu']['vals'][:,1345])
            #print(bias)
            #print(scale_factor)
            #print(self.data['imu_calib']['vals'][:,1345])     
        else:
            print('No IMU Data Found')
    

class model():
    def __init__(self):
        # This class if representing the model
        self.data = None
        self.quat = None
        self.quat_gt = None
        self.optimised = None
        self.cost_array = None
    
    def load_data(self,data):
        # Loads Data
        self.data = data
        if 'imu_calib' in self.data:
            self.quat = np.zeros((4,self.data['imu_calib']['vals'].shape[1]))
        else:
            print('Please Calibrate IMU DATA First')
    
    def set_init_position(self,init):
        # Sets initial calculation of the Q vector and calculates the ground truth
        if self.quat is not None:
            self.quat[:,0] = np.array(init)
            print('Quarternion Shape', self.quat.shape)
            print('Initial Position', self.quat[:,0])
            self.__firstintegration()
            self.__getgroundtruth()

        else:
            print('Please Load data first')

    def __firstintegration(self):
        # This is for getting 1st set or the intial values
        self.__exponential()
        for i in range(self.delta_expo.shape[1]):
            self.quat[:,i+1] = self.__qmult(self.quat[:,i].reshape(4,1),self.delta_expo[:,i].reshape(4,1))
    
    def __qmult(self,q1,q2):
        # Multiplies two Quarternions
        q3 = np.zeros((4,1))
        q3[0,0] = q1[0,0]*q2[0,0] - np.dot(q1[1:].T,q2[1:])
        q3[1:] = q1[0,0]*q2[1:] + q2[0,0]*q1[1:] + np.cross(q1[1:].ravel(),q2[1:].ravel()).reshape(3,1)
        return q3[:,0]
    
    def __getgroundtruth(self):
        try:
            gt_data = self.data['vicon']['rots']
            self.quat_gt = np.zeros((4,gt_data.shape[2]))
            for i in range(gt_data.shape[2]):
                self.quat_gt[:,i] =  quaternions.mat2quat(gt_data[:,:,i])
                #self.quat_gt[:,i] = gotrans.quaternion_from_matrix(gt_data[:,:,i],isprecise=True)
        except:
            self.quat_gt = 0 

    def __exponential(self):
        ## This function makes the exponential angular velocity matrix of size 4XN
        self.data['imu_calib']['velocity'] = self.data['imu_calib']['vals'][3:,:].copy()
        assert self.data['imu_calib']['velocity'].shape[0] == 3
        
        self.data['imu_calib']['velocity'] = np.roll(self.data['imu_calib']['velocity'], -1, axis=0)
        self.time = np.diff(self.data['imu']['ts'])

        angles = self.data['imu_calib']['velocity'][:,:-1]*self.time
        angles = angles/2
        print(self.data['imu_calib']['vals'][:,1450])
        print(self.data['imu_calib']['velocity'][:,1450])
                
        angles_norm = np.linalg.norm(angles, axis=0)
        self.delta_expo = np.vstack((np.cos(angles_norm), np.sin(angles_norm) * angles/angles_norm))
    
    def __getangles(self):
        # Converts Quarternion to Ground Truth Angles
        if not isinstance(self.quat_gt, np.ndarray):
            try:
                gt_data = self.data['vicon']['rots']
                self.gt_angles = np.zeros((3,gt_data.shape[2]))
                for i in range(gt_data.shape[2]):
                    self.gt_angles[:,i] =  euler.mat2euler(gt_data[:,:,i])
                    #self.quat_gt[:,i] = gotrans.quaternion_from_matrix(gt_data[:,:,i],isprecise=True)
            except:
                self.gt_angles = int(0)    
            
        else:
            self.gt_angles = np.zeros((3,self.quat_gt.shape[1]))
            for i in range(self.quat_gt.shape[1]):
                self.gt_angles[:,i] = euler.quat2euler(self.quat_gt[:,i])
    
        self.euler_angles = np.zeros((3,self.quat.shape[1]))
        if self.optimised is not None:
            self.optimised_angles = np.zeros((3,self.optimised.shape[1]))
            for j in range(self.quat.shape[1]):
                self.optimised_angles[:,j] = euler.quat2euler(self.optimised[:,j])
        
        for j in range(self.quat.shape[1]):
            self.euler_angles[:,j] = euler.quat2euler(self.quat[:,j])
    
    def __getacceleration(self):
        # Converts acceleration to body frame Truth Angles
        self.gt_acc = self.data['imu_calib']['vals'][0:3,:]
        
        base_acc = jnp.array([[0,0,0,-9.81]]*self.quat.shape[1])
        
        quart_product = jax.vmap(self.__getprediction, in_axes = 0, out_axes =0)
        quart_inverse = jax.vmap(self.__getinverse, in_axes = 0, out_axes =0)
        
        q_inv = quart_inverse(jnp.array(self.quat.T.copy()))
        ac1 = quart_product(q_inv,base_acc)
        ac2 = quart_product(ac1,jnp.array(self.quat.T.copy()))
        self.projected_acc = np.array(ac2).T
        self.projected_acc = self.projected_acc[1:,:]

        if self.optimised is not None:
            q_inv = quart_inverse(jnp.array(self.optimised.T.copy()))
            ac1 = quart_product(q_inv,base_acc)
            ac2 = quart_product(ac1,jnp.array(self.optimised.T.copy()))
            self.optimised_acc = np.array(ac2).T
            self.optimised_acc = self.optimised_acc[1:,:]
        
        

    def plot_quaternion(self, loc = None):
        # Plots gt and predicted Quaternions
        if self.quat is not None and self.quat_gt is not None:
            labels = ['$q_w$','$q_x$','$q_y$','$q_z$']
            plt.figure(figsize=(10, 6))
            for i in range(self.quat.shape[0]):
                plt.subplot(2,2,i+1)
                plt.plot(self.data['imu']['ts'][0], self.quat[i,:], label=f'Estimated_{labels[i]}')
                if isinstance(self.quat_gt, np.ndarray):
                    plt.plot(self.data['vicon']['ts'][0], self.quat_gt[i,:], label=f'GT_{labels[i]}')
                if self.optimised is not None:
                    plt.plot(self.data['imu']['ts'][0], self.optimised[i,:], label=f'Optimised_{labels[i]}')
                plt.xlabel('Time')
                plt.ylabel('Quaternion Data')
                plt.legend(loc='upper right', fontsize="7")
                plt.grid(True)
            plt.suptitle('Ground Truth Vs Model Prediction of Quarternions')
            if loc is not None:
                plt.savefig(f'{loc}Quaternions.png',dpi= 199)
            plt.show()
            
        else:
            print('Please Set Initial Position or Run Optimization')

    
    def plot_acceleration(self, loc = None):
        if self.quat is not None:
            self.__getacceleration()
            labels = ['$A_x$','$A_y$','$A_z$']
            plt.figure(figsize=(10, 6))
            for i in range(self.projected_acc.shape[0]):
                plt.subplot(3,1,i+1)
                plt.plot(self.data['imu']['ts'][0], self.projected_acc[i,:], label=f'Estimated_{labels[i]}')
                plt.plot(self.data['imu']['ts'][0], self.gt_acc[i,:], label=f'GT_{labels[i]}')
                if self.optimised is not None:
                    plt.plot(self.data['imu']['ts'][0], self.optimised_acc[i,:], label=f'Optimised_{labels[i]}')
                plt.xlabel('Time')
                plt.ylabel('Acceleration Data')
                plt.legend(loc='upper right', fontsize="7")
                plt.grid(True)
            plt.suptitle('Ground Truth Vs Model Prediction of Acceleration')
            if loc is not None:
                plt.savefig(f'{loc}accelerations.png',dpi= 199)
            plt.show()
            
        else:
            print('Please Set Initial Position or Run Optimization')
    
    def plot_angles(self, loc = None):
        if self.quat is not None and self.quat_gt is not None:
            self.__getangles()
            labels = ['Roll','Pitch','Yaw']
            plt.figure(figsize=(10, 6))
            for i in range(self.euler_angles.shape[0]):
                plt.subplot(3,1,i+1)
                plt.plot(self.data['imu']['ts'][0], self.euler_angles[i,:]*180/np.pi, label=f'Estimated_{labels[i]}')
                if isinstance(self.gt_angles, np.ndarray):
                    plt.plot(self.data['vicon']['ts'][0], self.gt_angles[i,:]*180/np.pi, label=f'GT_{labels[i]}')
                if self.optimised is not None:
                    plt.plot(self.data['imu']['ts'][0], self.optimised_angles[i,:]*180/np.pi, label=f'Optimised_{labels[i]}')
                plt.xlabel('Time')
                plt.ylabel('Euler Angle Data')
                plt.legend(loc='upper right', fontsize="7")
                plt.grid(True)
            plt.suptitle('Ground Truth Vs Model Prediction of Euler Angles')
            if loc is not None:
                plt.savefig(f'{loc}angles.png',dpi= 199)
            plt.show()
            
        else:
            print('Please Set Initial Position or Run Optimization')
    
    def __plot_cost(self,loc):
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_array)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.title('Cost Vs Iterations')
        if loc is not None:
            plt.savefig(f'{loc}cost.png',dpi= 199)
        plt.show()
        
    def __getprediction(self,b,a):
        # getting f term in cost function. Takes 4x1 products
        c_s = jnp.array([a[0]*b[0] - jnp.dot(a[1:].T,b[1:])])
        c_v = a[0]*b[1:] + b[0]*a[1:] - jnp.cross(a[1:],b[1:])
        return jnp.concatenate([c_s,c_v])
    
    def __getinverse(self,q):
        #  process for calculating inverse of 4x1 array
        #q_norm = jnp.max(jnp.array([jnp.linalg.norm(q),1e-3]))
        q_norm = jnp.linalg.norm(q) + 0.001
        q = q/q_norm
        q = q*jnp.array([1,-1,-1,-1])
        return q

    def __getlog(self,q):
        # Calculates log of 4x1 quaternion
        #q_norm = jnp.max(jnp.array([jnp.linalg.norm(q),1e-3]))
        q_norm = jnp.linalg.norm(q) + 0.001
        qv_norm = jnp.linalg.norm(q[1:]) + 0.001
        l_s = jnp.array([jnp.log(q_norm)])
        l_v = q[1:]*jnp.arccos((q[0]/q_norm))/qv_norm
        return jnp.concatenate([l_s, l_v])

    def __getunit(self,q):
        #q_norm = jnp.max(jnp.array([jnp.linalg.norm(q),1e-3]))
        q_norm = jnp.linalg.norm(q) + 0.001
        return q/q_norm
        
    def __cost(self,q_array_init):
        # Calculates the cost function
        ## Foward part
        ### Calculte Predicted part
        ####f0 has the shape Nx4
        q_array = jnp.concatenate([jnp.array([[1,0,0,0]]), q_array_init[:-1,:]],axis = 0)
        #print(q_array.shape)
        f0 = self.get_prediction_parallel(self.expo_array,q_array)
        #print(f0.shape)
        #### Calculate Variable Part
        q_inv = self.get_inverse_parallel(q_array_init)
        #print(q_inv.shape)
        #### Multiply Both
        f1 = self.get_prediction_parallel(q_inv,f0)
        #print(f1.shape)
        #### Get log Term
        f2 = self.get_log_parallel(f1)
        #print(f2.shape)
        #### Cost predictive
        cost = 2*(jnp.linalg.norm(f2,ord = 'fro')**2)

        ### Accleration Part
        a_world = jnp.array([[0,0,0,-9.81]]*q_array_init.shape[0])
        #print(a_world.shape)
        #### First Multiplication
        a1 = self.get_prediction_parallel(q_inv,a_world)
        #print(a1.shape)
        #### Second Multiplication 
        a2 = self.get_prediction_parallel(a1,q_array_init)
        #print(a2.shape)
        #### Find Difference 
        a3 = jnp.array(self.data['imu_calib']['vals'][0:3,1:].T) - a2[:,1:]
        #print(a3.shape)
        # Final Cost 
        cost = cost + 0.5*(jnp.linalg.norm(a3,ord = 'fro')**2)
        return cost

    def optimize(self, max_iter = 200, alpha = 0.001, epsilon = 0.001, num_count = 10, loc = None):
        # Main function for optimization
        self.cost_array = []
        self.get_prediction_parallel = jax.vmap(self.__getprediction, in_axes = 0, out_axes =0)
        self.get_inverse_parallel = jax.vmap(self.__getinverse, in_axes = 0, out_axes =0)
        self.get_log_parallel = jax.vmap(self.__getlog, in_axes = 0, out_axes =0)
        self.get_unit_parallel = jax.vmap(self.__getunit, in_axes = 0, out_axes =0)
        
        # Here expo_array and q_array_init are Nx4 and N+1x4 vectors
        self.expo_array = jnp.array(self.delta_expo.T)
        q_array_init = jnp.array(self.quat[:,1:].T)
        #print(q_array_init.shape)
        #print(self.__cost(q_array_init))
        
        # Gradient Finction
        grad_f = jax.grad(self.__cost)
        
        # Start Optimization
        
        j = 0
        for i in tqdm(range(max_iter)):
            self.cost_array.append(self.__cost(q_array_init))
            grad_at_x = grad_f(q_array_init)
            q_array_init =q_array_init -  alpha*grad_at_x
            q_array_init =  self.get_unit_parallel(q_array_init)
            if i > 0:
                if abs(1 - (self.cost_array[i]/self.cost_array[i-1])) < epsilon:
                    j = j + 1
                else:
                    j = 0
                if j > num_count:
                    print(f'Converged at {i} iteration')
                    break
        self.optimised = self.quat.copy()
        self.optimised[:,1:] = np.array(q_array_init).T

        np.save('optimised.npy', self.optimised)
        np.save('Data.npy',self.data)
        self.__plot_cost(loc)


class panaroma():
    # This Class is responsible for making panaroma from images
    def __init__(self, data, quart):
        self.data = data
        self.quart = quart
        self.base_image = None
        self.camera_properties = {'v_view':45, 'h_view': 60}
        self.pan_properties = {'h_view':2*np.pi, 'v_view': np.pi, 'height': 720, 'width': 1080}
    def __make_baseimage(self):
        # Make a nxmx2 matrix with index position as values. Row on 0 and column on 1
        N = self.data['cam']['cam'].shape[0]
        M = self.data['cam']['cam'].shape[1]

        x_indices, y_indices = np.indices((N, M),dtype = float)
        self.base_image = np.stack((x_indices, y_indices), axis=-1)

    def __imgtosp(self):
        if self.base_image is not None:
            self.base_image[:,:,0] = self.base_image[:,:,0]*self.camera_properties['v_view']/(self.base_image.shape[0] -1)
            self.base_image[:,:,0] = self.base_image[:,:,0] - (self.camera_properties['v_view']/2)

            self.base_image[:,:,1] = self.base_image[:,:,1]*self.camera_properties['h_view']/(self.base_image.shape[1]-1)
            self.base_image[:,:,1] = self.base_image[:,:,1] - (self.camera_properties['h_view']/2)

            self.base_image = self.base_image*np.pi/180
    def img_to_sphere(self):
        ## This function converts image coordinates to spherical coordinates
        ## 0 position is phi, and 1 position is lamda
        self.__make_baseimage()
        #print('Base Image', self.base_image[147,54,:])

        self.__imgtosp()
        #print('Camera_Spherical Cooridinate', self.base_image[147,54,:])
    
    def sphere_to_cartesian(self):
        ## This function changes spherical coordinates to cartesian coordinates in camera frame
        ## R = 1
        ## x = cos(lamda).cos(phi), y = sin(lamda).cos(phi), z = sin(phi)
        self.camera_coord = np.zeros((self.base_image.shape[0],self.base_image.shape[1],3))
        c_lamda = np.cos(self.base_image[:,:,1])
        s_lamda = np.sin(self.base_image[:,:,1])
        c_phi = np.cos(self.base_image[:,:,0])
        s_phi = np.sin(self.base_image[:,:,0])
        
        # This is what we did first
        self.camera_coord[:,:,0] = c_lamda*c_phi
        self.camera_coord[:,:,1] = -1*s_lamda*c_phi
        self.camera_coord[:,:,2] = -1*s_phi
        
        # This is new
        #self.camera_coord[:,:,0] = s_lamda*c_phi
        #self.camera_coord[:,:,1] = s_phi
        #self.camera_coord[:,:,2] = c_lamda*c_phi
        #print('Camera Cartesian Cooridnates', self.camera_coord[147,54,:])
    
    def __closest_time(self):
        # Calculate the absolute differences between every element in A and every element in B
        abs_diff = np.abs(self.data['imu']['ts'][0,:][:, np.newaxis] - self.data['cam']['ts'][0,:][np.newaxis, :])
        #abs_diff = np.abs(self.data['vicon']['ts'][0,:][:, np.newaxis] - self.data['cam']['ts'][0,:][np.newaxis, :])
        
        # Find the index of the minimum difference for each element in B
        index = np.argmin(abs_diff, axis=0)
        # Get the required Quarternions
        self.close_quart = self.quart[:,index]
        #self.close_quart = self.data['vicon']['rots'][:,:,index]
        
        #print('closest Quarternion', self.close_quart[:,923])
    
    def __camera_to_world(self):
        # Returns nXMxNx3 vector of world coordinates for each pixel for each image
        reshaped_matrix = np.transpose(self.rot_matrix_cam,(2,0,1) )
        reshaped_matrix_2 = np.transpose(self.camera_coord,(2,0,1)).reshape(3,-1)
        ans = np.tensordot(reshaped_matrix, reshaped_matrix_2, axes = 1)
        num, M, N = self.rot_matrix_cam.shape[2], self.camera_coord.shape[0], self.camera_coord.shape[1]
        self.world_coords = np.transpose(ans,(0,2,1)).reshape(num,M,N,3)
        #print('world_cods', self.world_coords[923,147,54,:])
        
    def img_to_world(self):
        # This function converts img cartesian coordinates to world cartesian coordinates
        ## 1st Get all the timesteps closer to images
        self.__closest_time()

        ## loop over the quarternions to get the rotation matrix:
        self.rot_matrix_imu = np.zeros((3,3,self.close_quart.shape[1]))
        #self.rot_matrix_imu = np.zeros((3,3,self.close_quart.shape[2]))
        
        #cam_to_imu = np.array([[0,0,-1],[0,1,0],[-1,0,0]])
        #cam_to_imu = np.array([[1,0,0],[0,0,0],[0,1,0]])
        
        for i in range(self.rot_matrix_imu.shape[2]):
            #self.rot_matrix_imu[:,:,i] = np.dot(quaternions.quat2mat(self.close_quart[:,i]),cam_to_imu)
            self.rot_matrix_imu[:,:,i] = quaternions.quat2mat(self.close_quart[:,i])
            #self.rot_matrix_imu[:,:,i] = self.close_quart[:,:,i]

        ## Get camera_frame to world matrix
        self.rot_matrix_cam = self.rot_matrix_imu    
        
        #print('Rotation Matrix', self.rot_matrix_cam[:,:,923])
        ## Get world_coordinates
        self.__camera_to_world()

        #print('world cartesian Coordinates', self.world_coords[923,147,54,:])
    
    def __getR(self):
        ans3 = np.linalg.norm(self.world_coords,axis = 3,keepdims=True) + 1e-4
        self.world_coords = self.world_coords/ans3
    
    def cartesion_to_sphere(self):
        # This function Converts world cartesian to spherical coordinates
        ## We have an ixmxnx3 vector
        ## We need to have ixmxnx2
        ### x = Rcos(phi)cos(lamda)
        ### y = Rsin(lamda)cos(phi)
        ### z = Rsin(phi)

        # Normalize to remove R
        self.__getR()
        # Get an array
        self.sphere_world = np.zeros((self.world_coords.shape[0],
                                      self.world_coords.shape[1],
                                      self.world_coords.shape[2],2))

        #self.sphere_world[:,:,:,0] = np.arcsin(self.world_coords[:,:,:,2])
        #self.sphere_world[:,:,:,1] = np.arctan2(self.world_coords[:,:,:,1],self.world_coords[:,:,:,0])

        self.sphere_world[:,:,:,0] = np.arcsin(-1*self.world_coords[:,:,:,2])
        self.sphere_world[:,:,:,1] = np.arctan2(-1*self.world_coords[:,:,:,1],self.world_coords[:,:,:,0])

        # This is New
        #self.sphere_world[:,:,:,0] = np.arcsin(self.world_coords[:,:,:,1])
        #self.sphere_world[:,:,:,1] = np.arctan2(self.world_coords[:,:,:,0],self.world_coords[:,:,:,2])
        #print('world spherical', self.sphere_world[923,147,54,:])
    
    def get_pixel_coord(self):
        # This function converts spherical to image pixel coordinates 
        ## row: i = (world_coord + (range/2))*height/range
        ## column: j = (world_coord + (range/2))*width/range

        self.pixel_coord = np.zeros(self.sphere_world.shape)
        self.pixel_coord[:,:,:,0] = self.sphere_world[:,:,:,0] + (self.pan_properties['v_view']/2)
        self.pixel_coord[:,:,:,0] = self.pixel_coord[:,:,:,0]*(self.pan_properties['height']-1)/self.pan_properties['v_view']
        self.pixel_coord[:,:,:,0] = np.clip(self.pixel_coord[:,:,:,0],0,self.pan_properties['height']-1)

        self.pixel_coord[:,:,:,1] = self.sphere_world[:,:,:,1] + (self.pan_properties['h_view']/2)
        self.pixel_coord[:,:,:,1] = self.pixel_coord[:,:,:,1]*(self.pan_properties['width']-1)/self.pan_properties['h_view']
        self.pixel_coord[:,:,:,1] = np.clip(self.pixel_coord[:,:,:,1],0,self.pan_properties['width']-1)
        self.pixel_coord = self.pixel_coord.astype(int)


        #print('Pan_pixel', self.pixel_coord[923,147,54,:])

    def stitch_pan(self, loc = None):
        self.pan_img = np.zeros((self.pan_properties['height'], self.pan_properties['width'], 3),dtype = np.uint8)
        i,m,n = self.pixel_coord.shape[0], self.pixel_coord.shape[1], self.pixel_coord.shape[2]
        i_indices,m_indices, n_indices = np.indices((i,m, n))
        r_indices = self.pixel_coord[:,:,:,0]
        s_indices = self.pixel_coord[:,:,:,1]
        self.img_array = np.transpose(self.data['cam']['cam'], (3,0,1,2))
        #print(type(self.img_array[0,0,0,0]))
        #print(self.img_array.shape)
        self.pan_img[r_indices, s_indices] = self.img_array[i_indices, m_indices, n_indices]  
        #print(self.pan_img.shape)
        #import cv2
        #cv2.imshow('hell0',self.pan_img)
        #cv2.waitKey(0)
        plt.figure(figsize=(10, 6))
        plt.imshow(self.pan_img)
        if loc is not None:
            plt.savefig(f'{loc}panimg.png',dpi= 199)

        plt.show()  
        
        

        
    


        
    
        





        


