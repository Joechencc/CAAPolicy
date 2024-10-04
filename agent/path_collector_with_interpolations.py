import carla
import math
import torch
import time
  

import numpy as np
import matplotlib.pyplot as plt

 


import cv2 
import time, sys
print('\n\nRemember to update your path of carla folder....')
sys.path.append('/home/taf1syv/Desktop/e2e-parking-carla')
sys.path.append('/home/taf1syv/Desktop/e2e-parking-carla/carla/PythonAPI')
sys.path.append('/home/taf1syv/Desktop/e2e-parking-carla/carla/PythonAPI/carla')



import math
import agent.hybrid_A_star as hybrid_a_star
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import agent.draw as draw
from functools import reduce
import open3d as o3d #### for lidar point clouds
from matplotlib import cm
from scipy.interpolate import CubicSpline 




VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])


W = 2.16  # width of car
LF = 3.8  # distance from rear to vehicle front end
LB = 1.0  # distance from rear to vehicle back end
# vehicle rectangle vertices
VRX = [LF, LF, -LB, -LB, LF]
VRY = [W / 2, -W / 2, -W / 2, W / 2, W / 2]
MOVE_STEP = 0.2
WD = 0.7 * W
WB = 2.9
TR = 0.4
TW = 0.8



def interpolate_path_cubic(np_path, num_waypoints):
    # Extract the x, y, yaw, and direction
    x_vals = np_path[:, 0]
    y_vals = np_path[:, 1]
    yaw_vals = np.unwrap(np_path[:, 2])

 
    # Create an array of indices for the original points
    t = np.arange(len(x_vals))
 
    # Generate a finer set of points for interpolation
    t_fine = np.linspace(t[0], t[-1], len(x_vals) * num_waypoints)
 
    # Perform cubic spline interpolation for x and y
    cs_x = CubicSpline(t, x_vals)
    cs_y = CubicSpline(t, y_vals)
    cs_yaw = CubicSpline(t, yaw_vals)  # Nonlinear interpolation for yaw
 
    x_interp = cs_x(t_fine)
    y_interp = cs_y(t_fine)
    yaw_interp = cs_yaw(t_fine)%(2*np.pi)
   
    # Combine all interpolated results
    interpolated_path = np.column_stack((x_interp, y_interp, yaw_interp))
   
    return interpolated_path



def pi_2_pi(theta):
    while theta > math.pi:
        theta -= 2.0 * math.pi

    while theta < -math.pi:
        theta += 2.0 * math.pi

    return theta    


class Path_collector:
    def __init__(self, data_generator, round):
        self.rgb_rear = None
        self.rgb_right = None
        self.rgb_left = None
        self.rgb_front = None
        self.pre_target_point = None

        self.net_eva = data_generator
        self.world = data_generator.world
        self.player = data_generator.world._player

        ################# bounding box info #####################
        self.bbox = self.player.bounding_box.get_local_vertices() 
        ##########################################################
        self.bbox_min = [self.bbox[0].x, self.bbox[0].y, -1.6+self.bbox[0].z] ### with respect to lidar z
        self.bbox_max = [self.bbox[7].x, self.bbox[7].y, -1.6+self.bbox[7].z]
        
        self.solution_required = True
        if self.net_eva._parking_goal_index <= 31:
            self.target_yaw = 180
        else:
            self.target_yaw = 0    
        self.current_target = [self.net_eva._parking_goal.x, self.net_eva._parking_goal.y, self.target_yaw]

        ##### get target with respect to the Carla world coordinate (vehicle center)
        self.target_x, self.target_y = self.net_eva._parking_goal.x, self.net_eva._parking_goal.y 
        ### shift the center to rear center in world coordinate system
        self.target_x += -1.37*math.cos(np.deg2rad(-self.target_yaw))
        self.target_y += 1.37*math.sin(np.deg2rad(-self.target_yaw))

    
        self.path_stage = None ### Breaking the path into multiple segments and using this index to access the segment
        self.current_stage = None
        self.r_trajectory = None
        self.my_control = carla.VehicleControl()
        self.prev_dist = None

        self.round = round
        self.exp_name = 'seed_'+str(self.round)+'_'+'target_index_'+str(self.net_eva._parking_goal_index)
        
            
        self.is_init = False
        self.intrinsic_crop = None
        self.extrinsic = None
         
        self.process_frequency = 3  # process sensor data for every 3 steps 0.1s
        self.step = -1

        self.prev_xy_thea = None

        self.trans_control = carla.VehicleControl()
      
        self.stop_count = 0
        self.boost = False
        self.boot_step = 0

        #self.init_agent()

        plt.ion()


    #
    # def init_agent(self):
    #     w = self.world.cam_config['width']
    #     h = self.world.cam_config['height']
    #
    #     veh2cam_dict = self.world.veh2cam_dict
    #     front_to_ego = torch.from_numpy(veh2cam_dict['camera_front']).float().unsqueeze(0)
    #     left_to_ego = torch.from_numpy(veh2cam_dict['rgb_left']).float().unsqueeze(0)
    #     right_to_ego = torch.from_numpy(veh2cam_dict['rgb_right']).float().unsqueeze(0)
    #     rear_to_ego = torch.from_numpy(veh2cam_dict['rgb_rear']).float().unsqueeze(0)
    #     self.extrinsic = torch.cat([front_to_ego, left_to_ego, right_to_ego, rear_to_ego], dim=0)
    #
    #
    #
    #     self.step = -1
    #     self.pre_target_point = None
    #
    #     self.ego_speed = []

    

    def tick(self, carla_world):
        if self.current_target[:2] != [self.net_eva._parking_goal.x, self.net_eva._parking_goal.y]: ### we also need to consider the initial position change for ego
            print('Parking task changes, hence we need to find a new path')
            if self.net_eva._parking_goal_index <= 31:
                self.target_yaw = 180
            else:
                self.target_yaw = 0    
            self.current_target = [self.net_eva._parking_goal.x, self.net_eva._parking_goal.y, self.target_yaw]
            self.solution_required = True
            ##### get target with respect to the Carla world coordinate (vehicle center)
            self.target_x, self.target_y = self.net_eva._parking_goal.x, self.net_eva._parking_goal.y 
            ### shift the center to rear center in world coordinate system
            self.target_x += -1.37*math.cos(np.deg2rad(-self.target_yaw))
            self.target_y += 1.37*math.sin(np.deg2rad(-self.target_yaw))
            self.r_trajectory = None
            self.current_stage = None
            self.exp_name = 'seed_'+str(self.round)+'_'+'target_index_'+str(self.net_eva._parking_goal_index)


        self.step += 1

        # stop 1s for new eva
        if self.step < 30: ### 30 frames per second
            self.player.apply_control(carla.VehicleControl())
            #self.net_eva._ego_transform_generator.get_data_gen_ego_transform()
            #self.player.set_transform(self.net_eva.ego_transform)
            return

        if self.step % self.process_frequency == 0:
            data_frame = self.world.sensor_data_frame

            if not data_frame:
                return

            data = self.get_model_data(data_frame)
            try:
                voxelGrid = o3d.geometry.VoxelGrid.create_from_point_cloud(data['point_list'], 0.1) 
            except:
                voxelGrid = None    
            
            if self.solution_required:                                               ## semantic  ## lidar pc
                result, ox_real, oy_real, gx, gy, gyaw = self.hybrid_A_star_solution(data['feng'], voxelGrid)
                #sx, sy, syaw = 16.0, 14.63, np.deg2rad(90.0) ## ego rear center pos in BEV coordinate system (right hand)

                if result is not None:
                    self.solution_required = False
                    self.path_stage = 0 ### go through the path segments one by one, starts with the first segment
                    self.ego_traject = [] ### to log ego trajectory under the controller
                    self.hist_use = 10000 ### a large number
                    self.kk = -1
                   
                    self.start_time = time.time()
                    
                    #### in current ego frame coordinate system
                    x, y, yaw, direction = result

                    ###### convert the path to world coordinate
                    print('Convert the path to world coordinate')
                    self.positions = []
                    for k in range(0, len(x), 1):
                        y_graph = x[k]-16.0  #Rear center in ego coordinate x heading (since this path uses rear center)
                        x_graph = y[k]-16.0  
                        yaw_graph = 90-np.rad2deg(yaw[k])  #np.deg2rad(90)-(yaw[k])  ## right hand to left hand
                        ###first convert this point from ego coordinate to world coordinate, both are left hand
                        y_world = math.cos(self.ab_yaw_init)*y_graph - math.sin(self.ab_yaw_init)*x_graph + self.a_y_init
                        x_world = math.sin(self.ab_yaw_init)*y_graph + math.cos(self.ab_yaw_init)*x_graph + self.b_x_init
                        yaw_world = np.deg2rad(yaw_graph)-self.ab_yaw_init ### notice here self.ab_yaw_init is negative
                        #### Now we only care about rear center
                        # ###Get the center in world coordinate
                        x_world += 1.37*math.cos(yaw_world)
                        y_world += 1.37*math.sin(yaw_world)
                        #location = carla.Location(x=x_world, y= y_world, z = 0.2)
                        #carla_world.debug.draw_point(location, size=0.02,  color=carla.Color(255,0,0),life_time=30)
                        self.positions.append([x_world, y_world, yaw_world, direction[k]])
                    self.positions.append(
                        [self.net_eva._parking_goal.x, self.net_eva._parking_goal.y, np.deg2rad(self.target_yaw),
                         direction[k]])
                    np_vec = np.array(self.positions)
                    #np.savetxt('rear_path_{}.txt'.format(self.exp_name), np_vec)  ##np.loadtxt('rear_path.txt') ==> a np array

                    oy_real_w, ox_real_w = [], []
                    for ele1, ele2 in zip(ox_real, oy_real):
                        oy_real_w.append(math.cos(self.ab_yaw_init)*(ele1-16) - math.sin(self.ab_yaw_init)*(ele2-16)+ self.a_y_init)
                        ox_real_w.append(math.sin(self.ab_yaw_init)*(ele1-16) + math.cos(self.ab_yaw_init)*(ele2-16)+ self.b_x_init)

                    
                    plt.cla()
                    #plt.scatter(self.player.get_transform().location.y, self.player.get_transform().location.x, label='ego center')
                    plt.plot(oy_real_w, ox_real_w, "sk", markersize=1)
                    plt.scatter(self.player.get_transform().location.y-1.37*math.sin(np.deg2rad(self.player.get_transform().rotation.yaw)), self.player.get_transform().location.x-1.37*math.cos(np.deg2rad(self.player.get_transform().rotation.yaw)), label='ego rear start')
                    plt.scatter(self.target_y, self.target_x, label='target rear')
                    plt.plot(np_vec[:, 1],np_vec[:,0], label='planned path rear') ### using UE coordinate system
                    plt.legend()
                    plt.title('Visulize the planned path in world coordinate for 5 seconds')
                    plt.show()
                    plt.pause(5)
                    plt.close()  


                    print('We interpolate the generated path....')
                    self.new_points = interpolate_path_cubic(np_vec[:,:-1], 3)
                    plt.cla()
                    plt.plot(oy_real_w, ox_real_w, "sk", markersize=1)
                    plt.scatter(self.player.get_transform().location.y-1.37*math.sin(np.deg2rad(self.player.get_transform().rotation.yaw)), self.player.get_transform().location.x-1.37*math.cos(np.deg2rad(self.player.get_transform().rotation.yaw)), label='ego rear start')
                    plt.scatter(self.target_y, self.target_x, label='target rear')
                    plt.plot(np_vec[:, 1],np_vec[:,0], label='planned path rear') ### using UE coordinate system
                    plt.plot(self.new_points[:, 1],self.new_points[:,0], label='interpolated planned path rear') 
                    plt.legend()
                    plt.title('Visulize the planned path in world coordinate for 5 seconds')
                    plt.show()
                    plt.pause(5)
                    plt.close()  
                    
                    self.positions = self.new_points.copy()
                    
                    # # #### This block manually assign the next position for ego
                    # for k in range(0, len(x), 1):
                    #     y_graph = x[k]-16.0  #Rear center in ego coordinate (since this path uses rear center)
                    #     x_graph = y[k]-16.0  
                    #     yaw_graph = 90-np.rad2deg(yaw[k])  #np.deg2rad(90)-(yaw[k])  ## right hand to left hand
                    #     ###first convert this point from ego coordinate to world coordinate
                    #     y_world = math.cos(self.ab_yaw_init)*y_graph - math.sin(self.ab_yaw_init)*x_graph + self.a_y_init
                    #     x_world = math.sin(self.ab_yaw_init)*y_graph + math.cos(self.ab_yaw_init)*x_graph + self.b_x_init
                    #     yaw_world = np.deg2rad(yaw_graph)-self.ab_yaw_init
                    
                    #     ###Get the center in world coordinate
                    #     x_world += 1.37*math.cos(yaw_world)
                    #     y_world += 1.37*math.sin(yaw_world)
                    #     ### Assign position
                    #     transform = self.player.get_transform()
                    #     transform.location.x = x_world  
                    #     transform.location.y = y_world  
                    #     transform.rotation.yaw = np.rad2deg(yaw_world)
                    #     self.player.set_transform(transform) 
                    #     carla_world.tick()

                else:
                    print('\nHybrid A* fails to find a path, soft movement to trigger a new search position...')
                    plt.cla()
                    plt.plot(ox_real, oy_real, "sk", markersize=1)
                    plt.title("Hybrid A* fails to find a path")
                    plt.axis("equal")
                    plt.show()        
                    #### set ego new position as a steady throttle
                    self.trans_control.throttle = 1  
                    self.trans_control.brake = 0  
                    self.trans_control.steer = 0 
                    self.player.apply_control(self.trans_control)
                return    
            else:
                self.kk += 1 ##self.kk = min(self.kk, len(self.positions)-1)
                if self.kk < len(self.positions):
                    x_world = self.positions[self.kk][0]
                    y_world = self.positions[self.kk][1]
                    yaw_world = self.positions[self.kk][2]
                
                    ###Get the center in world coordinate
                    # x_world += 1.37*math.cos(yaw_world)
                    # y_world += 1.37*math.sin(yaw_world)
                    ### Assign position
                    transform = self.player.get_transform()
                    transform.location.x = x_world  
                    transform.location.y = y_world  
                    transform.rotation.yaw = np.rad2deg(yaw_world)
                    self.player.set_transform(transform)
                else:
                    return     

    #################################################################################################
    def hybrid_A_star_solution(self, semantic_img, voxels=None):
        if voxels is None:
            ### crop
            center = semantic_img.shape### 640*640*3
            w, h = 320, 320
            x = center[1]/2 - w/2
            y = center[0]/2 - h/2
            crop_img = semantic_img[int(y):int(y+h), int(x):int(x+w), 0] ###only use Red channel
            ox = []
            oy = []
            ox_real = []
            oy_real = []
            for i in range(6,26): ### add search boundary
                ox.append(6)
                oy.append(i)
                ox.append(26)
                oy.append(i)
            for i in range(6, 26):
                ox.append(i)
                oy.append(26) 
                ox.append(i)
                oy.append(6)    

            cell_size = 0.1
            N, M =  crop_img.shape
            for i in range(N): ## row 
                for j in range(M): ## column
                    if crop_img[i][j] == 81 or crop_img[i][j] == 153 or crop_img[i][j] == 128 or crop_img[i][j] == 50: ### Ground (81, 0, 81), Pole (153, 153, 153), Road (128, 64, 128), Lane (50, 234, 157)
                        continue
                    elif 134<=i<=185 and 148<=j<=171: ###tesla model 3 size range (we canuse https://pixspy.com/ to get) 
                        ## cv2.imwrite('test.png', semantic_img[int(y):int(y+h), int(x):int(x+w), :]) ## H:4.823m, W: 2.226m 
                        continue
                    else:
                        for x_a, y_a in [[-cell_size/2,-cell_size/2], [cell_size/2, cell_size/2], [-cell_size/2, cell_size/2], [cell_size/2, -cell_size/2]]: 
                            x_ = j*cell_size+x_a
                            y_ = (N-i)*cell_size-y_a
                           
                            if 6 < y_ < 26: 
                                ox.append(x_)
                                oy.append(y_)
                                ox_real.append(x_)
                                oy_real.append(y_)

                        #### we might only use the center for simiplification purpose and speed up the path searching        
                        # x_ = j*cell_size
                        # y_ = (N-i)*cell_size ### shift to right-hand x,y coordinate
                        # if 6 < y_ < 26: ### we only care about the local region centered on ego. 6 < x_< 26 and 
                        #     ox.append(x_)
                        #     oy.append(y_)
                        #     ox_real.append(x_)
                        #     oy_real.append(y_)
            
        else:
            print('Using voxel grid')
            ox = []
            oy = []
            indexes = [v.grid_index for v in voxels.get_voxels()]
            occupied_list = np.array([np.round(voxels.get_voxel_center_coordinate(index), 1) for index in indexes])
            
            ox = (occupied_list[:,0]+16.0).tolist()
            oy = (occupied_list[:,1]+16.0).tolist()

            ###### more accurate occ
            # for direction in [[-0.05, -0.05], [-0.05, 0.05], [0.05, -0.05], [0.05, 0.05]]:
            #     ox += (occupied_list[:,0]+16.0+direction[0]).tolist()
            #     oy += (occupied_list[:,1]+16.0+direction[1]).tolist()
 
            ox_real = ox.copy()  ## ox, oy can add some extra fake points to limit the solution space
            oy_real = oy.copy()

            ##### limit the searching space
            for i in range(6,26): ### add search boundary
                ox.append(6)
                oy.append(i)
                ox.append(26)
                oy.append(i)
            for i in range(6, 26):
                ox.append(i)
                oy.append(26) 
                ox.append(i)
                oy.append(6)  

        print('hybrid_A_star_solution input obstacle length: ', len(ox)) 
        ##### in graph coordinate right hand,        
        sx, sy, syaw = 16.0, 14.63, np.deg2rad(90.0) ### assume the RB = 1m ==> 2.37-1= 1.37 (16.0-1.37=14.63)
        ##### transform target rear center to current frame coordinate
        self.ab_yaw_init = np.deg2rad(-self.player.get_transform().rotation.yaw) ## clockwise is negative in ab coordinate
        self.b_x_init = self.player.get_transform().location.x
        self.a_y_init = self.player.get_transform().location.y
        
        #### +16 and swap x, y axis
        gyaw = np.deg2rad(90-self.target_yaw)-self.ab_yaw_init  ### in the right hand
        gx = 16 + math.cos(self.ab_yaw_init)*(self.target_y- self.a_y_init) + math.sin(self.ab_yaw_init)*(self.target_x- self.b_x_init)  
        gy = 16 - math.sin(self.ab_yaw_init)*(self.target_y- self.a_y_init) + math.cos(self.ab_yaw_init)*(self.target_x - self.b_x_init)   

        start_time = time.time()  
        print('Starting to find a path...')  
        result = hybrid_a_star.hybrid_astar_planning(sx, sy, syaw, gx, gy, gyaw, ox, oy)
        if result is not None:
            print('It takes {} seconds to find a path'.format((time.time()-start_time)), 'total waypoints: ', len(result[0]))
            return result, ox_real, oy_real, gx, gy, gyaw 
        else:
            print('It takes {} seconds but still cannot find a path'.format((time.time()-start_time)))  
            return None, ox_real, oy_real, gx, gy, gyaw  
            ### choose a safe action, say moving forward with a small step 
    #################################################################################################        

    def get_model_data(self, data_frame):
        data = {}
        #########################################################################################################################
        ####### semantic camera always rotate with ego so it is always using ego coordinate system.
        #print('semantic data: ', type(data_frame['BEV_semantic']))
        # data_frame['BEV_semantic'].convert(carla.ColorConverter.CityScapesPalette)
        # feng = np.copy(np.frombuffer(data_frame['BEV_semantic'].raw_data, dtype=np.dtype("uint8")))
        # feng = np.reshape(feng, (data_frame['BEV_semantic'].height, data_frame['BEV_semantic'].width, 4))
        # ### convert BGRA to RGB
        # feng = feng[:, :, :3]
        # feng = feng[:,:, ::-1] ## since B & R channels share the same value, we can save the reverse process
        # data['feng'] = feng  
        # # cv2.imshow('semantic', feng[160:480, 160:480, :]) 
        # # cv2.waitKey(0)
        # # filename = str(time.time()) + '.png'
        # # cv2.imwrite('log/'+filename, feng)
        data['feng'] = None ### I disabled the semantic sensor 
        #########################################################################################################################
        ##### lidar
               
        feng2 = np.copy(np.frombuffer(data_frame['feng_lidar_occ'].raw_data, dtype=np.dtype('f4'))) 
        feng2 = np.reshape(feng2, (int(feng2.shape[0]/4), 4))
        intensity = feng2[:, -1] 
        intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
        int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

        points = feng2[:, :-1] 

        #####filter out ego and z < 0
        mask = np.all((points >= self.bbox_min) & (points <= self.bbox_max), axis=1)
        filtered_points = points[~mask]

        local_lidar_points = filtered_points.T
        local_lidar_points2 = np.r_[local_lidar_points, [np.ones(local_lidar_points.shape[1])]]
        world_points = np.dot(data_frame['feng_lidar_occ'].transform.get_matrix(), local_lidar_points2)
        filtered_points = (np.dot(np.array(self.player.get_transform().get_inverse_matrix()), world_points)).T[:,:-1] 

        #print('Point cloud z range: ', min(filtered_points[:,2]), max(filtered_points[:,2]), '\n')
        filtered_points = filtered_points[filtered_points[:,2]>0.2] ###-height ##originall for lidar coordinate system
        filtered_points[:,2] = 0.05  ### flatten all points to the same z

        point_list = o3d.geometry.PointCloud()           ######## change left handed to righ handed
        filtered_points[:, [0,1]] = filtered_points[:, [1,0]]
        point_list.points = o3d.utility.Vector3dVector(filtered_points) ### right handed  ##
        point_list.colors = o3d.utility.Vector3dVector(int_color)

        data['point_list'] = point_list
        ##########################################################################################################################
        return data

     

     
