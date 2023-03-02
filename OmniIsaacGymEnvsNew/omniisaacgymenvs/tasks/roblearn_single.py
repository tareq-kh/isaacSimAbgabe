from omniisaacgymenvs.tasks.base.rl_task_single import RLTaskSingle
from omniisaacgymenvs.robots.articulations.cartpole import Cartpole
from omniisaacgymenvs.robots.articulations.jetbot import Jetbot
import omniisaacgymenvs.tasks.utils.roblearn.map_factory as map


from omni.isaac.core.utils.nucleus import get_assets_root_path

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.objects import VisualCuboid
from pxr import UsdGeom, Gf, UsdPhysics , UsdShade, Sdf, Usd

import numpy as np
import torch
import math

import omni 
import random 

num_steps_per_minute = 0 
avg_steps_per_second = 0

class RoblearnSingleTask(RLTaskSingle):
    def __init__(
        self,
        name,                # name of the Task
        sim_config,    # SimConfig instance for parsing cfg
        env,          # env instance of VecEnvBase or inherited class
        offset=None               # transform offset in World
    ) -> None:
         
        # parse configurations, set task-specific members
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._jetbot_positions = torch.tensor([0.0, 1.0, 0.0])
        self.num = 0

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_episode_length = 50000

        # Get the lidar parameters
        self._lidar_min_range = self._task_cfg["lidar"]["min_range"]
        self._lidar_max_range = self._task_cfg["lidar"]["max_range"]
        self._lidar_draw_points = self._task_cfg["lidar"]["draw_points"]
        self._lidar_draw_lines = self._task_cfg["lidar"]["draw_lines"]
        self._lidar_horizontal_fov = self._task_cfg["lidar"]["horizontal_fov"]
        self._lidar_vertical_fov = self._task_cfg["lidar"]["vertical_fov"]
        self._lidar_horizontal_resolution = self._task_cfg["lidar"]["horizontal_resolution"]
        self._lidar_vertical_resolution = self._task_cfg["lidar"]["vertical_resolution"]
        self._lidar_rotation_rate = self._task_cfg["lidar"]["rotation_rate"]
        self._lidar_high_lod = self._task_cfg["lidar"]["high_lod"]
        self._lidar_yaw_offset = self._task_cfg["lidar"]["yaw_offset"]
        self._lidar_enable_semantics = self._task_cfg["lidar"]["enable_semantics"]

        self._num_observations = 16 + (int)(self._lidar_horizontal_fov / self._lidar_horizontal_resolution)
        self._num_actions = 2
        self._max_velocity = 10.0
    
        self._goal_position = [[0.5, 1.0, 0.0],[0.5,5.0,0],[-4.0,5.0,0],[-4.0,8.0,0],[1,8.0,0],[3.0,9.3,0.0]]
        self.active_goals = torch.zeros(self.num_envs)
        self.goal_abs = [[35.5, -34.0], [35.5, -30.0], [31.0, -30.0], [31.0, -27.0], [36.0, -27.0], [38.0, -25.7], [35.5, -24.0], [35.5, -20.0], [31.0, -20.0], [31.0, -17.0], [36.0, -17.0], [38.0, -15.7], [35.5, -14.0], [35.5, -10.0], [31.0, -10.0], [31.0, -7.0], [36.0, -7.0], [38.0, -5.7], [35.5, -4.0], [35.5, 0.0], [31.0, 0.0], [31.0, 3.0], [36.0, 3.0], [38.0, 4.33], [35.5, 6.0], [35.5, 10.0], [31.0, 10.0], [31.0, 13.0], [36.0, 13.0], [38.0, 14.33], [35.5, 16.0], [35.5, 20.0], [31.0, 20.0], [31.0, 23.0], [36.0, 23.0], [38.0, 24.33], [35.5, 26.0], [35.5, 30.0], [31.0, 30.0], [31.0, 33.0], [36.0, 33.0], [38.0, 34.3], [35.5, 36.0], [35.5, 40.0], [31.0, 40.0], [31.0, 43.0], [36.0, 43.0], [38.0, 44.3], [25.5, -34.0], [25.5, -30.0], [21.0, -30.0], [21.0, -27.0], [26.0, -27.0], [28.0, -25.7], [25.5, -24.0], [25.5, -20.0], [21.0, -20.0], [21.0, -17.0], [26.0, -17.0], [28.0, -15.7], [25.5, -14.0], [25.5, -10.0], [21.0, -10.0], [21.0, -7.0], [26.0, -7.0], [28.0, -5.7], [25.5, -4.0], [25.5, 0.0], [21.0, 0.0], [21.0, 3.0], [26.0, 3.0], [28.0, 4.33], [25.5, 6.0], [25.5, 10.0], [21.0, 10.0], [21.0, 13.0], [26.0, 13.0], [28.0, 14.33], [25.5, 16.0], [25.5, 20.0], [21.0, 20.0], [21.0, 23.0], [26.0, 23.0], [28.0, 24.33], [25.5, 26.0], [25.5, 30.0], [21.0, 30.0], [21.0, 33.0], [26.0, 33.0], [28.0, 34.3], [25.5, 36.0], [25.5, 40.0], [21.0, 40.0], [21.0, 43.0], [26.0, 43.0], [28.0, 44.3], [15.5, -34.0], [15.5, -30.0], [11.0, -30.0], [11.0, -27.0], [16.0, -27.0], [18.0, -25.7], [15.5, -24.0], [15.5, -20.0], [11.0, -20.0], [11.0, -17.0], [16.0, -17.0], [18.0, -15.7], [15.5, -14.0], [15.5, -10.0], [11.0, -10.0], [11.0, -7.0], [16.0, -7.0], [18.0, -5.7], [15.5, -4.0], [15.5, 0.0], [11.0, 0.0], [11.0, 3.0], [16.0, 3.0], [18.0, 4.33], [15.5, 6.0], [15.5, 10.0], [11.0, 10.0], [11.0, 13.0], [16.0, 13.0], [18.0, 14.33], [15.5, 16.0], [15.5, 20.0], [11.0, 20.0], [11.0, 23.0], [16.0, 23.0], [18.0, 24.33], [15.5, 26.0], [15.5, 30.0], [11.0, 30.0], [11.0, 33.0], [16.0, 33.0], [18.0, 34.3], [15.5, 36.0], [15.5, 40.0], [11.0, 40.0], [11.0, 43.0], [16.0, 43.0], [18.0, 44.3], [5.5, -34.0], [5.5, -30.0], [1.0, -30.0], [1.0, -27.0], [6.0, -27.0], [8.0, -25.7], [5.5, -24.0], [5.5, -20.0], [1.0, -20.0], [1.0, -17.0], [6.0, -17.0], [8.0, -15.7], [5.5, -14.0], [5.5, -10.0], [1.0, -10.0], [1.0, -7.0], [6.0, -7.0], [8.0, -5.7], [5.5, -4.0], [5.5, 0.0], [1.0, 0.0], [1.0, 3.0], [6.0, 3.0], [8.0, 4.33], [5.5, 6.0], [5.5, 10.0], [1.0, 10.0], [1.0, 13.0], [6.0, 13.0], [8.0, 14.33], [5.5, 16.0], [5.5, 20.0], [1.0, 20.0], [1.0, 23.0], [6.0, 23.0], [8.0, 24.33], [5.5, 26.0], [5.5, 30.0], [1.0, 30.0], [1.0, 33.0], [6.0, 33.0], [8.0, 34.3], [5.5, 36.0], [5.5, 40.0], [1.0, 40.0], [1.0, 43.0], [6.0, 43.0], [8.0, 44.3], [-4.5, -34.0], [-4.5, -30.0], [-9.0, -30.0], [-9.0, -27.0], [-4.0, -27.0], [-2.0, -25.7], [-4.5, -24.0], [-4.5, -20.0], [-9.0, -20.0], [-9.0, -17.0], [-4.0, -17.0], [-2.0, -15.7], [-4.5, -14.0], [-4.5, -10.0], [-9.0, -10.0], [-9.0, -7.0], [-4.0, -7.0], [-2.0, -5.7], [-4.5, -4.0], [-4.5, 0.0], [-9.0, 0.0], [-9.0, 3.0], [-4.0, 3.0], [-2.0, 4.33], [-4.5, 6.0], [-4.5, 10.0], [-9.0, 10.0], [-9.0, 13.0], [-4.0, 13.0], [-2.0, 14.33], [-4.5, 16.0], [-4.5, 20.0], [-9.0, 20.0], [-9.0, 23.0], [-4.0, 23.0], [-2.0, 24.33], [-4.5, 26.0], [-4.5, 30.0], [-9.0, 30.0], [-9.0, 33.0], [-4.0, 33.0], [-2.0, 34.3], [-4.5, 36.0], [-4.5, 40.0], [-9.0, 40.0], [-9.0, 43.0], [-4.0, 43.0], [-2.0, 44.3], [-14.5, -34.0], [-14.5, -30.0], [-19.0, -30.0], [-19.0, -27.0], [-14.0, -27.0], [-12.0, -25.7], [-14.5, -24.0], [-14.5, -20.0], [-19.0, -20.0], [-19.0, -17.0], [-14.0, -17.0], [-12.0, -15.7], [-14.5, -14.0], [-14.5, -10.0], [-19.0, -10.0], [-19.0, -7.0], [-14.0, -7.0], [-12.0, -5.7], [-14.5, -4.0], [-14.5, 0.0], [-19.0, 0.0], [-19.0, 3.0], [-14.0, 3.0], [-12.0, 4.33], [-14.5, 6.0], [-14.5, 10.0], [-19.0, 10.0], [-19.0, 13.0], [-14.0, 13.0], [-12.0, 14.33], [-14.5, 16.0], [-14.5, 20.0], [-19.0, 20.0], [-19.0, 23.0], [-14.0, 23.0], [-12.0, 24.33], [-14.5, 26.0], [-14.5, 30.0], [-19.0, 30.0], [-19.0, 33.0], [-14.0, 33.0], [-12.0, 34.3], [-14.5, 36.0], [-14.5, 40.0], [-19.0, 40.0], [-19.0, 43.0], [-14.0, 43.0], [-12.0, 44.3], [-24.5, -34.0], [-24.5, -30.0], [-29.0, -30.0], [-29.0, -27.0], [-24.0, -27.0], [-22.0, -25.7], [-24.5, -24.0], [-24.5, -20.0], [-29.0, -20.0], [-29.0, -17.0], [-24.0, -17.0], [-22.0, -15.7], [-24.5, -14.0], [-24.5, -10.0], [-29.0, -10.0], [-29.0, -7.0], [-24.0, -7.0], [-22.0, -5.7], [-24.5, -4.0], [-24.5, 0.0], [-29.0, 0.0], [-29.0, 3.0], [-24.0, 3.0], [-22.0, 4.33], [-24.5, 6.0], [-24.5, 10.0], [-29.0, 10.0], [-29.0, 13.0], [-24.0, 13.0], [-22.0, 14.33], [-24.5, 16.0], [-24.5, 20.0], [-29.0, 20.0], [-29.0, 23.0], [-24.0, 23.0], [-22.0, 24.33], [-24.5, 26.0], [-24.5, 30.0], [-29.0, 30.0], [-29.0, 33.0], [-24.0, 33.0], [-22.0, 34.3], [-24.5, 36.0], [-24.5, 40.0], [-29.0, 40.0], [-29.0, 43.0], [-24.0, 43.0], [-22.0, 44.3], [-34.5, -34.0], [-34.5, -30.0], [-39.0, -30.0], [-39.0, -27.0], [-34.0, -27.0], [-32.0, -25.7], [-34.5, -24.0], [-34.5, -20.0], [-39.0, -20.0], [-39.0, -17.0], [-34.0, -17.0], [-32.0, -15.7], [-34.5, -14.0], [-34.5, -10.0], [-39.0, -10.0], [-39.0, -7.0], [-34.0, -7.0], [-32.0, -5.7], [-34.5, -4.0], [-34.5, 0.0], [-39.0, 0.0], [-39.0, 3.0], [-34.0, 3.0], [-32.0, 4.33], [-34.5, 6.0], [-34.5, 10.0], [-39.0, 10.0], [-39.0, 13.0], [-34.0, 13.0], [-32.0, 14.33], [-34.5, 16.0], [-34.5, 20.0], [-39.0, 20.0], [-39.0, 23.0], [-34.0, 23.0], [-32.0, 24.33], [-34.5, 26.0], [-34.5, 30.0], [-39.0, 30.0], [-39.0, 33.0], [-34.0, 33.0], [-32.0, 34.3], [-34.5, 36.0], [-34.5, 40.0], [-39.0, 40.0], [-39.0, 43.0], [-34.0, 43.0], [-32.0, 44.3]]
        
        
        RLTaskSingle.__init__(self, name, env)
        return
    
     
    def get_jetbot(self):
        """
        Initialize robots in an Articulation View
        for Articulation view, see Isaacgym docs
        """ 
        assets_root_path = get_assets_root_path()
        jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        jetbot = Jetbot(prim_path=self.default_zero_env_path + "/Jetbot" , name="Jetbot",usd_path=jetbot_asset_path, translation=self._jetbot_positions)
        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings("Jetbot", get_prim_at_path(jetbot.prim_path), self._sim_config.parse_actor_config("Jetbot"))

    
    def create_lidars(self):
        """
        inits/parametrize laser scaner for every robot 
        """ 
        # Imports the python bindings to interact with lidar sensor
        from omni.isaac.range_sensor import _range_sensor 

        # Used to access Geometry
        stage = omni.usd.get_context().get_stage()  

        # Used to interact with the LIDAR                    
        self.lidarInterface = _range_sensor.acquire_lidar_sensor_interface() 
        
        base_prim_path = "/World/envs"
        for i in range(self._num_envs):

            env_path = "/env_" + str(i)

            jetbot_path = "/Jetbot"
            parent_prim = base_prim_path + env_path + jetbot_path + "/chassis"
            lidar_path = "/LidarName"
            result, prim = omni.kit.commands.execute(
                        "RangeSensorCreateLidar",
                        path=lidar_path,
                        parent=parent_prim,
                        min_range=self._lidar_min_range,
                        max_range=self._lidar_max_range,
                        draw_points=self._lidar_draw_points,
                        draw_lines=self._lidar_draw_lines,
                        horizontal_fov=self._lidar_horizontal_fov,
                        vertical_fov=self._lidar_vertical_fov,
                        horizontal_resolution=self._lidar_horizontal_resolution,
                        vertical_resolution=self._lidar_vertical_resolution,
                        rotation_rate=self._lidar_rotation_rate,
                        high_lod=self._lidar_high_lod,
                        yaw_offset=self._lidar_yaw_offset,
                        enable_semantics=self._lidar_enable_semantics
                    )




    def set_up_map(self, stage, map_id):
        """
        :param stage: the stage of the simulation
        :param map_id: Id of the required map  

        takes maps fro the procedural map generator as input and transforms them into 3D maps.
        """
        arrs= map.get_lines_positions(map_id)
        poss_lenths= map.get_lines_poss_lengths(arrs[0],arrs[1])
        poss= poss_lenths[0] 
        lenths= poss_lenths[1]
        p10 = [3.0, 4.0]
        p20 = [6., 6] 
        p1 = [-2.0, -1.0] 
        p2 = [1., 1] 
        pos0 = map.get_line_pos(p10,p20) 
        pos= map.get_line_pos(p1,p2) 
        length = map.get_line_length(p1,p2)
        i=0       
        for pos,length,p1,p2 in zip(poss,lenths,arrs[0],arrs[1]):

            theta = math.atan((p2[1] - p1[1])/(p2[0] - p1[0]))
            cubePath = f"/World/envs/env_0/Cube{i}"
            cubeGeom = UsdGeom.Cube.Define(stage, cubePath)
            cubePrim = stage.GetPrimAtPath(cubePath)
            cubeGeom.AddTranslateOp().Set(Gf.Vec3f(-self._env_spacing/2 +pos[0], pos[1], 0.0))
            cubeGeom.CreateSizeAttr(0.1)
            cubeGeom.AddRotateZOp().Set(theta * 180/ math.pi)
            cubeGeom.AddScaleOp().Set(Gf.Vec3f(map.get_line_length(p1,p2)*self._env_spacing , 2, 6))
            collisionAPI = UsdPhysics.CollisionAPI.Apply(cubePrim)      
            i = i+1 

    def set_up_goals (self, stage):
        """
        generates target areas for the robots to drive to
        the areas are represented throw  spheres in the sim 

        :param stage: the stage of the simulation
        """
        for i in range(len(self._goal_position)):
            cubePath = f"/World/envs/env_0/GoalCube"+str(i)
            cubeGeom = UsdGeom.Sphere.Define(stage, cubePath)
            cubePrim = stage.GetPrimAtPath(cubePath)
            # cubeGeom.AddTranslateOp().Set(Gf.Vec3f(-self._env_spacing/2 + 0.5, 1.0, 0.0))
            cubeGeom.AddTranslateOp().Set(Gf.Vec3f(self._goal_position[i]))
            #cubeGeom.CreateSizeAttr(0.1)
            cubeGeom.AddScaleOp().Set(Gf.Vec3f(0.1 , 0.1, 0.1))
            #cube_prim.GetAttribute("physics:collisionEnabled").Set(True)
            cubePrim.GetAttribute("visibility").Set("invisible")


            # Get the Material prim for the cube geometry
            materialPath = f"{cubePath}/Material"
            materialPrim = stage.DefinePrim(materialPath, "Material")

            # Set the material's display color to red
            material = UsdShade.Material.Define(stage, materialPath)
            material.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 0, 0))

            # Bind the material to the cube geometry
            materialBinding = UsdShade.MaterialBindingAPI(cubePrim)
            materialBinding.Bind(material)



    def reset_outer_walls_collision(self,stage):
        """
        resets the collision parameters of the outer walls 
        used to fix a coliision bug, where the outer walls collision doesnt work properly

        :param stage: the stage of the simulation
        """  
        for cube_num in range(4):
            cube_prim = stage.GetPrimAtPath(f"/World/envs/env_{0}/Cube{cube_num}")
            cube_prim.GetAttribute("physics:collisionEnabled").Set(False)
            cube_prim.GetAttribute("physics:collisionEnabled").Set(True)

    def set_up_scene(self, scene) -> None:
        """
        sets up the scene
        """
        stage = omni.usd.get_context().get_stage()

        self.get_jetbot()
        self.create_lidars()
        self.set_up_map(stage,1)
        self.reset_outer_walls_collision(stage)
        
        self.set_up_goals(stage)
        self.reset_outer_walls_collision(stage)
        super().set_up_scene(scene)
        for env_num in range(self._num_envs):
            for cube_num in range(4):
                cube_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_num}/Cube{cube_num}")
                cube_prim.GetAttribute("physics:collisionEnabled").Set(False)
                cube_prim.GetAttribute("physics:collisionEnabled").Set(True)

        for env_num in range(self._num_envs):
            for cube_num in range(4):
                cube_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_num}/Cube{cube_num}")
                cube_prim.GetAttribute("physics:collisionEnabled").Set(False)
                cube_prim.GetAttribute("physics:collisionEnabled").Set(True)
        for env_num in range(self._num_envs):
            for cube_num in range(4):
                cube_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_num}/Cube{cube_num}")
                cube_prim.GetAttribute("physics:collisionEnabled").Set(False)
                cube_prim.GetAttribute("physics:collisionEnabled").Set(True)
        self._jetbots = ArticulationView(prim_paths_expr="/World/envs/.*/Jetbot", name="jetbot_view")      
        scene.add(self._jetbots)
        
        return

    def pre_physics_step(self, actions) -> None:
        """
        updates reset buffer
        scale/apply velocity actions

        :param actions: action values generated by the NN  
        """
        print("before squeez:" ,self.reset_buf)
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        print("after squeez:" ,self.reset_buf.nonzero(as_tuple=False).squeeze(-1))

        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # save current position
        self._previous_jetbot_position, jetbot_world_orientation = self._jetbots.get_world_poses()

        # scale and apply actions
        actions = actions.to(self._device)
        velocities = torch.zeros((self._jetbots.count, self._jetbots.num_dof), dtype=torch.float32, device=self._device)
        velocities =self._max_velocity * actions
        indices = torch.arange(self._jetbots.count, dtype=torch.int32, device=self._device)
        self._jetbots.set_joint_velocity_targets(velocities, indices=indices)

    def get_lidar_data(self):
        """
        gets the distance data from all active Lidar sensors 
        """
        base_prim_path = "/World/envs"
        distances_buf = torch.empty(self.num_envs, (int)(self._lidar_horizontal_fov / self._lidar_horizontal_resolution))
    
        for i in range(self._num_envs):

            env_path = "/env_" + str(i)
            jetbot_path = "/Jetbot"
            parent_prim = base_prim_path + env_path + jetbot_path + "/chassis"
            lidar_path = parent_prim + "/LidarName"

            distances = self.lidarInterface.get_linear_depth_data(lidar_path)
            distances_torch = torch.from_numpy(distances)
            distances_flat = torch.flatten(distances_torch)
            distances_scaled = (distances_flat - self._lidar_min_range) / (self._lidar_max_range - self._lidar_min_range)
            
            distances_buf[i:] = distances_scaled

        return distances_buf
    
    def get_goal_pos_old(self):
        x_tensor = torch.empty(self._num_envs, len(self._goal_position))
        y_tensor = torch.empty(self._num_envs, len(self._goal_position))
        l=[]

        stage = omni.usd.get_context().get_stage()
        for i in range(self._num_envs): 
            for n in range (len(self._goal_position)) :
                # Get the prim for the cube geometry
                cubePath = "/World/envs/env_"+str(i)+"/GoalCube"+str(n)
                cubePrim = stage.GetPrimAtPath(cubePath)
                timeCode = Usd.TimeCode.Default()
                # Create an XformCache for the stage
                xformCache = UsdGeom.XformCache(Usd.TimeCode.Default())

                # Get the global transform for the cube prim
                cubeTransform = xformCache.GetLocalToWorldTransform(cubePrim)

                # Get the translation component of the transform
                cubeTranslation = Gf.Matrix4d(cubeTransform).ExtractTranslation()

                # Print the position of the cube
                x_tensor[i,n] = cubeTranslation[0]
                y_tensor[i,n] = cubeTranslation[1]
                l.append([cubeTranslation[0],cubeTranslation[1]])
                #print("Cube position:", cubeTranslation)
        print("tensor::::::", x_tensor, y_tensor)
        print("llllllll:",l)

        return x_tensor,y_tensor


    def get_goal_pos(self):

        x_tensor = torch.empty(self._num_envs)
        y_tensor = torch.empty(self._num_envs)

        for env in range(self._num_envs):
            active = int(self.active_goals.data[env])
            #print(active)
            value =self.goal_abs[env * len(self._goal_position) + active]
            x_tensor[env]=value[0]
            y_tensor[env]=value[1]
        return x_tensor, y_tensor



    def get_observations(self) -> dict:
        """
        update and get observation_buffer
        """

        jetbot_world_position, jetbot_world_orientation = self._jetbots.get_world_poses()
        jetbot_linear_velocity = self._jetbots.get_linear_velocities()
        jetbot_angular_velocity = self._jetbots.get_angular_velocities()
        jetbot_lidar = self.get_lidar_data()

        goal_x,goal_y = self.get_goal_pos()
        #print("goal_world_position: "+str(goal_world_position))
        

        self.obs_buf[:, 0] = jetbot_world_position[:, 0]
        self.obs_buf[:, 1] = jetbot_world_position[:, 1]
        self.obs_buf[:, 2] = jetbot_world_position[:, 2]
        self.obs_buf[:, 3] = jetbot_world_orientation[:, 0]
        self.obs_buf[:, 4] = jetbot_world_orientation[:, 1]
        self.obs_buf[:, 5] = jetbot_world_orientation[:, 2]
        self.obs_buf[:, 6] = jetbot_world_orientation[:, 3]
        self.obs_buf[:, 7] = jetbot_linear_velocity[:, 0]
        self.obs_buf[:, 8] = jetbot_linear_velocity[:, 1]
        self.obs_buf[:, 9] = jetbot_linear_velocity[:, 2]
        self.obs_buf[:, 10] = jetbot_angular_velocity[:, 0]
        self.obs_buf[:, 11] = jetbot_angular_velocity[:, 1]
        self.obs_buf[:, 12] = jetbot_angular_velocity[:, 2]
        self.obs_buf[:, 13] = goal_x
        self.obs_buf[:, 14] = goal_y
        #self.obs_buf[:, 15] = goal_world_position[:, 2]
        self.obs_buf[:, 16:] = jetbot_lidar

        #print("position_bot: ", jetbot_world_position)
        #print("position_x_goal: ", self.obs_buf[:, 0])
        observations = {
            self._jetbots.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations


    def post_reset(self):
        """
        reset all envs
        """
        print("resetting envXXXXXXXXXXXXXx")
        # randomize all envs
        indices = torch.arange(self._jetbots.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)
    


    def calculate_metrics(self) -> None:
        """
        calculate the Reward
        """
        current_jetbot_pos_x = self.obs_buf[:, 0]
        current_jetbot_pos_y = self.obs_buf[:, 1]
        previous_jetbot_pos_x = self._previous_jetbot_position[:, 0]
        previous_jetbot_pos_y = self._previous_jetbot_position[:, 1]
        goal_world_position_x = self.obs_buf[:,13]
        goal_world_position_y = self.obs_buf[:,14]

        #print("current_jetbot_pos : ", current_jetbot_pos_x,",",current_jetbot_pos_y)
        #print("goal_world_position_x: ",goal_world_position_x, ", ", goal_world_position_y)

        # Calculate previous distance of the jetbots to the goal
        previous_dist_to_goal_x = torch.square(previous_jetbot_pos_x - goal_world_position_x)
        previous_dist_to_goal_y = torch.square(previous_jetbot_pos_y - goal_world_position_y)
        previous_dist_to_goal = torch.sqrt(previous_dist_to_goal_x + previous_dist_to_goal_y)

        # Calculate current distance of the jetbots to the goal
        current_dist_to_goal_x = torch.square(current_jetbot_pos_x - goal_world_position_x)
        current_dist_to_goal_y = torch.square(current_jetbot_pos_y - goal_world_position_y)
        self._current_dist_to_goal = torch.sqrt(current_dist_to_goal_x + current_dist_to_goal_y)

        #print("_current_dist_to_goal: ",self._current_dist_to_goal)

        reward = previous_dist_to_goal - self._current_dist_to_goal

        reward = torch.where(self._current_dist_to_goal < self._reset_dist, torch.ones_like(reward) * 2.0, reward)
        print("reward: ",reward)
        print("activegoals:", self.active_goals)
        print("_current_dist_to_goal", self._current_dist_to_goal)
        
        self.rew_buf[:] = reward



    def is_done(self) -> None:

        """
        update reset buffer with 1 for each env that meets the reset condition, otherwise 0 
        """
        resets = torch.where(self._current_dist_to_goal < self._reset_dist, 1, 0)
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)      
        self.reset_buf[:] = resets


    def reset_idx(self, env_ids):
        """
        :param env_ids: tensor of env ids that have to be reseted
        """
        stage = omni.usd.get_context().get_stage()
        num_resets = len(env_ids)
        indices = env_ids.to(dtype=torch.int32)
        print(indices)
        print(env_ids)
        for env_id in indices:
            
            self.active_goals[env_id] = (self.active_goals[env_id] +1) % len(self._goal_position)
            oldcubePath = f"/World/envs/env_"+str(env_id.item())+"/GoalCube"+str(int(self.active_goals[env_id].item() - 1))  
            cubePath = f"/World/envs/env_"+str(env_id.item())+"/GoalCube"+str(int(self.active_goals[env_id].item()))
            print("cubePath",cubePath)
            oldcubePrim = stage.GetPrimAtPath(oldcubePath)
            cubePrim = stage.GetPrimAtPath(cubePath)
            # cubeGeom.AddTranslateOp().Set(Gf.Vec3f(-self._env_spacing/2 + 0.5, 1.0, 0.0))
           
            #cubeGeom.CreateSizeAttr(0.1)
          
            #cube_prim.GetAttribute("physics:collisionEnabled").Set(True)
            oldcubePrim.GetAttribute("visibility").Set("invisible")
            cubePrim.GetAttribute("visibility").Set("inherited")
            #0-> 0+1 % 6 = 1 
            # 10 war 0-> 1%
            print("resetting ", env_ids)
        
        for i in env_ids:
            print("resettet env it", i)
            cube_num = random.randint(4, 27)
            cube_prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Cube{cube_num}")

            #cube_prim.GetAttribute("physics:collisionEnabled").Set(False)
            cubattr = cube_prim.GetAttribute("physics:collisionEnabled")
            if cubattr.Get():
                cubattr.Set(False)
                cube_prim.GetAttribute("visibility").Set("invisible")
            else: 
                cubattr.Set(True)
                cube_prim.GetAttribute("visibility").Set("inherited")

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

