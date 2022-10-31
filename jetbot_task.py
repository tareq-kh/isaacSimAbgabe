from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import create_prim

from omni.isaac.wheeled_robots.robots import WheeledRobot
#from omni.isaac.motion_generation import WheelBasePoseController
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.prims import RigidPrimView

import omni.kit

import gym
from gym import spaces
import numpy as np
import torch
import math

class JetbotTask(BaseTask):

    def __init__(
        self,
        name,
        offset=None) -> None:
    
        # task-specific parameters
        self._jetbot_position = [0.0, 0.0, 0.1]
        self._goal_position = [100.0, 0.0, 0.0]
        self._max_velocity = 5.0
        #self._max_push_effort = 400.0

        # values used for defining RL buffers
        self._num_observations = 16 #5 #2vel +pos+ or +  goal
        self._num_actions = 2
        self._device = "cpu"
        self.num_envs = 1

        # a few class buffers to store RL-related states
        self.obs = torch.zeros((self.num_envs, self._num_observations))
        self.resets = torch.zeros((self.num_envs, 1))
        self.progress_buf = torch.zeros(self.num_envs, device=self._device, dtype=torch.long)

        # set the action and observation space for RL
        self.action_space = spaces.Box(np.ones(self._num_actions) * -1.0, np.ones(self._num_actions) * 1.0)
        self.observation_space = spaces.Box(np.ones(self._num_observations) * -np.Inf, np.ones(self._num_observations) * np.Inf)

        #self.max_velocity = 1
        self.max_angular_velocity = math.pi

        # trigger __init__ of parent class
        BaseTask.__init__(self, name=name, offset=offset)

    def set_up_scene(self, scene) -> None:
        # retrieve file path for the Jetbot USD file
        assets_root_path = get_assets_root_path()
        jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        create_prim(prim_path="/World/Fancy_Robot", prim_type="Xform", position=self._jetbot_position)
        add_reference_to_stage(jetbot_asset_path, "/World/Fancy_Robot")

        # create an ArticulationView wrapper for the jetbot
        self._jetbots = ArticulationView(prim_paths_expr="/World/Fancy_Robot", name="jetbot_view")

        # add the Jetbot ArticulationView and a ground plane to the Scene
        scene.add(self._jetbots)
        scene.add_default_ground_plane()

        # set the default camera viewport position and target
        self.set_initial_camera_params()
        
        #self._jetbot = WheeledRobot(
        #                   prim_path="/World/Fancy_Robot",
        #                   name="fancy_robot",
        #                   wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
        #                   create_robot=True,
        #                   usd_path=jetbot_asset_path,
        #               )
        
        #self._jetbot = scene.get_object("fancy_robot")

        #self._jetbot = ArticulationView(prim_paths_expr="/World/Fancy_Robot*", name="robo_view")
        #self._jetbot_view = ArticulationView(prim_paths_expr="/World/Fancy_Robot*", name="robo_view")
        #scene.add(self._jetbot)
        #scene.add(self._jetbot)
        
        #self._my_controller = DifferentialController(name="simple_control", wheel_radius=0.03, wheel_base=0.1125)
        #self._my_controller = WheelBasePoseController(name="cool_controller",
        #                                                open_loop_wheel_controller=
        #                                                    DifferentialController(name="simple_control",
        #                                                                            wheel_radius=0.03, wheel_base=0.1125),
        #                                            is_holonomic=False)

        #self.goal = scene.add(VisualCuboid(
        #                          prim_path="/World/new_cube_1",
        #                          name="visual_cube",
        #                          position=np.array([0.60, 0.30, 0.05]),
        #                          size=0.1,
        #                          color=np.array([1.0, 0, 0]),
        #            ))
        #block_asset_path = assets_root_path + "/Isaac/Props/Blocks/basic_block.usd"
        create_prim(prim_path="/World/new_cube_1", prim_type="Xform", position=self._goal_position)
        #add_reference_to_stage(block_asset_path, "/World/new_cube_1")
        
        self.goal = RigidPrimView(prim_paths_expr="/World/new_cube_1", name="object_view")
        scene.add(self.goal)
        #)

        # set default camera viewport position and target
        #self.set_initial_camera_params()

    def set_initial_camera_params(self, camera_position=[10, 10, 3], camera_target=[0, 0, 0]):
        viewport = omni.kit.viewport_legacy.get_default_viewport_window()
        viewport.set_camera_position(
            "/OmniverseKit_Persp", camera_position[0], camera_position[1], camera_position[2], True
        )
        viewport.set_camera_target("/OmniverseKit_Persp", camera_target[0], camera_target[1], camera_target[2], True)

    def post_reset(self):
        self._jetbot_left_idx = self._jetbots.get_dof_index("left_wheel_joint")
        self._jetbot_right_idx = self._jetbots.get_dof_index("right_wheel_joint")
        # randomize all envs
        indices = torch.arange(self._jetbots.count, dtype=torch.int64, device=self._device)
        #indices = torch.arange(self._jetbot_view.count, dtype=torch.int64, device=self._device)
        self.reset(indices)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        num_resets = len(env_ids)

        #dof_pos = torch.zeros((num_resets, self._jetbot.num_dof), device=self._device)
        # dof_pos[:, self._jetbot_left_idx]

        # randomize DOF velocities
        #dof_vel = torch.zeros((num_resets, self._jetbots.num_dof), device=self._device)
        #dof_vel[:, self._jetbot_left_idx] = 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        #dof_vel[:, self._jetbot_left_idx] = 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        #self._jetbot.set_joint_positions(dof_pos, indices=indices)
        #self._jetbots.set_joint_velocities(dof_vel, indices=indices)
        #self._jetbots.position = self._jetbot_position
        # bookkeeping
        self.resets[env_ids] = 0
        self.progress_buf[env_ids] = 0

        #poss = torch.tensor([[1, 2, 3]])
        #print(str(indices))
        #np.array([math.sin(alpha) * r, math.cos(alpha) * r, 0.05])
        #self.goal.set_world_poses(poss,indices=indices)
        #self.resets[env_ids] = 0
        #self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.resets.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset(reset_env_ids)

        self._previous_jetbot_position, jetbot_world_orientation = self._jetbots.get_world_poses()
        
        actions = torch.tensor(actions)
        velocities = torch.zeros((self._jetbots.count, self._jetbots.num_dof), dtype=torch.float32, device=self._device)
        velocities_left_wheel = torch.zeros((self._jetbots.count, self._jetbots.num_dof), dtype=torch.float32, device=self._device)
        velocities_right_wheel = torch.zeros((self._jetbots.count, self._jetbots.num_dof), dtype=torch.float32, device=self._device)
        #velocities[:, :] = self._max_velocity * actions
        velocities = self._max_velocity * actions
        #velocities_left_wheel[:, self._jetbot_left_idx] = self._max_velocity * actions[0]
        #velocities_right_wheel[:, self._jetbot_right_idx] = self._max_velocity * actions[1]
        

        indices = torch.arange(self._jetbots.count, dtype=torch.int32, device=self._device)
        #self._jetbots.set_joint_velocities(velocities, indices=indices)
        self._jetbots.set_joint_velocity_targets(velocities, indices=indices)
        #self._jetbots.set_joint_velocity_targets(velocities_left_wheel, indices=indices)
        #self._jetbots.set_joint_velocity_targets(velocities_right_wheel, indices=indices)

    def get_observations(self):
        self.progress_buf[:] += 1
        #self._my_world.render()
        jetbot_world_position, jetbot_world_orientation = self._jetbots.get_world_poses()
        #shape is (M, 6) linear and angular
        #jetbot_velocity = self._jetbots.get_velocities()
        jetbot_linear_velocity = self._jetbots.get_linear_velocities()
        jetbot_angular_velocity = self._jetbots.get_angular_velocities()
        #print("Lvelocity: "+str(jetbot_linear_velocity))
        #print("Avelocity: "+str(jetbot_angular_velocity))
        #print("velocity: "+str(jetbot_velocity))
        #print("Orienv: "+str(jetbot_velocity[:,2:]))

        #goal_world_position, _ = self.goal.get_world_poses()
        #print("goal_world_position: "+str(goal_world_position))

        self.obs[:, 0] = jetbot_world_position[:, 0]
        self.obs[:, 1] = jetbot_world_position[:, 1]
        self.obs[:, 2] = jetbot_world_position[:, 2]
        self.obs[:, 3] = jetbot_world_orientation[:, 0]
        self.obs[:, 4] = jetbot_world_orientation[:, 1]
        self.obs[:, 5] = jetbot_world_orientation[:, 2]
        self.obs[:, 6] = jetbot_world_orientation[:, 3]
        self.obs[:, 7] = jetbot_linear_velocity[:, 0]
        self.obs[:, 8] = jetbot_linear_velocity[:, 1]
        self.obs[:, 9] = jetbot_linear_velocity[:, 2]
        self.obs[:, 10] = jetbot_angular_velocity[:, 0]
        self.obs[:, 11] = jetbot_angular_velocity[:, 1]
        self.obs[:, 12] = jetbot_angular_velocity[:, 2]
        #self.obs[:, 13] = goal_world_position[:, 0]
        #self.obs[:, 14] = goal_world_position[:, 1]
        #self.obs[:, 15] = goal_world_position[:, 2]

        return self.obs

    def calculate_metrics(self) -> None:
        current_jetbot_pos_x = self.obs[:, 0]
        current_jetbot_pos_y = self.obs[:, 1]
        previous_jetbot_pos_x = self._previous_jetbot_position[:, 0]
        previous_jetbot_pos_y = self._previous_jetbot_position[:, 1]
        #goal_world_position_x = self.obs[:, 13]
        #goal_world_position_y = self.obs[:, 14]
        goal_world_position_x = self._goal_position[0]
        goal_world_position_y = self._goal_position[1]
        goal_world_position, _ = self.goal.get_world_poses()
        current_jetbot_position, _ = self._jetbots.get_world_poses()
        #goal_world_position = self.obs[:, 9]

        print(current_jetbot_pos_x)
        print(current_jetbot_pos_y)
        print("\n")
        print(goal_world_position_x)
        print(goal_world_position_y)
        print("\n")

        # Calculate previous distance of the jetbots to the goal
        previous_dist_to_goal_x = torch.square(previous_jetbot_pos_x - goal_world_position_x)
        previous_dist_to_goal_y = torch.square(previous_jetbot_pos_y - goal_world_position_y)
        previous_dist_to_goal = torch.sqrt(previous_dist_to_goal_x + previous_dist_to_goal_y)

        # Calculate current distance of the jetbots to the goal
        current_dist_to_goal_x = torch.square(current_jetbot_pos_x - goal_world_position_x)
        current_dist_to_goal_y = torch.square(current_jetbot_pos_y - goal_world_position_y)
        self._current_dist_to_goal = torch.sqrt(current_dist_to_goal_x + current_dist_to_goal_y)

        print(self._current_dist_to_goal)
        print("\n")

        # Calculate the reward based on the previous and current distance to the goal
        reward = previous_dist_to_goal - self._current_dist_to_goal
        #reward = -1 * self._current_dist_to_goal

        print(reward)
        print("\n")

        # compute reward based on the previous and current distance to the goal
        #previous_dist_to_goal = np.linalg.norm(goal_world_position - self.obs[:, 0])
        #previous_dist_to_goal = np.linalg.norm(goal_world_position - self._previous_jetbot_position)
        #previous_dist_to_goal = np.linalg.norm(self._goal_position - self._previous_jetbot_position)
        #previous_dist_to_goal = math.sqrt((current_jetbot_pos_x))
        #self._current_dist_to_goal = np.linalg.norm(goal_world_position - current_jetbot_position)
        #self._current_dist_to_goal = np.linalg.norm(self._goal_position - current_jetbot_position)
        #reward = previous_dist_to_goal - self._current_dist_to_goal
        #reward_tensor = torch.tensor(reward)
        #rewards = torch.zeros(self.num_envs)
        #print(reward_tensor)

        #return reward_tensor
        return reward.item()

    def is_done(self) -> None:
        #current_jetbot_pos = self.obs[:, 0]
        #goal_world_position = self.obs[:, 4]
        #current_dist_to_goal = torch.from_numpy(self._current_dist_to_goal)
        progress_buf = self.progress_buf

        # reset the robot if cart has reached reset_dist or pole is too far from upright
        #resets = torch.where(torch.abs(current_dist_to_goal) > 0.1, 1, 0)
        resets = torch.where(self._current_dist_to_goal < 0.1, 1, 0)
        resets = torch.where(progress_buf >= 1000 - 1, 1, 0)
        #resets = torch.zeros(self.num_envs)

        #resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        self.resets = resets
        #print(resets)

        return resets.item()
