from random import uniform

from isaacgym import gymapi, gymutil, gymtorch
import numpy as np
import torch, math
import kobuki_controller as controller
import map_factory as map

gym = gymapi.acquire_gym()


class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments





asset_descriptors = [
    AssetDesc("urdf/kobuki_standalone.urdf", False)
]

args = gymutil.parse_arguments(
    description="Kobuki: Animate degree-of-freedom ranges",
    custom_parameters=[
        {"name": "--asset_id", "type": int, "default": 0, "help": "Asset id (0 - %d)" % (len(asset_descriptors) - 1)},
        {"name": "--speed_scale", "type": float, "default": 1.0, "help": "Animation speed scale"},
        {"name": "--show_axis", "action": "store_true", "help": "Visualize DOF axis"}])

# get default set of parameters
sim_params = gymapi.SimParams()

# set common parameters
sim_params.dt = 1 / 60
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
# sim_params.use_gpu_pipeline = True
# set PhysX-specific parameters
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0

# set Flex-specific parameters
sim_params.flex.solver_type = 5
sim_params.flex.num_outer_iterations = 4
sim_params.flex.num_inner_iterations = 20
sim_params.flex.relaxation = 0.8
sim_params.flex.warm_start = 0.5


# create sim with these parameters
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)

# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0

# create the ground plane
gym.add_ground(sim, plane_params)

asset_root = "../../assets"
asset_file = "urdf/kobuki_standalone.urdf"
asset = gym.load_asset(sim, asset_root, asset_file)


# set up the env grid
num_envs = 16
envs_per_row = 4
env_spacing = 10.0

env_lower = gymapi.Vec3(0, 0, 0)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# cache some common handles for later use
envs = []
actor_handles = []

cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)
asset_options = gymapi.AssetOptions()
asset_options.density = 10.0
# create and populate the environments
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)

    #height = np.random.uniform(1.0, 2.5)
    arrs= map.get_lines_positions()
    poss_lenths= map.get_lines_poss_lengths(arrs[0],arrs[1])
    poss= poss_lenths[0]
    lenths= poss_lenths[1]

    p1 = [0.0, 0.0]
    p2 = [3.0, 2.0]
    p = map.get_line_pos(p1, p2)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.0)


    p1_pose = gymapi.Transform()
    p1_pose.p = gymapi.Vec3(p1[0], p1[1],0.0)

    p2_pose = gymapi.Transform()
    p2_pose.p = gymapi.Vec3(p2[0], p2[1],0.0)

    box_pose = gymapi.Transform()
    box_pose.p = gymapi.Vec3(p[0], p[1], 0.0)
    theta = math.atan2(p2[1], p2[0])#math.pi/2.0
    print(theta)
    box_pose.r = gymapi.Quat(math.cos(theta / 2),   math.sin(theta / 2),   0*math.sin(theta / 2), 0*math.sin(theta / 2))
    print(box_pose.r)
    #box_pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)

    box_asset = gym.create_box(sim, map.get_line_length(p1,p2), 0.2, 0.7, asset_options)
    capsule_asset = gym.create_sphere(sim, 0.15, asset_options)

    # actor_box = gym.create_actor(env, box_asset, pose1, "MyActor2", 0, 2)
    actor_handle = gym.create_actor(env, asset, pose, "MyActor", i, 1)
    box_handle = gym.create_actor(env, box_asset, box_pose, "MyActor", i, 1)
    p1_handle = gym.create_actor(env, capsule_asset, p1_pose, "MyActor", i, 1)
    p2_handle = gym.create_actor(env, capsule_asset, p2_pose, "MyActor", i, 1)
    actor_handles.append(actor_handle)

 #run sim
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim);
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)


"""
env = gym.create_env(sim, lower, upper, 8)




pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
pose.r = gymapi.Quat(-0.0, 0.0, 0.0, 0.707107)

pose1 = gymapi.Transform()
pose1.p = gymapi.Vec3(1.0, 0.0, 0.0)
pose1.r = gymapi.Quat(-0.0, 0.0, 0.0, 0.707107)

actor_handle = gym.create_actor(env, asset, pose, "MyActor", 0, 1)

num_bodies = gym.get_actor_rigid_body_count(env, actor_handle)
num_joints = gym.get_actor_joint_count(env, actor_handle)
num_dofs = gym.get_actor_dof_count(env, actor_handle)

print("num_bodies: ", num_bodies)
print("num_joints: ", num_joints)
print("num_dofs: ", num_dofs)
asset_options = gymapi.AssetOptions()
asset_options.density = 10.0

box_asset = gym.create_box(sim, 1, 0.3, 1, asset_options)
sphere_asset = gym.create_sphere(sim, 0.2, asset_options)
capsule_asset = gym.create_capsule(sim, 0.2, 0.3, asset_options)

# actor_box = gym.create_actor(env, box_asset, pose1, "MyActor2", 0, 2)

cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)

# configure the joints for effort control mode (once)
props = gym.get_actor_dof_properties(env, actor_handle)
props["driveMode"].fill(gymapi.DOF_MODE_VEL)
props["stiffness"].fill(0.0)
props["damping"].fill(600.0)
gym.set_actor_dof_properties(env, actor_handle, props)

# apply efforts (every frame)
vel_targets = np.random.uniform(3.14, 3.14, num_dofs).astype('f')
print(vel_targets)
gym.set_actor_dof_velocity_targets(env, actor_handle, vel_targets)

gym.prepare_sim(sim)

# acquire root state tensor descriptor
_root_tensor = gym.acquire_actor_root_state_tensor(sim)

# wrap it in a PyTorch Tensor and create convenient views
root_tensor = gymtorch.wrap_tensor(_root_tensor)
root_positions = root_tensor[:, 0:3]
root_orientations = root_tensor[:, 3:7]
root_linvels = root_tensor[:, 7:10]
root_angvels = root_tensor[:, 10:13]
saved_root_tensor = root_tensor.clone()
step = 0
offsets = torch.tensor([0, 1, 0])
root_positions += offsets

speed = (0,3.0)

while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)

    step += 1
    if step % 100 == 0:
        print((uniform(speed[0]/0.035, speed[0]/0.035), uniform(speed[1]/0.035, speed[1]/0.035)))
        speed = controller.getSpeed(uniform((speed[0]/0.035)*1.1, (speed[0]/0.035)*0.9), uniform(speed[1]/0.035+5, speed[1]/0.035-5))
        vel_targets = np.random.uniform(speed[0], speed[1], num_dofs).astype('f')

        print("SPEED",speed)
        print("VEL_TARGET",vel_targets)
        print(speed[0], speed[1])

        gym.set_actor_dof_velocity_targets(env, actor_handle, vel_targets)
    root_positions += offsets
    # print("reset")
    gym.fetch_results(sim, True)
    # print(root_linvels[0][0])
    # print(root_angvels)
    # print(gym.get_sim_rigid_body_count(sim))
    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
"""