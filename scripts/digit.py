import math
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments()
args.physics_engine = gymapi.SIM_FLEX

# simulation parameters
sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 3
sim_params.flex.solver_type = 5
sim_params.flex.num_outer_iterations = 4
sim_params.flex.num_inner_iterations = 20
sim_params.flex.relaxation = 0.8
sim_params.flex.warm_start = 0.7
sim_params.flex.shape_collision_distance = 0.0001  # Distance to be maintained between soft bodies and other bodies or ground plane
sim_params.flex.shape_collision_margin = 0.0002  # Distance from rigid bodies at which to begin generating contact constraints
sim_params.flex.friction_mode = 2  # Friction about all 3 axes (including torsional)
sim_params.flex.dynamic_friction = 7.83414394e-01

# enable Von-Mises stress visualization
sim_params.stress_visualization = True
sim_params.stress_visualization_min = 0.0
sim_params.stress_visualization_max = 1.e+5

sim_params.use_gpu_pipeline = False

sim = gym.create_sim(args.compute_device_id, args.compute_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load urdf for sphere asset used to create softbody
asset_root = "../urdf"
soft_asset_file = "digit.urdf"

soft_thickness = 0.1    # important to add some thickness to the soft body to avoid interpenetrations

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.thickness = soft_thickness
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

print("Loading asset '%s' from '%s'" % (soft_asset_file, asset_root))
soft_asset = gym.load_asset(sim, asset_root, soft_asset_file, asset_options)

asset_soft_body_count = gym.get_asset_soft_body_count(soft_asset)
asset_soft_materials = gym.get_asset_soft_materials(soft_asset)

# Print asset soft material properties
print('Soft Material Properties:')
for i in range(asset_soft_body_count):
    mat = asset_soft_materials[i]
    print(f'(Body {i}) youngs: {mat.youngs} poissons: {mat.poissons} damping: {mat.damping}')

# set up the env grid
num_envs = 1
spacing = 3.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
soft_body_height = 0.1 #TODO For unknown reason this value cannotbe set to zero

# cache some common handles for later use
envs = []
soft_actors = []

print("Creating %d environments" % num_envs)
num_per_row = int(math.sqrt(num_envs))
for i in range(num_envs):

    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, soft_body_height, 0.0)
    pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * math.pi)

    # add soft body + rail actor
    soft_actor = gym.create_actor(env, soft_asset, pose, "soft", i, 1)
    soft_actors.append(soft_actor)

    # set soft material within a range of default
    actor_default_soft_materials = gym.get_actor_soft_materials(env, soft_actor)
    actor_soft_materials = gym.get_actor_soft_materials(env, soft_actor)

    # enable pd-control on rail joint to allow
    # control of the press using the GUI
    gym.set_joint_target_position(env, gym.get_joint_handle(env, "soft", "rail"), 0.0)

# Point camera at environments
cam_pos = gymapi.Vec3(-0.05, soft_body_height + 0.05, -0.05)
cam_target = gymapi.Vec3(0.3, soft_body_height - 0.05, 0.3)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# Initialize matplotlib and axes3D
fig = plt.figure() # 创建一个画布figure，然后在这个画布上加各种元素。
ax = Axes3D(fig)
plt.figure(figsize=(8,10))

# options
flag_visualize = True


def get_surface_indexes():
    '''
    This is a test function for env[0], will be removed in the future
    '''
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    indexes_ = []
    # Get particle positions
    gym.refresh_particle_state_tensor(sim)
    particle_states = gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))
    num_particles = len(particle_states)
    num_particles_per_env = int(num_particles / num_envs)
    nodal_coords = np.zeros((num_envs, num_particles_per_env, 3))
    for global_particle_index, particle_state in enumerate(particle_states):
        pos = particle_state[:3]
        env_index = global_particle_index // num_particles_per_env # which env
        local_particle_index = global_particle_index % num_particles_per_env # the index of particles in the current env
        nodal_coords[env_index][local_particle_index] = pos.numpy()

    # Get positions in env0 for testing
    z_pos_env_0 = list(nodal_coords[0][:][:,1])

    # Filter points by its height to get the surafce vertices indexes
    surface_height_threshold = soft_body_height + 0.0305 # manually tuned
    for i in range(num_particles_per_env):
        if(z_pos_env_0[i] > surface_height_threshold):
            indexes_.append(i)
    
    return indexes_

indexes = get_surface_indexes()

# Sim loop
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Get particle positions
    gym.refresh_particle_state_tensor(sim)
    particle_states = gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))
    num_particles = len(particle_states)
    num_particles_per_env = int(num_particles / num_envs)
    nodal_coords = np.zeros((num_envs, num_particles_per_env, 3))
    for global_particle_index, particle_state in enumerate(particle_states):
        pos = particle_state[:3]
        env_index = global_particle_index // num_particles_per_env # which env
        local_particle_index = global_particle_index % num_particles_per_env # the index of particles in the current env
        nodal_coords[env_index][local_particle_index] = pos.numpy()

    # Get positions in env0 for testing
    x_pos_env_0 = list(nodal_coords[0][:][:,0])
    y_pos_env_0 = list(nodal_coords[0][:][:,2])
    z_pos_env_0 = list(nodal_coords[0][:][:,1])

    # Filter surafce vertices by indexes
    surface_height_threshold = soft_body_height + 0.03125 # manually tuned
    x_pos_filtered_env_0 = []
    y_pos_filtered_env_0 = []
    z_pos_filtered_env_0 = []
    for i in indexes:
        x_pos_filtered_env_0.append(x_pos_env_0[i])
        y_pos_filtered_env_0.append(y_pos_env_0[i])
        z_pos_filtered_env_0.append(z_pos_env_0[i])

    # Visualize positions
    if(flag_visualize):
        plt.clf()
        plt.scatter(x_pos_filtered_env_0, y_pos_filtered_env_0, c='r')
        # set axes range
        plt.xlim(-0.01, 0.01) # sensor size
        plt.ylim(-0.012, 0.0135) # sensor size
        plt.pause(0.01)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
