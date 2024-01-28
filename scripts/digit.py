import math
from isaacgym import gymapi
from isaacgym import gymutil

import numpy as np

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
    pose.p = gymapi.Vec3(0.0, 1.5, 0.0)
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
cam_pos = gymapi.Vec3(-0.05, 1.55, -0.05)
cam_target = gymapi.Vec3(0.3, 1.45, 0.3)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# options
flag_draw_contacts = True
flag_compute_pressure = False

while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    if (flag_draw_contacts):
        gym.clear_lines(viewer)
        for env in envs:
            gym.draw_env_soft_contacts(viewer, env, gymapi.Vec3(0.7, 0.2, 0.15), 0.0005, True, True)

    if (flag_compute_pressure):
        # read tetrahedral and triangle data from simulation
        (tet_indices, tet_stress) = gym.get_sim_tetrahedra(sim)
        (tri_indices, tri_parents, tri_normals) = gym.get_sim_triangles(sim)

        for env in envs:

            # get range (start, count) for each soft actor (assumes one soft body)
            tet_range = gym.get_actor_tetrahedra_range(env, 0, 0)
            tri_range = gym.get_actor_triangle_range(env, 0, 0)

            for i in range(tri_range.start, tri_range.start + tri_range.count):

                # 'parent' tetrahedron of this triangle
                parent = tri_parents[i]
                parent_stress = tet_stress[parent]

                # convert to numpy
                stress = np.matrix([(parent_stress.x.x, parent_stress.y.x, parent_stress.z.x),
                                    (parent_stress.x.y, parent_stress.y.y, parent_stress.z.y),
                                    (parent_stress.x.z, parent_stress.y.z, parent_stress.z.z)])

                normal = np.array((tri_normals[i].x, tri_normals[i].y, tri_normals[i].z))

                # compute force density (since normal is normalized)
                force = np.dot(stress, normal)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
