from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import os
import math
from scipy.spatial.transform import Rotation as R
import numpy as np

class INCREMENTAL_CONTROLLER:
    def __init__(self, increment=1.0e-4, eps=1.0e-4) -> None:
        self.increment_ = increment
        self.eps_ = eps

    def get_input(self, state, ref_state):
        # initialize input value
        input_ = state

        # to indicate whether current state has reached its reference state
        reached_ = False

        if(abs(ref_state - state) < self.eps_):
            reached_ = True
        elif(ref_state > state):
            input_ += self.increment_
        elif(ref_state < state):
            input_ -= self.increment_
        else:
            print("Unknown error in function get_input()")

        return input_, reached_
        

class DIGIT_SIM:
    def __init__(self) -> None:
        # Arguments for Isaac Gym
        self.gym_args = gymutil.parse_arguments()
        self.gym_args.physics_engine = gymapi.SIM_FLEX # Soft-body simulation backend

        # Parameters for Simulation
        self.sim_params = gymapi.SimParams()
        self.sim_params.dt = 1.0 / 60.0
        self.sim_params.substeps = 3
        self.sim_params.flex.solver_type = 5
        self.sim_params.flex.num_outer_iterations = 4
        self.sim_params.flex.num_inner_iterations = 20
        self.sim_params.flex.relaxation = 0.8
        self.sim_params.flex.warm_start = 0.7
        self.sim_params.flex.shape_collision_distance = 0.0001
        self.sim_params.flex.shape_collision_margin = 0.0002
        self.sim_params.flex.friction_mode = 2
        self.sim_params.flex.dynamic_friction = 7.83414394e-01
        # enable Von-Mises stress visualization
        self.sim_params.stress_visualization = True
        self.sim_params.stress_visualization_min = 0.0
        self.sim_params.stress_visualization_max = 1.e+5
        # disable GPU pipeline
        self.sim_params.use_gpu_pipeline = False

        # Specify Sensors to be Loaded
        self.sensors = ["digit_sensor"]
        self.sensor_dir = os.path.join('urdf', 'sensor')
        self.sensor_assets = []
        self.sensor_offset = 0.1 # shouldn't be zero for unknown reasons
        self.sensor_actors = []

        # Specify Indenters to be Loaded
        self.indenters = ["ball_indenter"]
        self.indenter_dir = os.path.join('urdf', 'indenters')
        self.indenter_assets = []
        self.indenter_offset = 0.12
        self.indenter_actors = []

        # Specify Env Parameters
        self.env_num = len(self.indenters)
        self.env_spacing = 0.1
        self.env_lower = gymapi.Vec3(-self.env_spacing, 0.0, -self.env_spacing)
        self.env_upper = gymapi.Vec3(self.env_spacing, self.env_spacing, self.env_spacing)
        self.envs = []

        # Specify Indenter Controllers
        self.controller_pd_gains = [1.0e+9, 0.0] # PD controller parameters

        # Initialize Incremental Controller for Indenters
        self.incremental_controller = INCREMENTAL_CONTROLLER(increment=2e-5, eps=5e-4)

        # Set target indenter pose, Isaac Gym's coordinate system is Y up.
        self.indent_target = [0.0, self.sensor_offset + 0.012, -0.005,  # Position of tip
                              1.0, 0.0, 0.0,         # Orientation of x-axis       
                              0.0, 1.0, 0.0,         # Orientation of y-axis
                              0.0, 0.0, 1.0]         # Orientation of z-axis

    def initialize_simulation_(self):
        # initialize Isaac Gym
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(self.gym_args.compute_device_id, 
                                       self.gym_args.compute_device_id, 
                                       self.gym_args.physics_engine, 
                                       self.sim_params)
        # check simulation
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        # add ground plane
        plane_params = gymapi.PlaneParams()
        self.gym.add_ground(self.sim, plane_params)

    def create_viewer_(self):
        # create viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        # check viewer
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()

        # Point camera at environments
        viewport_scale_xz = 0.25
        viewport_scale_y = 0.1
        cam_pos = gymapi.Vec3(-self.env_spacing * viewport_scale_xz, 
                              (self.sensor_offset + self.indenter_offset) / 2.0 + self.env_spacing * viewport_scale_y, 
                              -self.env_spacing * viewport_scale_xz)
        cam_target = gymapi.Vec3(self.env_spacing * viewport_scale_xz, 
                                 (self.sensor_offset + self.indenter_offset) / 2.0 - self.env_spacing * viewport_scale_y, 
                                 self.env_spacing * viewport_scale_xz)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
    
    def load_assets_(self, assets_, dir_, fix_base_link_=True, disable_gravity_=True):
        # Initialize asset options
        options_ = gymapi.AssetOptions()
        options_.flip_visual_attachments = False
        options_.armature = 0.0
        options_.thickness = 0.0
        options_.linear_damping = 0.0
        options_.angular_damping = 0.0
        options_.default_dof_drive_mode = gymapi.DOF_MODE_POS
        options_.min_particle_mass = 1e-20

        # set optionals 
        options_.fix_base_link = fix_base_link_
        options_.disable_gravity = disable_gravity_

        # get asset handles
        asset_handles_ = []
        for asset in assets_:
            asset = self.gym.load_asset(self.sim, dir_, asset + '.urdf', options_)
            print("Loading asset from '%s'" % (dir_))
            asset_handles_.append(asset)
        
        return asset_handles_

    def create_envs_(self):
        # setup environmets
        print("Number of environments: %d" % self.env_num)
        env_per_row = int(math.sqrt(self.env_num))
        for i in range(self.env_num):
            # create env
            env = self.gym.create_env(self.sim, self.env_lower, self.env_upper, env_per_row)
            self.envs.append(env)

    def setup_assets_(self):
        self.sensor_assets = self.load_assets_(self.sensors, self.sensor_dir, fix_base_link_=True, disable_gravity_=True)
        self.indenter_assets = self.load_assets_(self.indenters, self.indenter_dir, fix_base_link_=False, disable_gravity_=True)

    def setup_actors_(self):
        for i in range(self.env_num):
            # create sensor actors
            sensor_pose = gymapi.Transform()
            sensor_pose.p = gymapi.Vec3(0.0, self.sensor_offset, 0.0)
            sensor_r = R.from_euler('XYZ', [0, 0, 0], degrees=True)
            sensor_quat = sensor_r.as_quat()
            sensor_pose.r = gymapi.Quat(*sensor_quat)
            sensor_actor = self.gym.create_actor(self.envs[i], self.sensor_assets[0], sensor_pose, "sensor", i, 1)
            self.sensor_actors.append(sensor_actor)

            # create indenter actors
            indenter_pose = gymapi.Transform()
            indenter_pose.p = gymapi.Vec3(0.0, self.indenter_offset, 0.0)
            indenter_r = R.from_euler('XYZ', [0, 0, 0], degrees=True)
            indenter_quat = indenter_r.as_quat()
            indenter_pose.r = gymapi.Quat(*indenter_quat)
            indenter_actor = self.gym.create_actor(self.envs[i], self.indenter_assets[i], indenter_pose, "indenter", i, 1)
            self.indenter_actors.append(indenter_actor)

            # setup indenter controllers
            indenter_dof_props = self.gym.get_actor_dof_properties(self.envs[i], self.indenter_actors[i])
            indenter_dof_props['driveMode'][0] = gymapi.DOF_MODE_POS
            indenter_dof_props['stiffness'][0] = self.controller_pd_gains[0]
            indenter_dof_props['damping'][0] = self.controller_pd_gains[1]
            self.gym.set_actor_dof_properties(self.envs[i], self.indenter_actors[i], indenter_dof_props)

    def indenters_control_(self):
        # get indent target
        target_position = self.indent_target[:3]
        target_x_axis = self.indent_target[3:6]
        target_y_axis = self.indent_target[6:9]
        target_z_axis = self.indent_target[9:12]

        # to indicate whether all indenters has reached their targets by logical judgement (AND)
        envs_flag_ = 1

        # walk through all envs
        for i in range(self.env_num):
            # get state
            indenter_state = self.gym.get_actor_rigid_body_states(self.envs[i], self.indenter_actors[i], gymapi.STATE_ALL)

            # to indicate whether all indenters has reached their targets by logical judgement (AND)
            env_flag_ = 1

            # walk through all links in the indenter
            for j in range(len(indenter_state)):
                # control position
                position_state = indenter_state[j]['pose']['p']

                # to indicate whether all indenters has reached their targets by logical judgement (AND)
                link_flag_ = 1
                # walk through xyz in a position
                for k in range(len(position_state)):
                    # moniter states
                    # print("link index: "+str(k)+", x: "+str(position_state[0])+", y: "+str(position_state[1])+", z: "+str(position_state[2]))
                    
                    # get input from a incremental controller
                    indenter_state[j]['pose']['p'][k], position_flag_ = self.incremental_controller.get_input(state=float(position_state[k]), 
                                                                                                              ref_state=target_position[k])
                    # logical operation (AND)
                    link_flag_ = link_flag_ * position_flag_

                # logical operation (AND)
                env_flag_ = env_flag_ * link_flag_

                # control orientation
                r = R.from_matrix(np.asarray([target_x_axis, target_y_axis, target_z_axis]).transpose())
                quat = r.as_quat()
                indenter_state[j]['pose']['r'] = tuple(quat)

                # set linear and angular velocities
                indenter_state[j]['vel']['linear'] = (0.0, 0.0, 0.0)
                indenter_state[j]['vel']['angular'] = (0.0, 0.0, 0.0)

            # logical operation (AND)
            envs_flag_ = envs_flag_ * env_flag_

            # apply changes
            self.gym.set_actor_rigid_body_states(self.envs[i], self.indenter_actors[i], indenter_state, gymapi.STATE_ALL)
        
        return envs_flag_

    def sim_loop_(self):
        # Sim loop
        while not self.gym.query_viewer_has_closed(self.viewer):
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # control indenters
            reached_ = self.indenters_control_()
            # exit loop
            if(reached_==True):
                break

            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)

            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)

    def coda_(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def main(self):
        self.initialize_simulation_()
        self.create_viewer_()
        self.create_envs_()
        self.setup_assets_()
        self.setup_actors_()
        self.sim_loop_()
        self.coda_()

if __name__ == "__main__":
    digit_sim = DIGIT_SIM()
    digit_sim.main()