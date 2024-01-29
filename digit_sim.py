from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import os
import math

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
        ## enable Von-Mises stress visualization
        self.sim_params.stress_visualization = True
        self.sim_params.stress_visualization_min = 0.0
        self.sim_params.stress_visualization_max = 1.e+5
        ## disable GPU pipeline
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
        cam_pos = gymapi.Vec3(-self.env_spacing, 
                              (self.sensor_offset + self.indenter_offset) / 2.0 + self.env_spacing, 
                              -self.env_spacing)
        cam_target = gymapi.Vec3(self.env_spacing, 
                                 (self.sensor_offset + self.indenter_offset) / 2.0 - self.env_spacing, 
                                 self.env_spacing)
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
            sensor_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * math.pi)
            sensor_actor = self.gym.create_actor(self.envs[i], self.sensor_assets[0], sensor_pose, "sensor", i, 1)
            self.sensor_actors.append(sensor_actor)

            # create indenter actors
            indenter_pose = gymapi.Transform()
            indenter_pose.p = gymapi.Vec3(0.0, self.indenter_offset, 0.0)
            indenter_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * math.pi)
            indenter_actor = self.gym.create_actor(self.envs[i], self.indenter_assets[i], indenter_pose, "indenter", i, 1)
            self.indenter_actors.append(indenter_actor)

            # setup indenter controllers
            indenter_dof_props = self.gym.get_actor_dof_properties(self.envs[i], self.indenter_actors[i])
            indenter_dof_props['driveMode'][0] = gymapi.DOF_MODE_POS
            indenter_dof_props['stiffness'][0] = self.controller_pd_gains[0]
            indenter_dof_props['damping'][0] = self.controller_pd_gains[1]
            self.gym.set_actor_dof_properties(self.envs[i], self.indenter_actors[i], indenter_dof_props)

    def sim_loop_(self):
        # Sim loop
        while not self.gym.query_viewer_has_closed(self.viewer):
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            # update the viewer

            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)

            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)

    def main(self):
        self.initialize_simulation_()
        self.create_viewer_()
        self.create_envs_()
        self.setup_assets_()
        self.setup_actors_()
        self.sim_loop_()


if __name__ == "__main__":
    digit_sim = DIGIT_SIM()
    digit_sim.main()