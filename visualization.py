import numpy as np
import open3d as o3d
import time

class POINT_CLOUD_VISUALIZATION:
    def __init__(self, num_points_) -> None:
        # predifined num of points
        self.num_points = num_points_

        # create visualizer and window.
        vis = o3d.visualization.Visualizer()
        vis.create_window(height=480, width=640)

        # initialize pointcloud instance.
        pcd = o3d.geometry.PointCloud()
        points = np.random.rand(10000, 3)
        pcd.points = o3d.utility.Vector3dVector(points)

        # include it in the visualizer before non-blocking visualization.
        vis.add_geometry(pcd)

        # to add new points each dt secs.
        dt = 0.01
        # number of points that will be added
        n_new = 100

        previous_t = time.time()

        # run non-blocking visualization. 
        # To exit, press 'q' or click the 'x' of the window.
        keep_running = True
        while keep_running:
            if time.time() - previous_t > dt:

                points = np.random.rand(10000, 3)
                pcd.points = o3d.utility.Vector3dVector(points)

                
                vis.update_geometry(pcd)
                previous_t = time.time()

            keep_running = vis.poll_events()
            vis.update_renderer()
        
        vis.destroy_window()

if __name__ == "__main__":
    vis = POINT_CLOUD_VISUALIZATION(10000)
    vis.main()