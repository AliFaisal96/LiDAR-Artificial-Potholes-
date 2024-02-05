import numpy as np
import open3d as o3d
import laspy
from scipy.spatial import ConvexHull
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

print("Current working directory:", os.getcwd())
os.chdir('C:/Users\Ali\OneDrive - UBC\Desktop\Ph.D. Work\Potholes')


class PotholeCharacteristics:
    def __init__(self, eps, min_points, z_threshold=0.005):
        self.eps = eps
        self.min_points = min_points
        self.z_threshold = z_threshold

    def load_las_file(self, las_file_path):
        with laspy.open(las_file_path) as las_file:
            las = las_file.read()
            points = np.vstack((las.x, las.y, las.z)).transpose()
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)
            return point_cloud

    def segment_potholes(self, point_cloud):
        labels = np.array(point_cloud.cluster_dbscan(eps=self.eps, min_points=self.min_points))
        return labels

    def fit_plane_to_road_surface(self, points):
        max_z_value = np.max(points[:, 2])
        surface_points = points[np.abs(points[:, 2] - max_z_value) <= self.z_threshold]
        reg = LinearRegression().fit(surface_points[:, :2], surface_points[:, 2])
        return reg

    def calculate_pothole_area(self, pothole_id, pothole_points, plot_hull=False):
        plane_model = self.fit_plane_to_road_surface(pothole_points)
        predicted_z = plane_model.predict(pothole_points[:, :2])
        pothole_depth_points = pothole_points[predicted_z - pothole_points[:, 2] > self.z_threshold]
        points_2d = pothole_depth_points[:, :2]

        if len(points_2d) > 2:
            hull = ConvexHull(points_2d)
            area = hull.volume  # For 2D convex hulls, 'volume' gives the area

            if plot_hull:
                plt.figure()
                for simplex in hull.simplices:
                    plt.plot(points_2d[simplex, 0], points_2d[simplex, 1], 'k-')
                plt.plot(points_2d[hull.vertices, 0], points_2d[hull.vertices, 1], 'r--', lw=2)
                plt.fill(points_2d[hull.vertices, 0], points_2d[hull.vertices, 1], 'r', alpha=0.3)

                # Annotate the plot with the pothole area
                plt.text(np.mean(points_2d[:, 0]), np.mean(points_2d[:, 1]), f'Area: {area:.2f} sq units',
                         horizontalalignment='center', verticalalignment='center',
                         fontsize=9, color='blue', bbox=dict(facecolor='white', alpha=0.7))

                plt.xlabel('X')
                plt.ylabel('Y')
                plt.title(f'Pothole ID {pothole_id}: Convex Hull')
                plt.show()

            return area, len(pothole_depth_points)
        return 0, 0

    def calculate_pothole_dimensions(self, pothole_depth_points):
        """
        Calculate the dimensions (length, width, depth) of a pothole.
        Length and width are calculated from the 2D projection of pothole_depth_points.
        Depth is calculated as the difference between the maximum and minimum Z values.
        """
        # Project to 2D for length and width calculations
        points_2d = pothole_depth_points[:, :2]

        if len(points_2d) > 2:
            hull = ConvexHull(points_2d)
            length = np.max(hull.points[hull.vertices, 0]) - np.min(hull.points[hull.vertices, 0])
            width = np.max(hull.points[hull.vertices, 1]) - np.min(hull.points[hull.vertices, 1])

            # Calculate depth from the Z values of pothole_depth_points
            depth = np.max(pothole_depth_points[:, 2]) - np.min(pothole_depth_points[:, 2])

            return length, width, depth
        return 0, 0, 0

    def calculate_pothole_volume(self, pothole_points_3d, voxel_size=0.05):
        # Convert pothole 3D points to Open3D point cloud
        pothole_cloud = o3d.geometry.PointCloud()
        pothole_cloud.points = o3d.utility.Vector3dVector(pothole_points_3d)

        # Create a voxel grid from the pothole point cloud
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pothole_cloud, voxel_size)

        # Count the number of occupied voxels
        occupied_voxels = len(voxel_grid.get_voxels())

        # Calculate the volume
        volume = occupied_voxels * voxel_size**3

        return volume

    def process_las_file(self, las_file_path, plot_hulls=False):
        point_cloud = self.load_las_file(las_file_path)
        pothole_labels = self.segment_potholes(point_cloud)

        for i in range(pothole_labels.max() + 1):
            pothole_points = np.asarray(point_cloud.points)[pothole_labels == i]
            plane_model = self.fit_plane_to_road_surface(pothole_points)
            predicted_z = plane_model.predict(pothole_points[:, :2])
            pothole_depth_points = pothole_points[predicted_z - pothole_points[:, 2] > self.z_threshold]
            area, num_points = self.calculate_pothole_area(i, pothole_points, plot_hull=plot_hulls)
            length, width, depth = self.calculate_pothole_dimensions(pothole_depth_points)
            volume = self.calculate_pothole_volume(pothole_depth_points, voxel_size=0.05)
            print(f"Pothole {i}: Area = {area} square units, Number of Points = {num_points}, Length = {length:.2f} units, Width = {width:.2f} units, Depth = {depth:.2f} units, Volume = {volume:.4f} cubic units")


# Usage
characteristics = PotholeCharacteristics(eps=0.1, min_points=50)
characteristics.process_las_file('Outputs/Hwy22_1/GlobalMapper Post-Pothole Cleanup/Hwy22_1_postdetected.las',
                                 plot_hulls=True)

