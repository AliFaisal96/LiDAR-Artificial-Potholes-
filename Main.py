import open3d as o3d
import laspy
import os
import numpy as np
import random
import pandas as pd


# Print the current working directory
print("Current Working Directory:", os.getcwd())

class PointCloudProcessor:
    def __init__(self):
        pass

    def convert_las_to_ply(self, las_file_path, ply_file_path):
        print(f"Converting {las_file_path} to {ply_file_path}...")
        with laspy.open(las_file_path) as las_file:
            las = las_file.read()
            points = np.vstack((las.x, las.y, las.z)).transpose()
            ply_cloud = o3d.geometry.PointCloud()
            ply_cloud.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(ply_file_path, ply_cloud)
        print("Conversion complete.")

    def load_point_cloud(self, file_path):
        print(f"Loading point cloud from {file_path}...")
        return o3d.io.read_point_cloud(file_path, format='auto')

    def process_file(self, file_path):
        file_extension = os.path.splitext(file_path)[1]
        if file_extension == '.las':
            ply_file_path = file_path.replace('.las', '.ply')
            self.convert_las_to_ply(file_path, ply_file_path)
            return self.load_point_cloud(ply_file_path)
        elif file_extension == '.ply':
            return self.load_point_cloud(file_path)
        else:
            print("Unsupported file format.")
            return None

    def create_artificial_potholes_quadratic(self, point_cloud, num_potholes, max_radius, max_depth):
        points = np.asarray(point_cloud.points)

        # Initialize the colors array if the point cloud has colors
        if point_cloud.has_colors():
            colors = np.asarray(point_cloud.colors)
        else:
            colors = np.ones((len(points), 3)) * 0.5  # Grey color for all points

        # Generate pothole specifications and modify points accordingly
        for _ in range(num_potholes):
            center_x = random.uniform(points[:, 0].min(), points[:, 0].max())
            center_y = random.uniform(points[:, 1].min(), points[:, 1].max())
            radius = random.uniform(0.01, max_radius)
            depth = random.uniform(0.01, max_depth)
            pothole_color = np.random.rand(3)  # Random color for the pothole

            # Calculate distances from the pothole center for all points
            distances = np.sqrt((points[:, 0] - center_x) ** 2 + (points[:, 1] - center_y) ** 2)

            # Identify points within the pothole radius
            pothole_indices = distances < radius

            # Displace the Z values of points within the pothole
            depression_depths = depth * (1 - (distances[pothole_indices] / radius) ** 2)
            points[pothole_indices, 2] -= depression_depths
            colors[pothole_indices] = pothole_color  # Color the pothole points

        # Create and return the modified point cloud
        modified_point_cloud = o3d.geometry.PointCloud()
        modified_point_cloud.points = o3d.utility.Vector3dVector(points)
        modified_point_cloud.colors = o3d.utility.Vector3dVector(colors)  # Set the colors
        return modified_point_cloud

    # def gaussian_pothole_depth(self, distance, radius, max_depth):
    #     # Gaussian function to model the pothole depth
    #     sigma = radius   # Standard deviation for a gentle slope
    #     return max_depth * np.exp(- (distance ** 2) / (2 * sigma ** 2))
    #
    # def create_artificial_potholes_gaussian(self, point_cloud, num_potholes, max_radius, max_depth, surface_threshold, offset_correction):
    #     points = np.asarray(point_cloud.points)
    #     colors = np.asarray(point_cloud.colors) if point_cloud.has_colors() else np.ones((len(points), 3)) * 0.5
    #
    #     for _ in range(num_potholes):
    #         # Randomly choose a center for the pothole
    #         center_x = random.uniform(points[:, 0].min(), points[:, 0].max())
    #         center_y = random.uniform(points[:, 1].min(), points[:, 1].max())
    #
    #         # Estimate local road surface level by finding points around the center within a certain threshold
    #         local_surface_mask = (
    #             (np.abs(points[:, 0] - center_x) < surface_threshold) &
    #             (np.abs(points[:, 1] - center_y) < surface_threshold)
    #         )
    #         if np.any(local_surface_mask):
    #             # Calculate the median Z value of these points to estimate the local road surface
    #             local_road_surface_z = np.median(points[local_surface_mask, 2])
    #         else:
    #             # If no points are found within the threshold, skip pothole creation
    #             continue
    #
    #         # Apply offset correction to the local road surface estimation
    #         local_road_surface_z -= offset_correction
    #
    #         radius = random.uniform(0.5, max_radius)
    #         depth = random.uniform(0.1, max_depth)
    #         pothole_color = np.random.rand(3)
    #
    #         distances = np.sqrt((points[:, 0] - center_x) ** 2 + (points[:, 1] - center_y) ** 2)
    #         pothole_mask = (distances < radius) & (np.abs(points[:, 2] - local_road_surface_z) < surface_threshold)
    #
    #         # Apply the Gaussian depression to the points within the pothole area
    #         for i in np.where(pothole_mask)[0]:
    #             depth_change = self.gaussian_pothole_depth(distances[i], radius, depth)
    #             points[i, 2] = local_road_surface_z - depth_change  # Make sure the new Z is below the surface
    #             colors[i] = pothole_color  # Color the pothole points
    #
    #     modified_point_cloud = o3d.geometry.PointCloud()
    #     modified_point_cloud.points = o3d.utility.Vector3dVector(points)
    #     modified_point_cloud.colors = o3d.utility.Vector3dVector(colors)
    #     return modified_point_cloud

# Usage
processor = PointCloudProcessor()
cloud = processor.process_file('pothole_dataset/Hwy22_1_cc.ply')

# Parameters for artificial potholes
num_potholes = 1000  # Number of potholes to create
max_radius = 0.5  # Maximum radius of potholes
max_depth = 0.2  # Maximum depth of potholes (realistic value)


# Create artificial potholes with a quadratic depth model
cloud_with_potholes = processor.create_artificial_potholes_quadratic(cloud, num_potholes, max_radius, max_depth)
# cloud_with_potholes = processor.create_artificial_potholes_gaussian(cloud, num_potholes, max_radius, max_depth, surface_threshold, offset_correction)

# Visualize the point cloud to verify the potholes
o3d.visualization.draw_geometries([cloud_with_potholes])

# potholearray = np.asarray(cloud.points)
# import matplotlib.pyplot as plt
#
# # Extract z values
# z_values = potholearray[:, 2]
#
# # Create histogram
# plt.hist(z_values, bins=50)
# plt.title('Histogram of z values')
# plt.xlabel('z')
# plt.ylabel('Frequency')
# plt.show()
# Now, pothole_details contains information about each artificial pothole
# for pothole in pothole_details:
#     print(f"Pothole at X: {pothole['center_x']}, Y: {pothole['center_y']}, Radius: {pothole['radius']}, Depth: {pothole['depth']}")
o3d.io.write_point_cloud("Outputs/Hwy22_1_cc_potholes.ply", cloud_with_potholes)

