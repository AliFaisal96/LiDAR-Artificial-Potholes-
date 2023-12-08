import open3d as o3d
import laspy
import os
import numpy as np
import random


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
        colors = np.ones((len(points), 3)) * 0.5  # Initialize colors as grey

        # Generate random pothole centers, radii, and depths
        for _ in range(num_potholes):
            center_x = random.uniform(points[:, 0].min(), points[:, 0].max())
            center_y = random.uniform(points[:, 1].min(), points[:, 1].max())
            radius = random.uniform(0.01, max_radius)
            depth = random.uniform(0.01, max_depth)
            pothole_color = np.random.rand(3)  # Random color for the pothole

            for i, point in enumerate(points):
                distance = np.sqrt((point[0] - center_x) ** 2 + (point[1] - center_y) ** 2)
                if distance < radius:
                    # Calculate the depth based on the distance from the center using a quadratic model
                    depression_depth = depth * (1 - (distance / radius) ** 2)
                    points[i, 2] -= depression_depth
                    colors[i] = pothole_color  # Apply the unique color to pothole points

        # Create and return the modified point cloud
        modified_point_cloud = o3d.geometry.PointCloud()
        modified_point_cloud.points = o3d.utility.Vector3dVector(points)
        modified_point_cloud.colors = o3d.utility.Vector3dVector(colors)  # Set the colors
        return modified_point_cloud

# Usage
processor = PointCloudProcessor()
cloud = processor.process_file('pothole_dataset/Hwy22_1_cc_half.ply')

# Parameters for artificial potholes
num_potholes = 20  # Number of potholes to create
max_radius = 1  # Maximum radius of potholes
max_depth = 0.5  # Maximum depth of potholes (realistic value)

# Create artificial potholes with a quadratic depth model
cloud_with_potholes = processor.create_artificial_potholes_quadratic(cloud, num_potholes, max_radius, max_depth)

# Visualize the point cloud to verify the potholes
o3d.visualization.draw_geometries([cloud_with_potholes])

# potholearray = np.asarray(cloud_with_potholes.points)
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
o3d.io.write_point_cloud("Outputs/Hwy22_1_cc_halfpotholes1.ply", cloud_with_potholes)

