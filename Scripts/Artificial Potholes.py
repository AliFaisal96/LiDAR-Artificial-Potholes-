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


# Usage
processor = PointCloudProcessor()
cloud = processor.process_file('pothole_dataset/Hwy22_1_cc.ply')

# Parameters for artificial potholes
num_potholes = 1000  # Number of potholes to create
max_radius = 0.5  # Maximum radius of potholes
max_depth = 0.2  # Maximum depth of potholes (realistic value)


# Create artificial potholes with a quadratic depth model
cloud_with_potholes = processor.create_artificial_potholes_quadratic(cloud, num_potholes, max_radius, max_depth)

# Visualize the point cloud to verify the potholes
o3d.visualization.draw_geometries([cloud_with_potholes])


# Now, pothole_details contains information about each artificial pothole
# for pothole in pothole_details:
#     print(f"Pothole at X: {pothole['center_x']}, Y: {pothole['center_y']}, Radius: {pothole['radius']}, Depth: {pothole['depth']}")
o3d.io.write_point_cloud("Outputs/Hwy22_1_cc_potholes.ply", cloud_with_potholes)

#Pothole Detection
class PotholeDetector:
    def __init__(self, curvature_threshold, neighborhood_radius):
        self.curvature_threshold = curvature_threshold
        self.neighborhood_radius = neighborhood_radius

    def estimate_curvature(self, point_cloud):
        # Step 1: Estimate the surface curvature for each point in the cloud
        pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)
        curvatures = []

        for i in range(len(np.asarray(point_cloud.points))):
            if i % 10000 == 0:  # Print a message every 10,000 points
                print(f"Processing point {i}/{len(point_cloud.points)}")
            # Find neighbors of a point
            [_, idx, _] = pcd_tree.search_radius_vector_3d(point_cloud.points[i], self.neighborhood_radius)
            # Estimate the local surface using neighbors
            if len(idx) < 3:  # Not enough points to fit a plane
                curvatures.append(0)
                continue
            neighbors = np.asarray(point_cloud.points)[idx, :]
            covariance_matrix = np.cov(neighbors.T)
            eigenvalues, _ = np.linalg.eig(covariance_matrix)
            # Curvature is the smallest eigenvalue divided by the sum of all eigenvalues
            curvature = np.min(eigenvalues) / np.sum(eigenvalues)
            curvatures.append(curvature)

        curvatures = np.array(curvatures)
        return point_cloud, curvatures

    def segment_potholes(self, point_cloud, curvatures):
        # Convert Open3D PointCloud to NumPy array for indexing
        points_np = np.asarray(point_cloud.points)
        # Segment the point cloud based on the curvature threshold
        pothole_points = points_np[curvatures > self.curvature_threshold]
        # Convert the segmented points back to an Open3D PointCloud
        pothole_cloud = o3d.geometry.PointCloud()
        pothole_cloud.points = o3d.utility.Vector3dVector(pothole_points)
        return pothole_cloud

    def visualize_potholes(self, road_cloud, pothole_cloud):
        # Step 5: Visualize the detection
        road_cloud.paint_uniform_color([0.6, 0.6, 0.6])  # Grey color for road
        pothole_cloud.paint_uniform_color([1, 0, 0])  # Red color for potholes
        o3d.visualization.draw_geometries([road_cloud, pothole_cloud])

    def calculate_pothole_depth(self, pothole_cloud):
        if pothole_cloud.is_empty():
            print("No potholes detected.")
            return

        # Use DBSCAN to cluster the pothole cloud
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(pothole_cloud.cluster_dbscan(eps=0.05, min_points=10))

        max_label = labels.max()
        print(f"Detected {max_label + 1} potholes.")

        for i in range(max_label + 1):
            cluster = pothole_cloud.select_by_index(np.where(labels == i)[0])
            z_values = np.asarray(cluster.points)[:, 2]
            depth = np.max(z_values) - np.min(z_values)
            print(f"Estimated depth of pothole {i}: {depth}")

    def process_point_cloud(self, file_path):
        # Load the point cloud
        point_cloud = o3d.io.read_point_cloud(file_path)
        # Estimate curvatures
        point_cloud, curvatures = self.estimate_curvature(point_cloud)
        # Segment potholes based on curvature
        pothole_cloud = self.segment_potholes(point_cloud, curvatures)
        # Visualize the potholes
        self.visualize_potholes(point_cloud, pothole_cloud)
        # Calculate and print out the depth of each pothole
        self.calculate_pothole_depth(pothole_cloud)
        return pothole_cloud

# Usage
detector = PotholeDetector(curvature_threshold=0.001, neighborhood_radius=0.5)
pothole_cloud = detector.process_point_cloud('Outputs/Hwy22_1_cc_halfpotholes5.ply')
o3d.io.write_point_cloud("Outputs/Hwy22_1_segpothole.ply", pothole_cloud)
