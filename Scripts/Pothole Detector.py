import numpy as np
import open3d as o3d
import os
print("Current working directory:", os.getcwd())
os.chdir('C:/Users\Ali\OneDrive - UBC\Desktop\Ph.D. Work\Potholes')
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
detector = PotholeDetector(curvature_threshold=0.003, neighborhood_radius=0.25)
pothole_cloud = detector.process_point_cloud('Outputs/ ')
o3d.io.write_point_cloud("Outputs/Hwy22_1_segpothole3.ply", pothole_cloud)

