import open3d as o3d
import laspy
import os
import numpy as np
import random
import pandas as pd


# Print the current working directory
print("Current Working Directory:", os.getcwd())
os.chdir('C:/Users\Ali\OneDrive - UBC\Desktop\Ph.D. Work\Potholes')

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

    def get_centerline_from_txt(self, txt_file_path):
        centerline = []
        with open(txt_file_path, 'r') as file:
            for line in file:
                x, y = map(float, line.strip().split(','))
                centerline.append((x, y))
        return centerline

    def calculate_tangent(self, point, next_point):
        dx = next_point[0] - point[0]
        dy = next_point[1] - point[1]
        tangent = np.array([dx, dy])
        norm = np.linalg.norm(tangent)
        return tangent / norm if norm != 0 else tangent

    def is_overlapping(self, new_pothole, existing_potholes, min_distance):
        for pothole in existing_potholes:
            distance = np.sqrt((new_pothole[0] - pothole[0]) ** 2 + (new_pothole[1] - pothole[1]) ** 2)
            if distance < min_distance:
                return True
        return False

    def is_too_close_on_y_axis(self, new_pothole, existing_potholes, y_threshold):
        for pothole in existing_potholes:
            if abs(new_pothole[1] - pothole[1]) < y_threshold:
                return True
        return False

    def create_artificial_potholes_bowl_shaped(self, point_cloud, road_centerline, num_potholes, max_radius, max_depth, road_width):
        points = np.asarray(point_cloud.points)
        if point_cloud.has_colors():
            colors = np.asarray(point_cloud.colors)
        else:
            colors = np.ones((len(points), 3)) * 0.5

        pothole_details = []
        existing_potholes = []
        y_threshold = road_width / 3

        while len(pothole_details) < num_potholes:
            idx = random.randint(0, len(road_centerline) - 2)
            point, next_point = road_centerline[idx], road_centerline[idx + 1]

            tangent = self.calculate_tangent(point, next_point)
            perp_vector = np.array([-tangent[1], tangent[0]])

            offset = random.uniform(-road_width / 2, road_width / 2)
            center_x, center_y = point[0] + offset * perp_vector[0], point[1] + offset * perp_vector[1]

            radius = random.uniform(0.05, max_radius)
            depth = random.uniform(0.02, max_depth)
            pothole_color = np.random.rand(3)

            # Check if new pothole overlaps with existing ones and is not too close on the y-axis
            overlapping = self.is_overlapping((center_x, center_y), existing_potholes, 2 * radius)
            too_close_on_y = self.is_too_close_on_y_axis((center_x, center_y), existing_potholes, y_threshold)
            if overlapping or too_close_on_y:
                continue  # Skip this iteration and try a new pothole

            distances = np.sqrt((points[:, 0] - center_x) ** 2 + (points[:, 1] - center_y) ** 2)
            pothole_indices = distances < radius

            if np.any(pothole_indices) and np.sum(pothole_indices) >= 100:
                num_pothole_points = np.sum(pothole_indices)
                displacement = np.sqrt(radius ** 2 - distances[pothole_indices] ** 2)
                normalized_displacement = displacement / np.max(displacement)
                depression_depths = depth * normalized_displacement
                points[pothole_indices, 2] -= depression_depths
                colors[pothole_indices] = pothole_color

                print(
                    f"Pothole {len(pothole_details)}: Center=({center_x:.2f}, {center_y:.2f}), Radius={radius:.2f}, Depth={depth:.2f}, Points={num_pothole_points}")
                pothole_details.append({'center_x': center_x, 'center_y': center_y, 'radius': radius, 'depth': depth, 'num_points': num_pothole_points})
                existing_potholes.append((center_x, center_y, radius))
            else:
                print(f"Pothole {len(pothole_details)} skipped: No points within radius")

        modified_point_cloud = o3d.geometry.PointCloud()
        modified_point_cloud.points = o3d.utility.Vector3dVector(points)
        modified_point_cloud.colors = o3d.utility.Vector3dVector(colors)
        return modified_point_cloud
# Usage
processor = PointCloudProcessor()
cloud = processor.process_file('pothole_dataset/Hwy22_1_cc.ply')
road_centerline = processor.get_centerline_from_txt('pothole_dataset/Hwy22_1_cc_centerline.txt.txt')
num_potholes = 20
max_radius = 0.3
max_depth = 0.1
road_width = 7

cloud_with_potholes = processor.create_artificial_potholes_bowl_shaped(cloud, road_centerline, num_potholes, max_radius, max_depth, road_width)
o3d.io.write_point_cloud("Outputs/Hwy22_1_cc_potholes4.ply", cloud_with_potholes)
