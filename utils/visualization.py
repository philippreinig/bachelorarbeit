import uuid
import os

import numpy as np

os.environ["OPEN3D_RENDERING_BACKEND"] = "CPU"

import open3d as o3d

from matplotlib import pyplot as plt


def visualize_pcl_matplotlib(pcl):
    pcl_np = pcl.cpu().numpy()  # Convert to NumPy if it's a torch.Tensor
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pcl_np[:, 0], pcl_np[:, 1], pcl_np[:, 2], s=1, c=pcl_np[:, 2], cmap="viridis")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig(f"imgs/point_clouds/{uuid.uuid4()}.png")
    plt.close()

def visualize_pcl_open3d(pcl):
    
    pcl_np = pcl.cpu().numpy()  # Convert torch.Tensor to NumPy

     # Convert tensor to Open3D Point Cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl_np)

    # Set up an offscreen renderer
    width, height = 800, 600  # Adjust resolution as needed
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

    # Create a scene and add the point cloud
    scene = renderer.scene
    scene.add_geometry("pcl", pcd, o3d.visualization.rendering.MaterialRecord())

    # Set lighting and background
    scene.set_background([1, 1, 1, 1])  # White background
    scene.scene.enable_lighting(True)

    # Render to image
    img = renderer.render_to_image()
    o3d.io.write_image(f"imgs/o3d/{uuid.uuid4()}", img)


