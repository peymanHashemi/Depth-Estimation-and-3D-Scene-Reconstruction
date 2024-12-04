# Depth Estimation and 3D Scene Reconstruction
This project explores two key methods for estimating depth in computer vision: stereo vision and LiDAR sensors. The objective is to generate depth maps and reconstruct 3D scenes using data from the KITTI dataset, a well-known benchmark for autonomous driving research. The project emphasizes understanding and implementing fusion techniques that combine data from cameras and LiDAR for enhanced 3D perception.

# Content
- Table of Contents
  * [Project Tasks](#Project-Tasks) 
  * [Dataset and Guidelines](#Dataset-and-Guidelines)  
  * [METHOD](#METHOD)
    
## Project Tasks

### 1. LiDAR Data Visualization and Analysis:

Load and visualize point cloud data from a LiDAR sensor using the Open3D library.
Discuss the advantages and limitations of using point clouds for depth estimation, particularly in terms of density and accuracy.

### 2. Sensor Fusion and Point Cloud Mapping:

Implement a process to project the LiDAR point cloud onto a 2D image, mapping each point's depth to a corresponding pixel.
Visualize the result, ensuring that point colors reflect their depth, as shown in reference figures.

### 3. Depth Map Generation with Interpolation:

Create a depth map by interpolating missing depth values where LiDAR data is sparse.
Evaluate different interpolation methods and discuss which approach yields the most accurate and visually coherent results.

### 4. Stereo Vision for Depth Estimation:

Generate a dense depth map using stereo vision techniques with grayscale images from the dataset.
Compare this depth map with the one generated from the LiDAR point cloud, analyzing the strengths and weaknesses of each method.

### 5. 3D Scene Reconstruction:

Use the input image and its corresponding depth map to reconstruct a dense 3D point cloud.
Visualize the reconstructed scene using Open3D or other 3D visualization tools.
Discuss the benefits of stereo vision in creating more detailed 3D representations compared to LiDAR.

## Dataset and Guidelines
The project utilizes data from the KITTI dataset, which includes images and LiDAR scans collected from autonomous driving platforms. The dataset provides:

Stereo images from paired cameras.
LiDAR point clouds with calibration files for accurate mapping between sensors.
Calibration parameters, including camera matrices and transformation matrices between LiDAR and cameras, are essential for sensor fusion and accurate depth estimation.

## METHOD
