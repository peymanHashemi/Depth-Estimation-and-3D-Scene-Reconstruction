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

The content of your report includes a set of tasks related to depth estimation and 3D scene reconstruction using stereo vision and LiDAR data. Here's a step-by-step method for each task written in the first person, emphasizing implementation details:

---

### **Method**

### **1. LiDAR Data Visualization and Analysis**
To start, I loaded and visualized LiDAR point cloud data using the Open3D library. The steps were:
1. **Load the Point Cloud Data**: Using Open3D's `read_point_cloud()` function, I loaded the LiDAR `.pcd` file.
2. **Visualize the Data**: The `draw_geometries()` function was used to render the point cloud in a 3D space.
3. **Analysis**: I observed the density and distribution of points, noting that while point clouds offer high accuracy, their sparsity limits fine-detail depth representation.

#### Results:

From Above:

<div style="display: flex; flex-direction: row; justify-content: space-around;">

<div style="text-align: center;">
    <p><b>From Above</b></p>
    <img style="width: 400px;" src="https://github.com/user-attachments/assets/d6b89739-48f2-4041-999d-02444bb3f800">
</div>

<div style="text-align: center; margin-left: 20px;">
    <p><b>From Another Angle</b></p>
    <img style="width: 500px;" src="https://github.com/user-attachments/assets/c09deaf2-41c8-4733-a60a-f1e602bdd242">
</div>

</div>

### **2. Sensor Fusion and Point Cloud Mapping**
For sensor fusion, I mapped the LiDAR point cloud onto a 2D image plane:
1. **Projection Preparation**: I utilized the calibration parameters (camera matrices and transformations) provided in the KITTI dataset to align the LiDAR data with the image.
2. **Mapping Points**: Using the intrinsic camera matrix, I converted the 3D coordinates of the point cloud into 2D image pixels.
3. **Depth Visualization**: For each mapped point, I assigned a color based on its depth, creating a pseudo-color depth image.

#### Results:

<img style="width:400px" src="https://github.com/user-attachments/assets/ebb1359b-af55-4021-bb5c-57415242ed0d"> 


### **3. Depth Map Generation with Interpolation**
To handle the sparse depth data, I applied interpolation techniques:
1. **Data Selection**: Extracted depth values from the LiDAR point cloud mapped onto the image.
2. **Interpolation**: Used the `scipy.interpolate.griddata()` function to fill in the gaps in the depth map with linear interpolation.
3. **Evaluation**: Compared different interpolation methods (nearest, linear, cubic) to assess accuracy and visual coherence. Linear interpolation yielded the best balance between performance and realism.

#### Results:

<img style="width:400px" src="https://github.com/user-attachments/assets/ebb1359b-af55-4021-bb5c-57415242ed0d"> 


### **4. Stereo Vision for Depth Estimation**
Using stereo images from paired cameras:
1. **Preprocessing**: Converted stereo images to grayscale.
2. **Disparity Calculation**: Used OpenCV's `StereoBM` function to compute the disparity map between the left and right images.
3. **Depth Calculation**: Transformed the disparity map into a depth map using the known baseline distance and camera focal length.
4. **Comparison**: Evaluated the stereo-based depth map against the LiDAR-generated map, finding that stereo vision provided more dense but slightly less accurate depth maps.

#### Results:

<img style="width:400px" src="https://github.com/user-attachments/assets/ebb1359b-af55-4021-bb5c-57415242ed0d"> 

### **5. 3D Scene Reconstruction**
For reconstructing a dense 3D point cloud:
1. **Generate 3D Points**: Combined the depth map from stereo vision with the original image to back-project each pixel into 3D space using camera intrinsics.
2. **Color Mapping**: Assigned colors from the image to each 3D point for realistic visualization.
3. **Visualization**: Rendered the reconstructed scene using Open3D. The stereo-based depth map helped produce a more detailed 3D representation than LiDAR alone.

#### Results:

<img style="width:400px" src="https://github.com/user-attachments/assets/ebb1359b-af55-4021-bb5c-57415242ed0d"> 


