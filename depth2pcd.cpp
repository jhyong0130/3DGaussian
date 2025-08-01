#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

struct Point3DRGB {
    float x, y, z;
    float r, g, b;
};

struct CameraIntrinsics {
    float fx, fy, cx, cy;
};

// Function to convert depth image to point cloud with RGB values
vector<Point3DRGB> depth2pcd(const Mat& depth_image, 
                              const Mat& rgb_image,
                              const CameraIntrinsics& depth_intrinsics,
                              const CameraIntrinsics& rgb_intrinsics,
                              const Mat& R = Mat::eye(3, 3, CV_32F),
                              const Mat& T = Mat::zeros(3, 1, CV_32F)) {
    
    vector<Point3DRGB> point_cloud;
    
    // Get image dimensions
    int h_d = depth_image.rows;
    int w_d = depth_image.cols;
    int h_rgb = rgb_image.rows;
    int w_rgb = rgb_image.cols;
    
    // Camera intrinsics
    float FX_DEPTH = depth_intrinsics.fx;
    float FY_DEPTH = depth_intrinsics.fy;
    float CX_DEPTH = depth_intrinsics.cx;
    float CY_DEPTH = depth_intrinsics.cy;
    
    float FX_RGB = rgb_intrinsics.fx;
    float FY_RGB = rgb_intrinsics.fy;
    float CX_RGB = rgb_intrinsics.cx;
    float CY_RGB = rgb_intrinsics.cy;
    
    // Process each pixel in the depth image
    for (int i = 0; i < h_d; i++) {
        for (int j = 0; j < w_d; j++) {
            // Get depth value (assuming depth is in mm, convert to meters)
            float depth = depth_image.at<uint16_t>(i, j) / 1000.0f;
            
            // Skip invalid depth values
            if (depth <= 0 || depth > 10.0f) continue;
            
            // Convert pixel coordinates to 3D point in depth camera coordinates
            float x = (j - CX_DEPTH) * depth / FX_DEPTH;
            float y = (i - CY_DEPTH) * depth / FY_DEPTH;
            float z = depth;
            
            // Transform to RGB camera coordinates
            Mat point_depth = (Mat_<float>(3, 1) << x, y, z);
            Mat point_rgb = R * point_depth + T;
            
            float x_rgb = point_rgb.at<float>(0, 0);
            float y_rgb = point_rgb.at<float>(1, 0);
            float z_rgb = point_rgb.at<float>(2, 0);
            
            // Project to RGB image coordinates
            if (z_rgb <= 0) continue; // Skip points behind the camera
            
            float u = (x_rgb * FX_RGB) / z_rgb + CX_RGB;
            float v = (y_rgb * FY_RGB) / z_rgb + CY_RGB;
            
            // Round and clip to RGB image bounds
            int u_int = max(0, min(w_rgb - 1, (int)round(u)));
            int v_int = max(0, min(h_rgb - 1, (int)round(v)));
            
            // Get RGB color values (assuming BGR format, normalize to [0,1])
            Vec3b color = rgb_image.at<Vec3b>(v_int, u_int);
            
            Point3DRGB point;
            point.x = x_rgb;
            point.y = y_rgb;
            point.z = z_rgb;
            point.r = color[2] / 255.0f; // Red (BGR -> RGB)
            point.g = color[1] / 255.0f; // Green
            point.b = color[0] / 255.0f; // Blue
            
            point_cloud.push_back(point);
        }
    }
    
    return point_cloud;
}

// Example usage function
int main() {
    // Load depth and RGB images
    Mat depth_image = imread("depth_0.png", IMREAD_ANYDEPTH);
    Mat rgb_image = imread("DepthCapture_2025-07-24-14-00-07.png", IMREAD_COLOR);
    
    if (depth_image.empty() || rgb_image.empty()) {
        cerr << "Error: Could not load images!" << endl;
        return -1;
    }
    
    // Set up camera intrinsics (adjust these values based on your camera)
    CameraIntrinsics depth_intrinsics = {610.737f, 610.621f, 639.815f, 363.492f};
    CameraIntrinsics rgb_intrinsics = {610.737f, 610.621f, 639.815f, 363.492f};
    
    // Identity transformation (assuming depth and RGB cameras are aligned)
    Mat R = Mat::eye(3, 3, CV_32F);
    Mat T = Mat::zeros(3, 1, CV_32F);
    
    // Convert depth image to point cloud
    vector<Point3DRGB> point_cloud = depth2pcd(depth_image, rgb_image, 
                                               depth_intrinsics, rgb_intrinsics, R, T);
    
    cout << "Generated point cloud with " << point_cloud.size() << " points" << endl;
    
    // Example: Save point cloud to PLY format
    ofstream ply_file("output_pointcloud.ply");
    ply_file << "ply\n";
    ply_file << "format ascii 1.0\n";
    ply_file << "element vertex " << point_cloud.size() << "\n";
    ply_file << "property float x\n";
    ply_file << "property float y\n";
    ply_file << "property float z\n";
    ply_file << "property float red\n";
    ply_file << "property float green\n";
    ply_file << "property float blue\n";
    ply_file << "end_header\n";
    
    for (const auto& point : point_cloud) {
        ply_file << point.x << " " << point.y << " " << point.z << " "
                 << point.r << " " << point.g << " " << point.b << "\n";
    }
    ply_file.close();
    
    cout << "Point cloud saved to output_pointcloud.ply" << endl;
    
    return 0;
}
