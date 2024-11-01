#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <fstream>
#include <thread>
#include <opencv2/opencv.hpp>

std::vector<std::vector<double>> loadGroundTruthPoses(const std::string& filename) {
    std::vector<std::vector<double>> poses;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error opening ground-truth poses file: " << filename << std::endl;
        return poses;
    }
    double pose[12]; // Since each row is a 3x4 matrix flattened
    while (infile >> pose[0] >> pose[1] >> pose[2] >> pose[3] >> pose[4] >> pose[5] >> pose[6] >> pose[7] >> pose[8] >> pose[9] >> pose[10] >> pose[11]) {
        poses.push_back({pose[3], pose[7], pose[11]});
    }
    infile.close();
    return poses;
}

cv::Mat nthGroundTruthPose(const std::string& filename, int n) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error opening ground-truth poses file: " << filename << std::endl;
        return cv::Mat();
    }
    double pose[12];  // Each line is a 3x4 matrix flattened
    std::string line;
    int currentLine = 0;
    // Skip lines until reaching the nth line
    while (std::getline(infile, line)) {
        if (currentLine == n) {
            std::istringstream iss(line);
            for (int i = 0; i < 12; ++i) {
                iss >> pose[i];
            }
            infile.close();

            // Create a 3x4 matrix and populate it with pose data
            cv::Mat poseMat(3, 4, CV_64F);
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 4; ++col) {
                    poseMat.at<double>(row, col) = pose[row * 4 + col];
                }
            }
            return poseMat;
        }
        ++currentLine;
    }
    // return
}

void plotPosesPCL(const std::vector<std::vector<double>>& ground_truth_poses, const std::vector<Eigen::Affine3f>& estimated_poses) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr gt_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr est_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Pose Trajectories: Ground Truth vs Estimated"));
    viewer->addCoordinateSystem (1.0);
    viewer->setBackgroundColor(0, 0, 0);
    for (size_t i = 0; i < estimated_poses.size(); ++i) {
        if (i < ground_truth_poses.size()) {
            // gt_cloud->points.emplace_back(pcl::PointXYZ(ground_truth_poses[i][0], ground_truth_poses[i][1], ground_truth_poses[i][2]));
            gt_cloud->points.emplace_back(pcl::PointXYZ(ground_truth_poses[i][0], ground_truth_poses[i][2], 0));
        }
        est_cloud->points.emplace_back(pcl::PointXYZ(estimated_poses[i].translation().x(), estimated_poses[i].translation().z(), 0));
        // viewer->addCoordinateSystem(0.5, estimated_poses[i], "2nd cam");
    }

    // Ground truth in blue
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> gt_color(gt_cloud, 0, 0, 255);
    viewer->addPointCloud<pcl::PointXYZ>(gt_cloud, gt_color, "ground truth");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "ground truth");

    // Estimated poses in red
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> est_color(est_cloud, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(est_cloud, est_color, "estimated");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "estimated");

    // viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void savePosesToFile(const std::vector<std::vector<double>>& poses, const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    for (const auto& pose : poses) {
        outfile << pose[0] << " " << pose[1] << " " << pose[2] << std::endl;
    }
    outfile.close();
}

Eigen::Affine3f mat2Eigen(cv::Mat T) {
    Eigen::Matrix4f eig_mat;
    Eigen::Affine3f cam_pose;
    T.convertTo(T, CV_32F);
    eig_mat(0,0) = T.at<float>(0,0);eig_mat(0,1) = T.at<float>(0,1);eig_mat(0,2) = T.at<float>(0,2);
    eig_mat(1,0) = T.at<float>(1,0);eig_mat(1,1) = T.at<float>(1,1);eig_mat(1,2) = T.at<float>(1,2);
    eig_mat(2,0) = T.at<float>(2,0);eig_mat(2,1) = T.at<float>(2,1);eig_mat(2,2) = T.at<float>(2,2);
    eig_mat(3,0) = T.at<float>(3, 0); eig_mat(3,1) = T.at<float>(3, 1); eig_mat(3,2) = T.at<float>(3, 2);
    eig_mat(0,3) = T.at<float>(0, 3);
    eig_mat(1,3) = T.at<float>(1, 3);
    eig_mat(2,3) = T.at<float>(2, 3);
    eig_mat(3,3) = T.at<float>(3, 3);

    cam_pose = eig_mat;
    return cam_pose;
}
Eigen::Affine3f mat2Eigen(cv::Mat R, cv::Mat t) {
    Eigen::Matrix4f eig_mat;
    Eigen::Affine3f cam_pose;

    R.convertTo(R, CV_32F);
    t.convertTo(t, CV_32F);

    //this shows how a camera moves
    cv::Mat Rinv = R.t(); 
    cv::Mat T = -Rinv * t;

    eig_mat(0,0) = Rinv.at<float>(0,0);eig_mat(0,1) = Rinv.at<float>(0,1);eig_mat(0,2) = Rinv.at<float>(0,2);
    eig_mat(1,0) = Rinv.at<float>(1,0);eig_mat(1,1) = Rinv.at<float>(1,1);eig_mat(1,2) = Rinv.at<float>(1,2);
    eig_mat(2,0) = Rinv.at<float>(2,0);eig_mat(2,1) = Rinv.at<float>(2,1);eig_mat(2,2) = Rinv.at<float>(2,2);
    eig_mat(3,0) = 0.f; eig_mat(3,1) = 0.f; eig_mat(3,2) = 0.f;
    eig_mat(0, 3) = T.at<float>(0);
    eig_mat(1, 3) = T.at<float>(1);
    eig_mat(2, 3) = T.at<float>(2);
    eig_mat(3, 3) = 1.f;
    cam_pose = eig_mat;
    return cam_pose;
}
