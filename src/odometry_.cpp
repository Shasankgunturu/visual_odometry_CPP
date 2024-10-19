#include<twoImageOdometry.hpp>
#include <fstream>
#define NUM_FRAMES 55
#define FIRST_FRAME 12

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


int main() {
    cv::Mat K       = (cv::Mat_<double>(3, 3) << 
                            1253.1012617250694, 0.0, 471.6666914725001, 
                            0.0, 1233.192145117128, 327.9885858399886, 
                            0.0, 0.0, 1.0);
    cv::Mat P       = (cv::Mat_<double>(3,4) <<
                            1165.624268, 0.000000, 418.089806, 0.000000,
                            0.000000, 1252.924683, 333.552941, 0.000000,
                            0.000000, 0.000000, 1.000000, 0.000000);
    cv::Mat image1  = cv::imread("/home/shasankgunturu/personal/ComputerVisionBasics/src/images/data/captured_image_"+ std::to_string(FIRST_FRAME) +".jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat image2; 
    // if (image1.empty() || image2.empty()) {
    //     std::cerr << "can't find images" << std::endl;
    //     return -1;
    // }
    
    Odometry        o;
    o.K             = K;
    o.P             = P;
    cv::Mat curr_pose;
    std::vector<std::vector<double>> final_poses;
    for (int i=FIRST_FRAME+1; i<NUM_FRAMES; i++) {
        image2  = cv::imread("/home/shasankgunturu/personal/ComputerVisionBasics/src/images/data/captured_image_"+ std::to_string(i) +".jpg", cv::IMREAD_GRAYSCALE);
        o.image1        = image1;
        o.image2        = image2;
        cv::Mat new_transform = o.getOdom(10);
        cv::Mat new_pose;
        // std::cout << new_transform << std::endl;
        if (i==FIRST_FRAME+1) {
            new_pose = new_transform;
        }
        else {
            new_pose = curr_pose*new_transform.inv();
        }
        // std::cout << new_pose;
        curr_pose = new_pose;
        cv::Vec4f estimated_point = new_pose.col(3);
        final_poses.push_back({estimated_point[0], estimated_point[1], estimated_point[2]});
        image1 = image2;
        std::cout << "Handling Image: " << i-FIRST_FRAME << std::endl;
    }
    std::string filename = "/home/shasankgunturu/personal/ComputerVisionBasics/src/output/poses.txt";
    savePosesToFile(final_poses, filename);
}