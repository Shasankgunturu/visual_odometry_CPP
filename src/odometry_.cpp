#include<twoImageOdometry.hpp>
#include <fstream>

// Modify the below parameters before executing the code.
#define NUM_FRAMES 301
#define FIRST_FRAME 15
#define SKIPPING_THRESHOLD 10
#define MATCHES_THRESHOLD 10

std::string folder                  = "/home/shasankgunturu/personal/ComputerVisionBasics/src/images/data1";
std::string image_format            = ".png";
std::string estimated_poses_path    = "/home/shasankgunturu/personal/ComputerVisionBasics/src/output/poses.txt";
cv::Mat K                           = (cv::Mat_<double>(3, 3) << 
                                        1253.1012617250694, 0.0, 471.6666914725001, 
                                        0.0, 1233.192145117128, 327.9885858399886, 
                                        0.0, 0.0, 1.0);
cv::Mat P                           = (cv::Mat_<double>(3,4) <<
                                        1165.624268, 0.000000, 418.089806, 0.000000,
                                        0.000000, 1252.924683, 333.552941, 0.000000,
                                        0.000000, 0.000000, 1.000000, 0.000000);

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
    cv::Mat image2, curr_pose, image1;     
    image1              = cv::imread(folder+"/captured_image_"+ std::to_string(FIRST_FRAME) +image_format, cv::IMREAD_GRAYSCALE);
    
    Odometry            o;
    o.K                 = K;
    o.P                 = P;
    
    std::vector<std::vector<double>> final_poses;

    // Loop to process the images
    for (int i=FIRST_FRAME+1; i<NUM_FRAMES; i++) {
        cv::Mat                     new_pose;
        image2                      = cv::imread(folder+"/captured_image_"+ std::to_string(i) +image_format, cv::IMREAD_GRAYSCALE);
        o.image1                    = image1;
        o.image2                    = image2;
        cv::Mat new_transform       = o.getOdom(MATCHES_THRESHOLD);
        if (i==FIRST_FRAME+1) {
            new_pose                = new_transform;
        }
        else {
            new_pose                = curr_pose*new_transform.inv();
            cv::Mat tester          = new_pose-curr_pose;
            cv::Vec4f test_point    = tester.col(3);
            // Check for Outliers
            if ((test_point[0]>SKIPPING_THRESHOLD) || (test_point[1]>SKIPPING_THRESHOLD) || (test_point[2]>SKIPPING_THRESHOLD)) {
                std::cout << "Skipping Frame: " << i-FIRST_FRAME << std::endl;
                continue;
            }
        }
        curr_pose                   = new_pose;
        image1                      = image2;
        cv::Vec4f estimated_point   = new_pose.col(3);
        std::cout                   << "Handling Image: " << i-FIRST_FRAME << std::endl;
        final_poses.push_back({estimated_point[0], estimated_point[1], estimated_point[2]});
    }
    savePosesToFile(final_poses, estimated_poses_path);
}