#include <visualiseHelper.hpp>
#include <voHelper.hpp>
#define NUM_FRAMES 4000
#define DEBUGGING_FRAMES 10
#define FIRST_FRAME 0
#define SKIPPING_THRESHOLD 100
#define MATCHES_THRESHOLD 30

int main() {
    cv::Mat             image2, image1;
    std::string ground_truth_poses_path = "/home/shasank-gunturu/Downloads/data_odometry_poses/dataset/poses/00.txt";
    std::string folder                  = "/home/shasank-gunturu/Downloads/data_odometry_gray/dataset/sequences/00/image_0/";
    std::string image_format            = ".png";
    std::string estimated_poses_path    = "/home/shasank-gunturu/personal/visual_odometry_CPP/output/poses1.txt";
    cv::Mat K                           = (cv::Mat_<double>(3,3) <<
                                            7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02,
                                            0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 
                                            0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00);
    cv::Mat firstPose                   = (cv::Mat_<double>(3,4) << 1.000000e+00, 9.043680e-12, 2.326809e-11, 5.551115e-17,
                                            9.043683e-12, 1.000000e+00, 2.392370e-10, 3.330669e-16,
                                            2.326810e-11, 2.392370e-10, 9.999999e-01, -4.440892e-16,
                                            0,0,0,1);
    image1              = cv::imread(folder + cv::format("%06d", FIRST_FRAME) + image_format);
    vo                  vo_obj;
    vo_obj.K            = K;
    int total_frames    = NUM_FRAMES;
    std::vector<std::vector<double>> ground_truth_poses = loadGroundTruthPoses(ground_truth_poses_path);
    bool debugging      = false;
    std::vector<Eigen::Affine3f> final_poses;
    if (debugging==true) {
        total_frames    = DEBUGGING_FRAMES;
    }
    cv::Mat prevPose    = firstPose;
    for (int i=FIRST_FRAME+1; i<total_frames; i++) {
        image2          = cv::imread(folder + cv::format("%06d", i) + image_format); 
        vo_obj.image1   = image1;
        vo_obj.image2   = image2;
        cv::Mat T       = vo_obj.getT(SKIPPING_THRESHOLD);
        // cv::Mat T_inv   = T.inv();
        if (i%8==0) {
            prevPose    = nthGroundTruthPose(ground_truth_poses_path, i);
        }
        else {
            prevPose        = prevPose*T.inv();
        }
        final_poses.push_back(mat2Eigen(prevPose));
        image1          = image2;
        std::cout << "Image : " << i <<"\n#################################################################\n\n\n";
    }
    savePosesToFile(ground_truth_poses, final_poses, estimated_poses_path);
    plotPosesPCL(ground_truth_poses, final_poses);
}