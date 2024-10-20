#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/sfm/fundamental.hpp>

class Odometry {
    /*
    // Alternate method to find pose
    cv::Mat fundamentalMatrix = cv::findFundamentalMat(img1_pts, img2_pts, cv::FM_LMEDS);
    std::cout << "Fundamental Matrix:" << std::endl;
    std::cout << fundamentalMatrix << std::endl;
    cv::Mat essentialMatrix;
    cv::sfm::essentialFromFundamental(fundamentalMatrix, K, K, essentialMatrix);
    std::vector<cv::Mat> R, t;
    try {
        cv::sfm::motionFromEssential(essentialMatrix, R, t);
        std::cout << "Rotation Matrix:" << std::endl;
        for (int i=0; i<R.size(); i++) {
            std::cout << R[i] << std::endl;
        }
        std::cout << std::endl;
        std::cout << "Translation Matrix:" << std::endl;
        for (int i=0; i<t.size(); i++) {
            std::cout << t[i] << std::endl;
        }
        cv::Mat t0 = t[0];
        std::cout << std::endl;
        std::cout << "Odom: ";
        std::cout << sqrt(t0.at<double>(0,0) * t0.at<double>(0,0) +
                            t0.at<double>(0,1) * t0.at<double>(0,1) +
                            t0.at<double>(0,2) * t0.at<double>(0,2)) << std::endl<< std::endl;

        for (int i=0; i<t.size(); i++) {
            cv::Mat result;
            result = R[i].inv() * t[i];
            std::cout << "Result: " << result << std::endl;
            std::cout << "ODOM: " << sqrt(result.at<double>(0,0) * result.at<double>(0,0) +
                            result.at<double>(0,1) * result.at<double>(0,1) +
                            result.at<double>(0,2) * result.at<double>(0,2)) << std::endl << std::endl;
        }

    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Exception: " << e.what() << std::endl;
        return -1;
    }
    */
    public:
    
        cv::Mat     image1, image2, K, C1, P;
        double      roll, pitch, yaw;

        cv::Mat getOdom(int threshold = 10, bool view=false) {
            bestKeypointFinderORB(threshold, view);
            return getPose();
        }
    private:

        std::vector<cv::Point2f> img1_pts, img2_pts;
        
        void bestKeypointFinderORB(int threshold, bool view) {
            cv::Ptr<cv::ORB> orb            = cv::ORB::create();
            std::vector<cv::KeyPoint>       keypoints1, keypoints2;
            cv::Mat                         descriptors1, descriptors2;
            std::vector<cv::DMatch>         matches;
            std::vector<cv::DMatch>         best_matches;
            
            orb->detectAndCompute           (image1, cv::Mat(), keypoints1, descriptors1);
            orb->detectAndCompute           (image2, cv::Mat(), keypoints2, descriptors2);
            cv::BFMatcher matcher(cv::NORM_HAMMING, true);
            matcher.match(descriptors1, descriptors2, matches);

            int min_dist = 10000, max_dist = 0;
            for (int i=0; i<matches.size(); i++) {
                if (matches[i].distance     < min_dist) min_dist = matches[i].distance;
                if (matches[i].distance     > max_dist) max_dist = matches[i].distance;
            }
            for (int i=0; i<matches.size(); i++) {
                if (matches[i].distance     <= 10*min_dist) {
                    best_matches.push_back  (matches[i]);
                }
            }
            for (size_t i=0; i<best_matches.size(); i++) {
                cv::Point2f point1          = keypoints1[best_matches[i].queryIdx].pt;
                cv::Point2f point2          = keypoints2[best_matches[i].trainIdx].pt;
                img1_pts.push_back          (point1);
                img2_pts.push_back          (point2);
            }
            if (view) {
                cv::Mat image1_with_points  = image1.clone();
                cv::Mat image2_with_points  = image2.clone();
                
                for (size_t i = 0; i < img1_pts.size(); i++) {
                    cv::circle              (image1_with_points, img1_pts[i], 5, cv::Scalar(255, 0, 0), 2);
                    cv::circle              (image2_with_points, img2_pts[i], 5, cv::Scalar(255, 0, 0), 2);
                }

                std::cout << "No of matches: "      << matches.size() << std::endl;
                std::cout << "No of best matches: " << best_matches.size() << std::endl;
                cv::imshow("Image 1 - Keypoints", image1_with_points);
                cv::imshow("Image 2 - Keypoints", image2_with_points);
                cv::waitKey(0);
            }
        }
        
        cv::Mat convertFromHomogeneous(cv::Mat points4D) {
            cv::Mat points3D(points4D.rows, points4D.cols-1, points4D.type());
            for (int i = 0; i < points4D.rows; ++i) {
                cv::Vec4f row = points4D.row(i);
                double w = row[3];
                points3D.at<cv::Vec3f>(i)[0] = row[0] / w;
                points3D.at<cv::Vec3f>(i)[1] = row[1] / w;
                points3D.at<cv::Vec3f>(i)[2] = row[2] / w;
            }
            return points3D;
        }

        std::pair<cv::Mat, cv::Mat> getRT(cv::Mat essentialMatrix) {
            cv::Mat                     R1, R2, _t;
            std::vector<double>         sum_z{0,0,0,0}, scale{0,0,0,0};
            cv::decomposeEssentialMat   (essentialMatrix, R1, R2, _t);
            std::vector<cv::Mat>        R{R1,R1,R2,R2}; 
            std::vector<cv::Mat>        t{_t, -_t, _t, -_t};
            for (int i=0; i<R.size(); i++) {
                cv::Mat T               = cv::Mat::eye(4, 4, K.type());  
                R[i].copyTo             (T(cv::Rect(0, 0, 3, 3)));
                t[i].copyTo             (T(cv::Rect(3, 0, 1, 3)));              
                cv::Mat                 K_ext, point4D_1, point3D_1, point3D_2, T_changedType;
                cv::hconcat             (K, cv::Mat::zeros(3, 1, K.type()), K_ext);
                cv::Mat P_ext           = K_ext * T;
                cv::triangulatePoints   (P, P_ext, img1_pts, img2_pts, point4D_1);
                T.convertTo             (T_changedType, point4D_1.type());
                cv::Mat point4D_2       = T_changedType*point4D_1;
                point3D_1               = convertFromHomogeneous(point4D_1.t());
                point3D_2               = convertFromHomogeneous(point4D_2.t());
                for (int j=0; j<point3D_1.rows; j++) {
                    cv::Vec3f point1    = point3D_1.at<cv::Vec3f>(j);
                    cv::Vec3f point2    = point3D_2.at<cv::Vec3f>(j);
                    if (point1[2]>0) {
                        sum_z[i]        = sum_z[i] + point1[2];    
                    }
                    if (point2[2]>0) {
                        sum_z[i]        = sum_z[i] + point2[2];
                    }
                }
                for (int j=0; j<point3D_1.rows-1; j++) {
                    cv::Vec3f p1_1      = point3D_1.at<cv::Vec3f>(j);
                    cv::Vec3f p2_1      = point3D_1.at<cv::Vec3f>(j+1);
                    cv::Vec3f p1_2      = point3D_2.at<cv::Vec3f>(j);
                    cv::Vec3f p2_2      = point3D_2.at<cv::Vec3f>(j+1);
                    scale[i]            = scale[i] + (cv::norm(p1_1 - p1_2)/cv::norm(p2_1 - p2_2));
                }
                scale[i]                = scale[i]/(point3D_1.rows-1);
            }
            double max_z                = 0;
            int                         return_index;
            for (int i=0; i<sum_z.size(); i++) {
                if (sum_z[i]>max_z) {
                    max_z               = sum_z[i];
                    return_index        = i;
                }
            }
            t[return_index]             = t[return_index]*scale[return_index];

            return std::make_pair(R[return_index], t[return_index]);
        }

        cv::Mat getPose() {
            cv::Mat                     essentialMatrix, mask;
            essentialMatrix             = cv::findEssentialMat(img1_pts, img2_pts, K, cv::RANSAC,  0.999, 1.0, mask);
            auto RT                     = getRT(essentialMatrix);
            cv::Mat R                   = RT.first;
            cv::Mat t                   = RT.second;
            rotationMatrixToEulerAngles (R, t);
            C1                          = R * t;
            cv::Mat transform           = cv::Mat::eye(4, 4, K.type());  
            R.copyTo                    (transform(cv::Rect(0, 0, 3, 3)));
            t.copyTo                    (transform(cv::Rect(3, 0, 1, 3)));   
            return transform;
        }
        
        void rotationMatrixToEulerAngles(cv::Mat R, cv::Mat t) {
            assert  (R.rows == 3 && R.cols == 3);
            pitch   = atan2(-R.at<double>(2, 0), sqrt(R.at<double>(2, 1) * R.at<double>(2, 1) + R.at<double>(2, 2) * R.at<double>(2, 2)));
            yaw     = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
            roll    = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        }
};

