#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/sfm/fundamental.hpp>

class vo {
    private:
        std::vector<cv::Point2f> img1_pts, img2_pts;
        cv::Mat convertRt2T_4x4(const cv::Mat &R, const cv::Mat &t) {
            cv::Mat T = (cv::Mat_<double>(4, 4) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
                        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
                        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0),
                        0,0,0,1);
            return T;
        }
        cv::Mat convertK2T_3x4(const cv::Mat &T) {
            cv::Mat T_ = (cv::Mat_<double>(3, 4) << T.at<double>(0, 0), T.at<double>(0, 1), T.at<double>(0, 2), 0,
                        T.at<double>(1, 0), T.at<double>(1, 1), T.at<double>(1, 2), 0,
                        T.at<double>(2, 0), T.at<double>(2, 1), T.at<double>(2, 2), 0);
            return T_;
        }
        cv::Mat convertFrom4DHomogeneous(cv::Mat points4D) {
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
        void bestKeypointFinderORB(int threshold, bool view) {
            img1_pts.clear();
            img2_pts.clear();
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
                if (matches[i].distance     <= std::max(2*min_dist, threshold)) {
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
        cv::Mat getRT_CV(cv::Mat essentialMatrix) {
            cv::Mat             R,t,mask;
            cv::recoverPose     (essentialMatrix, img1_pts, img2_pts, K, R, t, mask);
            cv::Mat T           = convertRt2T_4x4(R, t);
            return T;
        }
        cv::Mat getRT(cv::Mat essentialMatrix) {
            cv::Mat                     R1, R2, _t;
            std::vector<double>         sum_z{0,0,0,0}, scale{0,0,0,0};
            cv::decomposeEssentialMat   (essentialMatrix, R1, R2, _t);
            std::vector<cv::Mat>        R{R1,R1,R2,R2}; 
            std::vector<cv::Mat>        t{_t, -_t, _t, -_t};
            for (int i=0; i<R.size(); i++) {
                cv::Mat pts4D_1, pts4D_2;
                cv::Mat T_temp = convertRt2T_4x4(R[i], t[i]);
                T_temp.convertTo(T_temp, K.type());
                cv::Mat P_temp = convertK2T_3x4(K) * T_temp;
                cv::triangulatePoints(convertK2T_3x4(K), P_temp, img1_pts, img2_pts, pts4D_1);
                pts4D_2.convertTo(pts4D_2, K.type());
                pts4D_1.convertTo(pts4D_1, K.type());
                pts4D_2 = (T_temp * pts4D_1);
                cv::Mat pts3D_1 = convertFrom4DHomogeneous(pts4D_1.t());
                cv::Mat pts3D_2 = convertFrom4DHomogeneous(pts4D_2.t());
                std::cout << "Sizes: "<< pts3D_1.size << ", " <<pts3D_2.size   <<"\n";
                for (int j=0; j<pts3D_1.rows; j++) {
                    cv::Vec3f point1    = pts3D_1.at<cv::Vec3f>(j);
                    cv::Vec3f point2    = pts3D_2.at<cv::Vec3f>(j);
                    if (point1[2]>0) {
                        sum_z[i]        = sum_z[i] + 1;    
                    }
                    if (point2[2]>0) {
                        sum_z[i]        = sum_z[i] + 1;
                    }
                }
                for (int j=0; j<pts3D_2.rows-1; j++) {
                    cv::Vec3f p1_1      = pts3D_1.at<cv::Vec3f>(j);
                    cv::Vec3f p2_1      = pts3D_1.at<cv::Vec3f>(j+1);
                    cv::Vec3f p1_2      = pts3D_2.at<cv::Vec3f>(j);
                    cv::Vec3f p2_2      = pts3D_2.at<cv::Vec3f>(j+1);
                    scale[i]            = scale[i] + (cv::norm(p1_1 - p2_1)/cv::norm(p1_2 - p2_2));
                }
                scale[i]                = scale[i]/(pts3D_1.rows-1);
            }
            auto maxElementIter = std::max_element(sum_z.begin(), sum_z.end());
            int maxIndex = std::distance(sum_z.begin(), maxElementIter);
            t[maxIndex]                 = t[maxIndex]*scale[maxIndex];
            cv::Mat T   = convertRt2T_4x4(R[maxIndex], t[maxIndex]);
            return T;

        }
        cv::Mat getMotion() {
            cv::Mat             mask, essentialMatrix, R, t;
            essentialMatrix     = cv::findEssentialMat(img1_pts, img2_pts, K, cv::RANSAC,  0.999, 1.0, mask);
            return getRT(essentialMatrix);
        }
    public:
        cv::Mat K, image1, image2;
        cv::Mat getT(int threshold, bool view=false) {
            bestKeypointFinderORB(threshold, view);
            return getMotion();
        }
};