#ifndef VISUALODOMETRY_H
#define VISUALODOMETRY_H

#include "myslam/common_include.h"
#include "myslam/map.h"
#include "projection.h"
#include "json/json.h"

#include <opencv2/features2d/features2d.hpp>

namespace myslam 
{
class VisualOdometry
{
public:
    typedef shared_ptr<VisualOdometry> Ptr;
    enum VOState {
        INITIALIZING1=-2,
	INITIALIZING2=-1,
        OK=0,
        LOST
    };
    
    VOState     state_;     // current VO status
    Map::Ptr    map_;       // map with all frames and map points
    
    Frame::Ptr  ref_;       // reference key-frame 			该关键帧为局部地图中的参考关键帧，将用于重建空间三位点
    Frame::Ptr  curr_;      // current frame 
    Frame::Ptr  last_;	//last frame
    
    vector<vector<cv::Vec3d>> FreeSpacePoints_;	//读取的freespace
    
    cv::Ptr<cv::ORB> orb_;  // orb detector and computer 
    cv::Ptr<Feature2D> sift_ ;	//sift
    vector<cv::KeyPoint>    keypoints_curr_;    // keypoints in current frame
    Mat                     descriptors_curr_;  // descriptor in current frame 
    
    vector< DMatch > cur_good_matches;	//当前帧与上一关键帧之间的匹配
    
    cv::FlannBasedMatcher   matcher_flann_;     // flann matcher
    vector<MapPoint::Ptr>   match_3dpts_;       // matched 3d points 
    vector<int>             match_2dkp_index_;  // matched 2d pixels (index of kp_curr)
   
    SE3 T_c_w_estimated_;    // the estimated pose of current frame 
    Mat Tcw_;		//在世界坐标系下的位置
    Matrix3d R_to_ref_frame_;
    Vector3d t_to_ref_frame_;
    int num_inliers_;        // number of inlier features in icp
    int num_lost_;           // number of lost times
    
    // parameters 
    int num_of_imgs_;	//number of total imgs
    int num_of_features_;   // number of features
    double scale_factor_;   // scale in image pyramid
    int level_pyramid_;     // number of pyramid levels
    float match_ratio_;     // ratio for selecting  good matches
    int max_num_lost_;      // max number of continuous lost times
    int min_inliers_;       // minimum inliers
    double key_frame_min_rot;   // minimal rotation of two key-frames
    double key_frame_min_trans; // minimal translation of two key-frames
    double  map_point_erase_ratio_; // remove map point ratio				可以适当提高删除mappoint的阈值～
    
    int Json_frame_id_;		//这个用来标记记录读json文件时的id号
    
public: // functions 
    VisualOdometry();
    ~VisualOdometry();
    
    bool addFrame( Frame::Ptr frame );      // add a new frame 
    
protected:  
    // inner operation 
    void ReadJson();
    void extractKeyPoints();
    void computeDescriptors(); 
    void featureMatching();			//特征匹配，同时恢复当前帧RT位姿
    void DeleteMappoint();
    void OptimizeTheFrame();		//这个函数执行的是普通帧每一帧内都要执行的优化，现在就很麻烦，包含了插入帧之后又删除帧
    void BundleAdjustment();			//执行BA，注意BA的对象包含局部地图中全部关键帧KF 空间点mappoint以及当前帧Frame，每帧执行
								//偏类似于滑动窗口FIFO，可以考虑放宽KF的加入，同时加快KF的删除，使IO提速
    void addKeyFrame();
    bool checkEstimatedPose(); 		//检测当前帧的预测是否合理，分为检测内点数和单帧移动距离
    bool checkKeyFrame();
    void reconstruct(Mat& K, Mat& R, Mat& T, vector<Point2f>& p1, vector<Point2f>& p2, Mat& structure);		//三角化空间点
    void MakeTheInitialMap();	
    void addMapPointsAndKeyFrame();
    
    double getViewAngle( Frame::Ptr frame, MapPoint::Ptr point );
    
};
}

#endif // VISUALODOMETRY_H
