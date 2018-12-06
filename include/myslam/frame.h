#ifndef FRAME_H
#define FRAME_H

#include "myslam/common_include.h"
#include "myslam/camera.h"

namespace myslam 
{
    
// forward declare 
class MapPoint;
class Frame
{
public:
    typedef std::shared_ptr<Frame> Ptr;
    unsigned long                  id_;         // id of this frame
    double                         time_stamp_; // when it is recorded
    SE3                            T_c_w_;      // transform from world to camera
    Mat				RT_Mat;		//相机的位姿矩阵		TODO
    Mat 				RT_Mat_to_ceres;	//这个矩阵是长度为6，用于ceres-solver做优化用的 
    Camera::Ptr                    camera_;     // Pinhole  Camera model 
    Mat                            color_; // color and depth image 					没有深度信息
    std::vector<cv::KeyPoint>      keypoints_;  // key points in image
    
    Mat                     		descriptors_;
    std::vector<MapPoint*>         map_points_; // associated map points
    bool                           is_key_frame_;  // whether a key-frame
    int				map_frame_id_;		//记录当前帧在局部地图中的id
    
    vector<MapPoint*> observed_mappoint_;		//当前帧可以看到哪些mappoint		
    //ATTENTION 这个变量里面的mappoint会不会发生重复呢？？？
    
    map<int, MapPoint*> matched_mappoint_;		//因此关键帧也必须去记录它能看到的这个keypoint对应于空间中的哪个mappoint
											//由mappoint找frame是mappoint类干的事
											//此外这个变量也可以用来判断当前keypoint是否有空间中的mappoint与之对应
public: // data members 
    Frame();
    Frame( long id, double time_stamp=0, SE3 T_c_w=SE3(), Camera::Ptr camera=nullptr, Mat color=Mat(), Mat depth=Mat() );
    ~Frame();
    
    static Frame::Ptr createFrame(); 
    
    // find the depth in depth map
    double findDepth( const cv::KeyPoint& kp );		//没有深度信息函数
    
    // Get Camera Center
    Vector3d getCamCenter() const;
    
    void setPose( const SE3& T_c_w );
    
    // check if a point is in this frame 
    bool isInFrame( const Vector3d& pt_world );
    
};

}

#endif // FRAME_H
