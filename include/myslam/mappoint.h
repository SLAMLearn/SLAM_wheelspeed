#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "myslam/common_include.h"

namespace myslam
{
    
class Frame;
class MapPoint
{
public:
    typedef shared_ptr<MapPoint> Ptr;
    unsigned long      id_;        // ID
    static unsigned long factory_id_;    // factory id		注意这个静态变量不属于任意对象
    bool        good_;      // wheter a good point 		TODO 现在还没太用到
    Vector3d    pos_;       // Position in world
    Vector3d    norm_;      // Normal of viewing direction 
    Mat         descriptor_; // Descriptor for matching 
    
    vector<Frame*>    observed_frames_;   // key-frames that can observe this point 		
    map<Frame*, int> matched_keypoint_;		//需要记录他匹配其可视关键帧中的哪一个keypoint			
    
    int         matched_times_;     // being an inliner in pose estimation			这个回头需要注意		还是先不考虑了吧	TODO
    int         visible_times_;     // being visible in current frame 
    
    Frame*  ref_KF_1_;		//建立该mappoint的参考关键帧
    int ref_KF_id_;
    
    MapPoint();
    MapPoint( 
        unsigned long id, 
        const Vector3d& position, 
        const Vector3d& norm, 
        Frame* ref_frame = nullptr, 
        const Mat& descriptor=Mat(), 		//还是在用这个表示描述子
	int ref_keypoint_id = 0	
    );
    
    inline cv::Point3f getPositionCV() const {
        return cv::Point3f( pos_(0,0), pos_(1,0), pos_(2,0) );
    }
    
    static MapPoint::Ptr createMapPoint( 
        const Vector3d& pos_world, 
        const Vector3d& norm_,
        const Mat& descriptor,
        Frame* ref_frame,
	int ref_keypoint_id );
};
}

#endif // MAPPOINT_H
