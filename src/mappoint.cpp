#include "myslam/common_include.h"
#include "myslam/mappoint.h"

namespace myslam
{

MapPoint::MapPoint()
: id_(-1), pos_(Vector3d(0,0,0)), norm_(Vector3d(0,0,0)), good_(true), visible_times_(0), matched_times_(0)
{

}

MapPoint::MapPoint ( long unsigned int id, const Vector3d& position, const Vector3d& norm, Frame* ref_frame, const Mat& descriptor, int ref_keypoint_id)
: id_(id), pos_(position), norm_(norm), ref_KF_1_(ref_frame), good_(true), visible_times_(1), matched_times_(1), descriptor_(descriptor), ref_KF_id_(ref_keypoint_id)
{
    observed_frames_.push_back(ref_frame);
    matched_keypoint_[ref_frame] = ref_keypoint_id;					//ATTENTION 	注意一下这两个变量的添加，能否在这个位置添加
    
}

MapPoint::Ptr MapPoint::createMapPoint ( 
    const Vector3d& pos_world, 
    const Vector3d& norm, 
    const Mat& descriptor, 
    Frame* ref_frame,
    int ref_keypoint_id
    )
{
    return MapPoint::Ptr( 
        new MapPoint( factory_id_++, pos_world, norm, ref_frame, descriptor, ref_keypoint_id )
    );
}

unsigned long MapPoint::factory_id_ = 0;		//这个用法其实我不太清楚该咋做

}
