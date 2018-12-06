#ifndef MAP_H
#define MAP_H

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/mappoint.h"
#include <opencv2/features2d/features2d.hpp>

namespace myslam
{
class Map
{
public:
    typedef shared_ptr<Map> Ptr;
    Camera::Ptr  camera_;  
    map<unsigned long, MapPoint::Ptr >  map_points_;        // all landmarks
    map<unsigned long, Frame::Ptr >     keyframes_;         // all key-frames
    cv::FlannBasedMatcher   matcher_flann_;     // flann matche	插入每一个mappoint后用于对图内关键帧建立连接
    //int cur_frame_;		//用来辅助frame记录自己是当前地图中的第几帧（map_frame_id）
    
    Map() {}
    
    void insertKeyFrame( Frame::Ptr frame );
    void insertMapPoint( MapPoint::Ptr map_point );
    void DeleteKeyFrame( Frame::Ptr frame);				//这个可以去删除当前Frame
    void DeleteFirstKeyFrame();
    void DeleteMappoint( MapPoint::Ptr map_point);
    void InsertAndFuseMapPoint(vector<MapPoint::Ptr> vMappoint_to_map);		//这个是向局部地图中添加新建的mappoint
    int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);	//计算两个描述之间的距离
};
}

#endif // MAP_H
