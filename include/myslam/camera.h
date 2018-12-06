#ifndef CAMERA_H
#define CAMERA_H

#include "myslam/common_include.h"

namespace myslam
{

// Pinhole RGBD camera model
class Camera
{
public:
    typedef std::shared_ptr<Camera> Ptr;
    float   fx_, fy_, cx_, cy_, k1_, k2_, k3_, p1_, p2_;

    Camera();
    Camera ( float fx, float fy, float cx, float cy, float k1, float k2, float k3, float p1, float p2) :
        fx_ ( fx ), fy_ ( fy ), cx_ ( cx ), cy_ ( cy ), k1_( k1 ), k2_( k2 ), k3_( k3 ), p1_( p1 ), p2_( p2 ) 
    {}

    // coordinate transform: world, camera, pixel
    Vector3d world2camera( const Vector3d& p_w, const SE3& T_c_w );
    Vector3d camera2world( const Vector3d& p_c, const SE3& T_c_w );
    Vector2d camera2pixel( const Vector3d& p_c );
    Vector2d world2pixel ( const Vector3d& p_w, const SE3& T_c_w );

};

}
#endif // CAMERA_H
