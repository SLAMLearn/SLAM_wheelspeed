
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>
#include <boost/timer.hpp>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"
#include "myslam/g2o_types.h"

namespace myslam
{

    VisualOdometry::VisualOdometry() :
            state_ ( INITIALIZING1 ), ref_ ( nullptr ), curr_ ( nullptr ), map_ ( new Map ), num_lost_ ( 0 ), num_inliers_ ( 0 ), matcher_flann_ ( new cv::flann::LshIndexParams ( 5,10,2 ) )
    {
        num_of_features_    = Config::get<int> ( "number_of_features" );
        scale_factor_       = Config::get<double> ( "scale_factor" );
        level_pyramid_      = Config::get<int> ( "level_pyramid" );
        match_ratio_        = Config::get<float> ( "match_ratio" );
        max_num_lost_       = Config::get<float> ( "max_num_lost" );
        min_inliers_        = Config::get<int> ( "min_inliers" );
        key_frame_min_rot   = Config::get<double> ( "keyframe_rotation" );
        key_frame_min_trans = Config::get<double> ( "keyframe_translation" );
        map_point_erase_ratio_ = Config::get<double> ( "map_point_erase_ratio" );
        orb_ = cv::ORB::create ( num_of_features_, scale_factor_, level_pyramid_ );
	sift_ = xfeatures2d::SIFT::create(0, 3, 0.04, 10, 1.6);
    }

    VisualOdometry::~VisualOdometry()
    {

    }

    bool VisualOdometry::addFrame ( Frame::Ptr frame )
    {
        switch ( state_ )
        {
            case INITIALIZING1:
            {
		ReadJson();
		Json_frame_id_ = 0;
                state_ = OK;
                last_ = curr_ = ref_ = frame;
                // extract features from first frame and add them into map
                extractKeyPoints();
                computeDescriptors();
                addKeyFrame();      // the first frame is a key-frame
                state_ = INITIALIZING2;
                break;
            }
            case INITIALIZING2:
            {
		Json_frame_id_++;
                curr_ = frame;
                // extract features from first frame and add them into map
                extractKeyPoints();
                computeDescriptors();
                featureMatching();
                MakeTheInitialMap();
                //BundleAdjustment();
		//OptimizeTheFrame();			//我觉得还是得用这个把。。第二帧我是不作为关键帧加入的啊
                state_ = OK;
		last_ = curr_;
                break;
            }
            case OK:
            {
		Json_frame_id_++;
                curr_ = frame;
                extractKeyPoints();
                computeDescriptors();
                featureMatching();      //每一帧都要和关键帧去匹配
		//OptimizeTheFrame();		//当前帧入局部地图，做一个简单的BA

                if ( checkEstimatedPose() == true ) // a good estimation		这个判断如果没进入也就崩了。。
                {
                    DeleteMappoint();
                    num_lost_ = 0;
                    if ( checkKeyFrame() == true ) // is a key-frame
                    {
                        addMapPointsAndKeyFrame();
                    }
                }
                else // bad estimation due to various reasons           lost的次数太多就崩了			TODO 需要补充RESET
                {
                    num_lost_++;
                    if ( num_lost_ > max_num_lost_ )		
                    {
                        state_ = LOST;
                    }
                    return false;
                }
                last_ = curr_;
                break;
            }
            case LOST:
            {
                cout<<"vo has lost."<<endl;
                break;
            }
        }

        return true;
    }

    void VisualOdometry::extractKeyPoints()		//包括去除畸变和去除json外的点
    {
	
	Mat ImgUndistort =curr_->color_.clone();				//图片去畸变
	Mat K = Mat(3, 3, CV_32FC1);
	Mat Distort;
		
		
	K = (Mat_<double>(3, 3) << 	ref_->camera_->fx_, 0, ref_->camera_->cx_,
						      0, ref_->camera_->fy_, ref_->camera_->cy_,
						      0, 0, 1);


	Distort = (Mat_<double>(5, 1) <<  ref_->camera_->k1_,  ref_->camera_->k2_, ref_->camera_->p1_, ref_->camera_->p2_,   ref_->camera_->k3_);
	
	undistort(curr_->color_, ImgUndistort, K, Distort);
	curr_->color_ = ImgUndistort.clone();
	
      
        boost::timer timer;
        //orb_->detect ( curr_->color_, keypoints_curr_ );					//特征点提取
	sift_->detect ( curr_->color_, keypoints_curr_ );
        cout<<"extract keypoints cost time: "<<timer.elapsed() <<endl;
	cout<<"keypoint size:"<< keypoints_curr_.size()<<endl;       
        for (vector<KeyPoint>::iterator it = keypoints_curr_.begin(); it != keypoints_curr_.end();){	//遍历每一个点
		cv::KeyPoint KP = *it;
		double  pos_x = KP.pt.x ;
		double pos_y = KP.pt.y ;
		bool to_delete = false;
      
		vector<double> points_x;
		vector<double> points_y;
		
		//cout<<"current frame id:"<<Json_frame_id_<<endl;
		//cout<<"vvFreeSpacePoints :"<<vvFreeSpacePoints[Json_frame_id_].size()<<endl;
		
		for(int i = 0; i < FreeSpacePoints_[Json_frame_id_].size(); i++){		//遍历当前帧下每一个freespace点
		  cv::Vec3d point = FreeSpacePoints_[Json_frame_id_][i];
		  points_x.push_back(point[0]);
		  points_y.push_back(point[1]);
		 // cout<<"x:"<<point[0]<<"y:"<<point[1]<<endl;
		}
		int num_of_points = FreeSpacePoints_[Json_frame_id_].size();
		
		int pos = 0;
		while(pos < points_x.size() && pos_x > points_x[pos] )		pos++;
		//cout<<"pos:"<<pos<<endl;
		if(pos == points_x.size() -1 || pos == points_x.size() - 2)	to_delete = true;			//这个终究会变成预编译阶段拟合的一条曲线，直接一步比较就ok
		double x1 = points_x[pos];													//不用一次次遍历
		double y1 = points_y[pos];
		pos++;
		double x2 = points_x[pos];
		double y2 =  points_y[pos];
		
		double target = y1 + (pos_x - x1) * (y2 - y1) / (x2 - x1);
		//cout<<"pos_y:"<<pos_y<<" target:"<<target<<endl;
		if(pos_y < target)	to_delete = true;
     
		if(to_delete){
		 // it++;
		it = keypoints_curr_.erase(it);
		    cout<<"finish the delete"<<endl;
		}
		else	it++;
		
	}

	cout<<"keypoint size:"<< keypoints_curr_.size()<<endl;
    }

    void VisualOdometry::computeDescriptors()
    {
        boost::timer timer;
        //orb_->compute ( curr_->color_, keypoints_curr_, descriptors_curr_ );
	sift_->compute ( curr_->color_, keypoints_curr_, descriptors_curr_ );
        cout<<"descriptor computation cost time: "<<timer.elapsed() <<endl;
    }

    void VisualOdometry::featureMatching()			//这个应该变成二维的，并且同时计算H恢复出RT
    {
		FlannBasedMatcher  matcher; 
		vector<DMatch> matches;  
		matcher.match(ref_->descriptors_, curr_->descriptors_, matches);       //特征点匹配  
  
		sort(matches.begin(), matches.end());  		//筛选匹配点  
		vector< DMatch > good_matches;                
		//int ptsPairs = std::min(50, (int)(matches.size() * 1));  		//ATTENTION  注意这个系数的调整
		int ptsPairs = matches.size();
		cout << "matches :"<<ptsPairs << endl;  
		for (int i = 0; i < ptsPairs; i++)  
		{  
		    good_matches.push_back(matches[i]);  		//好的特征点
		}  
		//Mat outimg;  
		//drawMatches(cur_frame, cur_key_points, last_frame, last_key_points, good_matches, outimg, Scalar::all(-1), Scalar::all(-1),vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);  //绘制匹配点  
    
		std::vector<Point2f> last_pt;  
		std::vector<Point2f> cur_pt;  
  
		for (size_t i = 0; i < good_matches.size(); i++)  
		{  
		    last_pt.push_back(ref_->keypoints_[good_matches[i].queryIdx].pt);  
		    cur_pt.push_back(curr_->keypoints_[good_matches[i].trainIdx].pt);  
		}  
		
		/*
		Point2f principle_point(ref_->camera_->cx_, ref_->camera_->cy_);
		cv::Mat E = cv::Mat::zeros(3,3,CV_64F);
		E = cv::findEssentialMat(last_pt, cur_pt, 2307.4750772461, principle_point, RANSAC, 0.999, 1.0);			//先这样，回头再看怎么从Ｈ重建
		cout<<E<<endl;
		cv::Mat R = cv::Mat::zeros(3,3,CV_64F);
		cv::Mat T = cv::Mat::zeros(1,3,CV_64F);
		int pass_count = recoverPose(E, last_pt, cur_pt, R, T, 2307.4750772461, principle_point);			//重建得到两帧图片之间的相对位姿关系
		
		Matrix3d Final_R ;
		Final_R<< R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2),
				 R.at<double>(1,0),R.at<double>(1,1),R.at<double>(1,2),
				 R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2)	;
						   
		Vector3d Final_t;
		Final_t<< T.at<double>(0,0),T.at<double>(0,1),T.at<double>(0,2)	;
*/
		
		Mat H = findHomography(last_pt, cur_pt, RANSAC);
		
		Mat K = Mat(3, 3, CV_64FC1);
		K = (Mat_<double>(3, 3) << 	ref_->camera_->fx_, 0, ref_->camera_->cx_,
						      0, ref_->camera_->fy_, ref_->camera_->cy_,
						      0, 0, 1);
		Mat G2G1_inv = K.inv() *  H * K;		//左乘逆，右乘正
		
		Matrix3d	 G1_R =  ref_->T_c_w_.rotation_matrix();
		Vector3d G2_t = ref_->T_c_w_.translation();
		
		Mat G1 = (Mat_<double>(3, 3) << G1_R(0,0), G1_R(1,0), G2_t(0),
										G1_R(1,0), G1_R(1,1), G2_t(1),
										G1_R(2,0), G1_R(1,2), G2_t(2));
		Mat G2 = G2G1_inv * G1;
		
		Vector3d r1(G2.at<double>(0,0), G2.at<double>(1,0), G2.at<double>(2,0));
		Vector3d r2(G2.at<double>(0,1), G2.at<double>(1,1), G2.at<double>(2,1));
		
		Vector3d r1_noraml  = r1;
		Vector3d r2_noraml  = r2;
		
		r1_noraml.normalize();
		r2_noraml.normalize();
		
		Matrix<double, 1, 3>	r3;
		r3 = r1_noraml.cross(r2_noraml);
		
		Matrix3d Final_R ;
		Final_R<< r1_noraml(0), r2_noraml(0), r3(0),
				r1_noraml(1), r2_noraml(1), r3(1),
				r1_noraml(2), r2_noraml(2), r3(2);
						   
		Vector3d Final_t;
		Final_t	<<G2.at<double>(0,2), G2.at<double>(1,2), G2.at<double>(2,3);			//注意这个得到的相对于上一帧的
		
		R_to_ref_frame_ = Final_R;
		t_to_ref_frame_ = Final_t;
		
		Mat T_ref = (Mat_<double>(4, 4) << 	R_to_ref_frame_(0,0), R_to_ref_frame_(1,0), R_to_ref_frame_(2,0),Final_t(0),			//这个是相对于上一帧的RT阵
										R_to_ref_frame_(1,0), R_to_ref_frame_(1,1), R_to_ref_frame_(2,1),Final_t(1),
										R_to_ref_frame_(2,0), R_to_ref_frame_(1,2), R_to_ref_frame_(2,2),Final_t(2),
										0,0,0,1);
		
		if(map_->keyframes_.size() == 1)	Tcw_ = T_ref;
		else		Tcw_ = ref_->RT_Mat * T_ref;					//这个存的是当前帧相对于世界坐标系的位姿
		
		curr_->RT_Mat = Tcw_;			//ATTENTION   注意一下我求出来的是相对于上一帧的，但这里存的应该是相对于世界坐标系的
		
		Mat cv_R_matrix =  (Mat_<double>(3, 3) << 	R_to_ref_frame_(0,0), R_to_ref_frame_(1,0), R_to_ref_frame_(2,0),			//这个是相对于上一帧的RT阵
										R_to_ref_frame_(1,0), R_to_ref_frame_(1,1), R_to_ref_frame_(2,1),
										R_to_ref_frame_(2,0), R_to_ref_frame_(1,2), R_to_ref_frame_(2,2)	);
		Mat cv_t =  (	Mat_<double>(1, 3) << 	Final_t(0), Final_t(1), Final_t(2)	);			//ATTENTION  回头要debug这个是行还是列的
		Mat cv_R_vector ;
		Rodrigues(cv_R_matrix, cv_R_vector);
		

		cv_R_vector.copyTo(curr_->RT_Mat_to_ceres.rowRange(0, 3));
		cv_t.copyTo(curr_->RT_Mat_to_ceres.rowRange(3, 6));

		
		curr_->T_c_w_ = SE3(Final_R, Final_t);			//ATTENTION 注意cw是谁相对于谁！！！Debug的时候一定要注意
		T_c_w_estimated_ = curr_->T_c_w_;
		
    }



    bool VisualOdometry::checkEstimatedPose()
    {
        // check if the estimated pose is good
        if ( num_inliers_ < min_inliers_ )						//ATTENTION
        {
            cout<<"reject because inlier is too small: "<<num_inliers_<<endl;			//
            return false;
        }
        // if the motion is too large, it is probably wrong
        SE3 T_r_c = ref_->T_c_w_ * last_->T_c_w_.inverse();
        Sophus::Vector6d d = T_r_c.log();									//这表明当前帧移动过大，明显存在问题
        if ( d.norm() > 10.0 )											
        {
            cout<<"reject because motion is too large : "<<d.norm() <<endl;
            return false;
        }
        return true;
    }

    bool VisualOdometry::checkKeyFrame()			//加入关键帧的判断条件有三，旋转了一个较大的角度/移动了较大的一个长度/空间中的mappoint点数量较少
    {
      bool NeedMappoint = false;
      bool LargeMotion = false;
      if ( match_2dkp_index_.size()<100 )
            NeedMappoint = true;
      
        SE3 T_r_c = ref_->T_c_w_ * curr_->T_c_w_.inverse();
        Sophus::Vector6d d = T_r_c.log();
        Vector3d trans = d.head<3>();
        Vector3d rot = d.tail<3>();
        if ( rot.norm() >key_frame_min_rot || trans.norm() >key_frame_min_trans )	//相对于上一个关键帧旋转了很大的角度或者移动了很长的距离
            LargeMotion = true;
        return (NeedMappoint || LargeMotion);
    }

    void VisualOdometry::addKeyFrame()			//也就是说插入关键帧注定要插入mappoint
    {											//主要插入mappoint，就必须得考虑mappoint融合
        if ( map_->keyframes_.empty() )				//想要融合就必须search in the window， 就要重写match函数
        {
	      map_->insertKeyFrame ( curr_ );
	     
	      return;
        }
        map_->insertKeyFrame ( curr_ );
    }

    void VisualOdometry::addMapPointsAndKeyFrame()					//也就是说插入关键帧注定要插入mappoint
    {																//主要做的就是投影并分别对mappoint和keyframe执行融合
        Mat K ;
	K = (Mat_<double>(3, 3) << 	ref_->camera_->fx_, 0, ref_->camera_->cx_,
							0, ref_->camera_->fy_, ref_->camera_->cy_,
							0, 0, 1);

	Mat R  = (Mat_<double>(3, 3) << 	R_to_ref_frame_(0,0), R_to_ref_frame_(1,0), R_to_ref_frame_(2,0),			//这个是相对于上一帧的RT阵
										R_to_ref_frame_(1,0), R_to_ref_frame_(1,1), R_to_ref_frame_(2,1),
										R_to_ref_frame_(2,0), R_to_ref_frame_(1,2), R_to_ref_frame_(2,2)
								  );
	
	Mat t  = (Mat_<double>(3, 1) << 	t_to_ref_frame_(0), t_to_ref_frame_(1), t_to_ref_frame_(2)		);		//确定是3/1而不是1/3
	
	std::vector<Point2f> last_pt;  
	std::vector<Point2f> cur_pt;  
  
	for (size_t i = 0; i < cur_good_matches.size(); i++)  
		{  
		    last_pt.push_back(ref_->keypoints_[cur_good_matches[i].queryIdx].pt);  
		    cur_pt.push_back(curr_->keypoints_[cur_good_matches[i].trainIdx].pt);  
		}  
	
	Mat Reconstructed_points;//4行N列的矩阵，每一列代表空间中的一个点（齐次坐标）
	
	
	vector<MapPoint::Ptr> vMappoint_to_map;
	reconstruct(K , R, t, last_pt, cur_pt, Reconstructed_points);
	
	for(int i = 0; i < Reconstructed_points.cols; i++){
	    Vector3d p_world = Vector3d( Reconstructed_points.at<double>(0,i)/Reconstructed_points.at<double>(3,i) ,
								Reconstructed_points.at<double>(1,i)/Reconstructed_points.at<double>(3,i) ,
								Reconstructed_points.at<double>(2,i)/Reconstructed_points.at<double>(3,i)       );
            Vector3d n = p_world - ref_->getCamCenter();
            n.normalize();
            MapPoint::Ptr map_point = MapPoint::createMapPoint(
                    p_world, n, descriptors_curr_.row(i).clone(), curr_.get(), cur_good_matches[i].trainIdx				//这个地方不是第一帧的初始化，所以应该是curr_的keypoint id
            );
            vMappoint_to_map.push_back(map_point);
	}
	
	map_->insertKeyFrame ( curr_ );						//先加Frame再加mappoint
	map_->InsertAndFuseMapPoint( vMappoint_to_map );
	
        ref_ = curr_;					//注意只有插入完关键帧之后才能修改这个ref_
    }

    void VisualOdometry::DeleteMappoint()				//注意这个函数是用来删除mappoint点的，包含无法投影到当前帧，视角无法观测到，匹配比观测数值过小等
    {
        // remove the hardly seen and no visible points
        for ( auto iter = map_->map_points_.begin(); iter != map_->map_points_.end(); )		//点无法投影到当前帧内，应当被删除
        {
            if ( !curr_->isInFrame(iter->second->pos_) )
            {
                iter = map_->map_points_.erase(iter);
                continue;
            }
            float match_ratio = float(iter->second->matched_times_)/iter->second->visible_times_;		//能被看到却很少被匹配到，表明该点特征不好
            if ( match_ratio < map_point_erase_ratio_ )
            {
                iter = map_->map_points_.erase(iter);
                continue;
            }

            double angle = getViewAngle( curr_, iter->second );			//视角无法被当前帧观测到，应该被删除
            if ( angle > M_PI/6. )
            {
                iter = map_->map_points_.erase(iter);
                continue;
            }
            iter++;		//erase需要在这里++
        }

        cout<<"map points: "<<map_->map_points_.size()<<endl;
    }

    double VisualOdometry::getViewAngle ( Frame::Ptr frame, MapPoint::Ptr point )
    {
        Vector3d n = point->pos_ - frame->getCamCenter();
        n.normalize();
        return acos( n.transpose()*point->norm_ );
    }

    void VisualOdometry::ReadJson(){
        cout<<"Reading Json File...."<<endl;
        Json::Reader reader;
        Json::Value root;
	FreeSpacePoints_.resize(num_of_imgs_);
        ifstream in("/home/z/Desktop/ORB_SLAM2-master/Examples/Monocular/freespace.json", ios::binary);						
        if( !in.is_open() )
        {
            cout << "Error opening file\n";
        }
        if(reader.parse(in,root))  {
            int i = 0;
            int num = root["freespace_frames"].size();		//一共有多少个freespace
            for(; i < num ; i++){
                int frame_id = root["freespace_frames"][i]["frame_id"].asInt();		//这和那啥不一样
                //cout<<"current i :"<<i<<endl;
                //cout<<"current frame id :"<<frame_id<<endl;

                if(frame_id >= 2400) 	break;
                const Json::Value FSobj = root["freespace_frames"][i]["points"];
                int points_num = FSobj.size();
                for( int j  = 0; j < points_num; j++){
                    int k = 0;
                    double x = FSobj[j][k++].asDouble();
                    double y = FSobj[j][k++].asDouble();
                    double z = FSobj[j][k++].asDouble();
                    cv::Vec3d point(x,y,z);
                    //cout<<"x:"<<x<<"y:"<<y<<"z:"<<z<<endl;
                    FreeSpacePoints_[frame_id].push_back(point);
                }
            }
        }
    }
    
void VisualOdometry::reconstruct(Mat& K, Mat& R, Mat& T, vector<Point2f>& p1, vector<Point2f>& p2, Mat& structure)
{
    //两个相机的投影矩阵[R T]，triangulatePoints只支持float型
    Mat proj1(3, 4, CV_32FC1);
    Mat proj2(3, 4, CV_32FC1);

    proj1(Range(0, 3), Range(0, 3)) = Mat::eye(3, 3, CV_32FC1);
    proj1.col(3) = Mat::zeros(3, 1, CV_32FC1);

    R.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
    T.convertTo(proj2.col(3), CV_32FC1);

    Mat fK;
    K.convertTo(fK, CV_32FC1);
    proj1 = fK*proj1;
    proj2 = fK*proj2;

    //三角化重建
    triangulatePoints(proj1, proj2, p1, p2, structure);
}
    
void VisualOdometry::MakeTheInitialMap()
{
	Mat K ;
	K = (Mat_<double>(3, 3) << 	ref_->camera_->fx_, 0, ref_->camera_->cx_,
							0, ref_->camera_->fy_, ref_->camera_->cy_,
							0, 0, 1);

	Mat R  = (Mat_<double>(3, 3) << 	R_to_ref_frame_(0,0), R_to_ref_frame_(1,0), R_to_ref_frame_(2,0),			//这个是相对于上一帧的RT阵
										R_to_ref_frame_(1,0), R_to_ref_frame_(1,1), R_to_ref_frame_(2,1),
										R_to_ref_frame_(2,0), R_to_ref_frame_(1,2), R_to_ref_frame_(2,2)
								  );
	
	Mat t  = (Mat_<double>(3, 1) << 	t_to_ref_frame_(0), t_to_ref_frame_(1), t_to_ref_frame_(2)		);		//确定是3/1而不是1/3
	
	std::vector<Point2f> last_pt;  
	std::vector<Point2f> cur_pt;  
  
	for (size_t i = 0; i < cur_good_matches.size(); i++)  
		{  
		    last_pt.push_back(ref_->keypoints_[cur_good_matches[i].queryIdx].pt);  
		    cur_pt.push_back(curr_->keypoints_[cur_good_matches[i].trainIdx].pt);  
		}  
	
	Mat Reconstructed_points;//4行N列的矩阵，每一列代表空间中的一个点（齐次坐标）
	
	reconstruct(K, R, t, last_pt, cur_pt, Reconstructed_points);
	
	for(int i = 0; i < Reconstructed_points.cols; i++){
	    Vector3d p_world = Vector3d( Reconstructed_points.at<double>(0,i)/Reconstructed_points.at<double>(3,i) ,
								Reconstructed_points.at<double>(1,i)/Reconstructed_points.at<double>(3,i) ,
								Reconstructed_points.at<double>(2,i)/Reconstructed_points.at<double>(3,i)       );
            Vector3d n = p_world - ref_->getCamCenter();
            n.normalize();
            MapPoint::Ptr map_point = MapPoint::createMapPoint(
                    p_world, n, descriptors_curr_.row(i).clone(), ref_.get(), cur_good_matches[i].queryIdx			//注意这个初始地图的时候参考关键帧是ref_，和后面的是不一样的
            );						//注意描述子没必要用ref的，当前的就可以							//后面都是以cur_作为参考关键帧的
            map_->insertMapPoint( map_point );
	}
	
}

void VisualOdometry::OptimizeTheFrame()			//这个执行的对当前普通帧的BA优化
{												//具体过程就是通过与关键帧的特征点，建立一个小型的约束场
	ceres::Problem problem;
	ceres::LossFunction* loss_function = new ceres::HuberLoss(4);
	vector<MapPoint*> mappoint_to_optimize;
	Vector4d	K(ref_->camera_->fx_, ref_->camera_->fy_, ref_->camera_->cx_, ref_->camera_->cy_);
	
	
	problem.AddParameterBlock(&K(0), 4); // fx, fy, cx, cy
	problem.SetParameterBlockConstant(&K(0));				//固定内参传入

    // load intrinsic


	
	for(int i = 0 ; i < cur_good_matches.size(); i++){		//现在的问题就是取出全部需要参与优化的mappoint
	    int kp_idx = cur_good_matches[i].queryIdx;
	    mappoint_to_optimize.push_back(ref_->matched_mappoint_[kp_idx]);
	}
	for(MapPoint* mappoint : mappoint_to_optimize){				//先添加地图中全部KeyFrame
	    for(Frame* KF : mappoint->observed_frames_){
		int target_keypoint_idx = mappoint->matched_keypoint_[KF];
		KeyPoint kp = KF->keypoints_[target_keypoint_idx];
		Point2f observed = kp.pt;
		ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectCost, 2, 4, 6, 3>(new ReprojectCost(observed));

		problem.AddResidualBlock(
                cost_function,
                loss_function,  
		&K(0),
                KF->RT_Mat.ptr<double>(),
		&mappoint->pos_(0)
		);
	    }
	}
	
	for(MapPoint* mappoint : mappoint_to_optimize){					//再添加当前Frame
		int target_keypoint_idx = mappoint->matched_keypoint_[curr_.get()];
		KeyPoint kp = curr_->keypoints_[target_keypoint_idx];
		Point2f observed = kp.pt;
		ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectCost, 2, 4, 6, 3>(new ReprojectCost(observed));

		problem.AddResidualBlock(
                cost_function,
                loss_function,
		&K(0),
                &mappoint->pos_(0),  // View Rotation and Translation
                curr_->RT_Mat_to_ceres.ptr<double>()          // Point in 3D space
		);
	}
	
	ceres::Solver::Options ceres_config_options;
	ceres_config_options.minimizer_progress_to_stdout = false;
	ceres_config_options.logging_type = ceres::SILENT;
	ceres_config_options.num_threads = 1;
	ceres_config_options.preconditioner_type = ceres::JACOBI;
	ceres_config_options.linear_solver_type = ceres::SPARSE_SCHUR;
	ceres_config_options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;

	ceres::Solver::Summary summary;
	ceres::Solve(ceres_config_options, &problem, &summary);

	if (!summary.IsSolutionUsable())
	{
	    std::cout << "Bundle Adjustment failed." << std::endl;
	}
    }
    
void VisualOdometry::BundleAdjustment()
{
     ceres::Problem problem;
     
     Vector4d	K(ref_->camera_->fx_, ref_->camera_->fy_, ref_->camera_->cx_, ref_->camera_->cy_);
	
	
     problem.AddParameterBlock(&K(0), 4); // fx, fy, cx, cy
     problem.SetParameterBlockConstant(&K(0));				//固定内参传入

    // load points
    ceres::LossFunction* loss_function = new ceres::HuberLoss(4); 
    for (size_t kf_idx = 0; kf_idx < map_->keyframes_.size(); ++kf_idx)
    {
        Frame::Ptr KF = map_->keyframes_[kf_idx];
        vector<MapPoint*>& map_points = KF->map_points_;
        for (size_t point_idx = 0; point_idx < map_points.size(); ++point_idx)
        {
	    MapPoint* mp = map_points[point_idx];
	    int target_keypoint_idx = mp->matched_keypoint_[KF.get()];
	    KeyPoint kp = KF->keypoints_[target_keypoint_idx];
            Point2f observed = kp.pt;
            // 模板参数中，第一个为代价函数的类型，第二个为代价的维度，剩下三个分别为代价函数第一第二还有第三个参数的维度
            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectCost, 2, 4, 6, 3>(new ReprojectCost(observed));

            problem.AddResidualBlock(
                cost_function,
                loss_function,
		&K(0),
		KF->RT_Mat_to_ceres.ptr<double>(),
                &mp->pos_(0)  
            );
        }
    }

    // Solve BA
    ceres::Solver::Options ceres_config_options;
    ceres_config_options.minimizer_progress_to_stdout = false;
    ceres_config_options.logging_type = ceres::SILENT;
    ceres_config_options.num_threads = 1;
    ceres_config_options.preconditioner_type = ceres::JACOBI;
    ceres_config_options.linear_solver_type = ceres::SPARSE_SCHUR;
    ceres_config_options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;

    ceres::Solver::Summary summary;
    ceres::Solve(ceres_config_options, &problem, &summary);

    if (!summary.IsSolutionUsable())
    {
        std::cout << "Bundle Adjustment failed." << std::endl;
    }
}
    
}


