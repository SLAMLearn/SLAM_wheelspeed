#include "myslam/map.h"

namespace myslam
{

void Map::insertKeyFrame ( Frame::Ptr frame )				//现在关键帧和mappoint点之间需要建立双向连接了，因此插入KF的过程
{													//需要将关键和其他所有的mappoint建立联系
    
    cout<<"Key frame size = "<<keyframes_.size()<<endl;		//现在也不需要记录当前frame是局部地图中的第几个点了
    keyframes_[ frame->id_ ] = frame;
    
    cout<<"Key frame size = "<<keyframes_.size()<<endl;

    if(keyframes_.size() == 6)	    DeleteFirstKeyFrame();			//加入当前帧超过上限后应该把局部地图中的第一帧删掉
    
    double project_radius = 10;
    for(auto mappoint : map_points_){				//第一步：遍历map下的所有局部地图点
	Vector3d pos_of_mappoint = mappoint.second->pos_;										//向该帧下投影
	Mat Homogeneous_of_mappoint = (Mat_<double>(4, 1) << pos_of_mappoint(0), pos_of_mappoint(1), pos_of_mappoint(2), 1);	//齐次坐标
																					//ATTENTION 确定一下是4,1	不是1,4
	Mat K = (Mat_<double>(3, 3) << 	frame->camera_->fx_, 0, frame->camera_->cx_,
						      0, frame->camera_->fy_, frame->camera_->cy_,
						      0, 0, 1);
	Mat	project_pos = K * frame->RT_Mat * Homogeneous_of_mappoint;			//投影
	
	vector<int> vkp_idx;		//这个地方存keypoint是没有用的，得存的是keypoint对应于当前帧下的第几个特征点
	Mat desp_map;
	for(int i = 0 ; i < frame->keypoints_.size(); i++){	//其次遍历改关键帧下的每一个特征点，取出在投影区域内的特征点
	    KeyPoint kp = frame->keypoints_[i];
	    if(kp.pt.x < project_pos.at<double>(0,0) + project_radius &&
	       kp.pt.x > project_pos.at<double>(0,0) - project_radius &&
	       kp.pt.y < project_pos.at<double>(1,0) + project_radius &&
	       kp.pt.x > project_pos.at<double>(1,0) - project_radius			//该特征点在投影区域内
	    ){
		  vkp_idx.push_back(i);
		  desp_map.push_back(frame->descriptors_);			//存储特征点和描述子
	    }
	}
	vector<cv::DMatch> matches;
	Mat mappoint_desp_mat;			//这里选择把当前描述子复制三次，如果匹配最佳的远好于匹配次佳的，那么就接受		会极大地增加匹配时间，存疑
	for(int k = 0; k < 1; k++)				//ATTENTION			TODO
	    mappoint_desp_mat.push_back(mappoint.second->descriptor_);
	
	matcher_flann_.match ( mappoint_desp_mat, desp_map, matches );			//这种匹配方式还是得看看。。感觉一个点一个点匹配不靠谱
																				//ATTENTION 不行的话需要手动计算两个descriptor之间的距离
	if(matches.empty())	continue;
	
	sort(matches.begin(), matches.end());
	//if(matches[0].distance < 0.05 ||    matches[0].distance * 0.7 < matches[1])
	if(matches[0].distance > 0.1)	continue;
	int target_keypoint = vkp_idx[matches[0].queryIdx];			//找到对应当前帧下的第几个keypoint
														//我加入的流程是先加帧，后加点，因此此时不考虑当前关键帧和上一个参考关键帧
														//对于我现在找到的这个keypoint的影响，那些放到mappoint里面去考虑
	//也就是说我现在找到的这个特征点一定没有对应的空间mappoint，就不需要判断，直接连接就可以
	frame->observed_mappoint_.push_back(mappoint.second.get());
	frame->matched_mappoint_[target_keypoint] = mappoint.second.get();
    }
}

void Map::insertMapPoint ( MapPoint::Ptr map_point )			
{														
    map_points_[map_point->id_] = map_point;
}

void Map::DeleteFirstKeyFrame()		//下面那个是删除新进来的Frame，但这里需要去删除map中最早建立的那一帧，模拟FIFO
{
    map<unsigned long, Frame::Ptr >::iterator it = keyframes_.begin();
    DeleteKeyFrame((*it).second);
}

void Map::DeleteKeyFrame( Frame::Ptr frame){				//和插入frame一样，为了建立优化函数需要动态更新frame的id
  for(auto mp : map_points_){
    if(mp.second->matched_keypoint_[frame.get()]){				//相比于mappoint这个倒是好删
      mp.second->matched_keypoint_.erase(frame.get());			//删除mappoint对该帧的观测
      for(auto it = mp.second->observed_frames_.begin(); it != mp.second->observed_frames_.end(); it++){
	    if(*it == frame.get()){
		mp.second->observed_frames_.erase(it);
	    }
      }
    }
    if(mp.second->observed_frames_.empty())			//当前点一个KF都观测不到的话就删除了吧
      DeleteMappoint(mp.second);
  }
  keyframes_.erase(frame.get()->id_);
  
}

void Map::DeleteMappoint( MapPoint::Ptr map_point){		//图中只记录了mappoint到keyframe的单向指针，所以mappoint的删除相对较为容易，直接删就可以了
      for(Frame* KF : map_point->observed_frames_){		//遍历能看到该mappoint的所有关键帧
	  vector<MapPoint*>::iterator iter = find(KF->observed_mappoint_.begin(),KF->observed_mappoint_.end(),map_point.get());		//删除的是对应Frame向该mappoint的连接
	  if(iter==KF->observed_mappoint_.end())
	      iter = KF->observed_mappoint_.erase(iter);
	  for(map<int, MapPoint *>::iterator it=KF->matched_mappoint_.begin(); it!=KF->matched_mappoint_.end(); it++){
	      if( it->second == map_point.get() ){
		  KF->matched_mappoint_.erase(it);
		  break;							//同一帧不会有多个keypoint同时对应一个mappoint
	      }
	  }
      map_points_.erase(map_point.get()->id_);
      }
}

void Map::InsertAndFuseMapPoint(vector<MapPoint::Ptr > vMappoint_to_map)
{
    //Mat desp_map;
    for(MapPoint::Ptr map_point : vMappoint_to_map){			//注意一开始不能直接把点加进图中，应该先判断该点是否已经在图中了
	
	//desp_map.push_back(map_point->descriptor_);			//特征点存成一个矩阵来存储，相对于search by projection，待定方法
   
  
    
    double project_radius = 10;	//TODO 单位为pixel， 可以回头集成到map接口上
    for(auto KF : keyframes_){					//TODO 首先遍历map下的每一个关键帧
	if(KF.second.get() == map_point->ref_KF_1_)	continue;	//ATTENTION  注意一下这个判断是否能够成立
	Vector3d pos_of_mappoint = map_point->pos_;										//向该帧下投影
	Mat Homogeneous_of_mappoint = (Mat_<double>(4, 1) << pos_of_mappoint(0), pos_of_mappoint(1), pos_of_mappoint(2), 1);	//齐次坐标
																					//ATTENTION 确定一下是4,1	不是1,4
	Mat K = (Mat_<double>(3, 3) << 	KF.second->camera_->fx_, 0, KF.second->camera_->cx_,
						      0, KF.second->camera_->fy_, KF.second->camera_->cy_,
						      0, 0, 1);
	Mat	project_pos = K * KF.second->RT_Mat * Homogeneous_of_mappoint;			//投影
	
	vector<KeyPoint> vkp;
	Mat desp_map;
	for(int i = 0 ; i < KF.second->keypoints_.size(); i++){	//其次遍历改关键帧下的每一个特征点，取出在投影区域内的特征点
	    KeyPoint kp = KF.second->keypoints_[i];
	    if(kp.pt.x < project_pos.at<double>(0,0) + project_radius &&
	       kp.pt.x > project_pos.at<double>(0,0) - project_radius &&
	       kp.pt.y < project_pos.at<double>(1,0) + project_radius &&
	       kp.pt.y > project_pos.at<double>(1,0) - project_radius			//该特征点在投影区域内
	    ){
		  vkp.push_back(kp);
		  desp_map.push_back(KF.second->descriptors_);			//存储特征点和描述子
	    }
	}
	
	vector<cv::DMatch> matches;
	matcher_flann_.match ( map_point.get()->descriptor_, desp_map, matches );
	if(matches.empty())	continue;
	if(matches[0].distance > 0.1	)		continue;		//现在就是只能用这种最原始的方式了。。但好歹是有个判断依据			可以描述子复制三次	TODO
	
	int target_idx = matches[0].trainIdx;
	if(KF.second->matched_mappoint_.count(target_idx)){		//如果当前对应的keypoint 在局部地图中已经有了mappoint
		MapPoint* mp =  KF.second->matched_mappoint_[target_idx];
		Frame* fm = map_point->ref_KF_1_;				//那么为该点连接一个连接帧，为当前加入点的参考关键帧
		
		mp->observed_frames_.push_back(fm);				//然后其他帧也不需要再考虑了，当前mappoint已经处理完毕了	ATTENTION  再注意一下思路对不对
		mp->matched_keypoint_[fm] = mp->ref_KF_id_;
		
		fm->observed_mappoint_.push_back(mp);
		fm->matched_mappoint_[mp->ref_KF_id_] = mp;
		break;		//当前mappoint已经处理完毕了
	}
	else{				//对应的keypoint在局部地图中并没有mappoint对应，那么把当前mappoint加入到局部地图，只考虑当前两帧！！！（简化的办法了已经是）
		    map_points_[map_point->id_] = map_point;
																	 //第二步是为点连接
		    Frame* fm1 = map_point->ref_KF_1_;
		    Frame* fm2 = KF.second.get();
		    map_point->observed_frames_.push_back(fm1);				
		    map_point->matched_keypoint_[fm1] = map_point->ref_KF_id_;
		
		    fm1->observed_mappoint_.push_back(map_point.get());
		    fm1->matched_mappoint_[map_point->ref_KF_id_] = map_point.get();
		    
		    map_point->observed_frames_.push_back(fm2);				
		    map_point->matched_keypoint_[fm2] = map_point->ref_KF_id_;
		
		    fm2->observed_mappoint_.push_back(map_point.get());
		    fm2->matched_mappoint_[map_point->ref_KF_id_] = map_point.get();
		    break;		//当前mappoint已经处理完毕了
	}
    }
  }

}	//end of InsertAndFuseMapPoint


}	//end of namespace
