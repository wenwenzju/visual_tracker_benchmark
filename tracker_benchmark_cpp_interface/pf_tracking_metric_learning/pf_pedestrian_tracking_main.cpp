#include "tracking_using_re_id_and_pf.h"
#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"
#include "opencv2/opencv.hpp"
#include "timer.h"

#include <string>
#include <fstream>
#include <iomanip>

using namespace boost::program_options;
using namespace std;

int main(int argc, char** argv)
{
	using namespace cv;
	options_description desc("tracking using re-identification and particle filter");
	desc.add_options()
		("help,h", "produce help message")
		("particles,p", value<int>()->default_value(52), "particles number")
		("optical_flow,o", value<bool>()->default_value(true), "use optical flow lk")
		("update_online,u", value<bool>()->default_value(true), "re-id model update online")
		("learning_rate,l", value<double>()->default_value(0.4), "re-id model learning rate")
		("sigma_x,x", value<double>()->default_value(20), "sigma of x")
		("sigma_y,y", value<double>()->default_value(10), "sigma of y")
		("sigma_w,s", value<double>()->default_value(8), "sigma of width")
		("exp_coeff_re_id,e", value<double>()->default_value(0.5), "exp coefficient of re-id distance")
		("re-id_weight,w", value<double>()->default_value(0.8), "weight of re-id in observation model")
		("hog_particle_expand,d", value<double>()->default_value(1.4), "hog particle expand")
		("thread_num,t", value<int>()->default_value(4), "thread number")
		("aspect_ratio,r", value<double>()->default_value(0.43), "aspect ratio")

		("seq_path", value<string>(), "sequence path")
		("seq_name", value<string>(), "sequence name")
		("start_frame", value<int>()->default_value(1), "start frame number")
		("end_frame", value<int>(), "end frame number")
		("nz", value<int>()->default_value(4), "number of zeros")
		("ext", value<string>()->default_value("jpg"), "image format")
		("initx", value<int>(), "initial x")
		("inity", value<int>(), "initial y")
		("initw", value<int>(), "initial width")
		("inith", value<int>(), "initial height");
	variables_map vm;
	int particles;
	bool use_optical_flow_lk;
	bool update_online;
	double learning_rate;
	double sigmax;
	double sigmay;
	double sigmaw;
	double exp_coeff_re_id;
	double re_id_weight;
	double hog_particle_expand;
	int thread_num;
	double aspect_ratio;

	string seq_path, seq_name;
	int start_frame, end_frame, nz;
	string ext;
	int initx, inity, initw, inith;
	try
	{
		store(parse_command_line(argc, argv, desc), vm);
		notify(vm);
		if (vm.count("help"))
		{
			cout << desc << endl;
			return 1;
		}
		if (vm.count("particles"))
			particles = vm["particles"].as<int>();
		if (vm.count("optical_flow"))
			use_optical_flow_lk = vm["optical_flow"].as<bool>();
		if (vm.count("update_online"))
			update_online = vm["update_online"].as<bool>();
		if (vm.count("learning_rate"))
			learning_rate = vm["learning_rate"].as<double>();
		if (vm.count("sigma_x"))
			sigmax = vm["sigma_x"].as<double>();
		if (vm.count("sigma_y"))
			sigmay = vm["sigma_y"].as<double>();
		if (vm.count("sigma_w"))
			sigmaw = vm["sigma_w"].as<double>();
		if (vm.count("exp_coeff_re_id"))
			exp_coeff_re_id = vm["exp_coeff_re_id"].as<double>();
		if (vm.count("re-id_weight"))
			re_id_weight = vm["re-id_weight"].as<double>();
		if (vm.count("hog_particle_expand"))
			hog_particle_expand = vm["hog_particle_expand"].as<double>();
		if (vm.count("thread_num"))
			thread_num = vm["thread_num"].as<int>();
		if (vm.count("aspect_ratio"))
			aspect_ratio = vm["aspect_ratio"].as<double>();
		if (vm.count("seq_path")){
			seq_path = vm["seq_path"].as<string>();
			if (seq_path[seq_path.size()-1] != '\\' && seq_path[seq_path.size()-1] != '/')
				seq_path += "/";
		}
		if (vm.count("seq_name"))
			seq_name = vm["seq_name"].as<string>();
		if (vm.count("start_frame"))
			start_frame = vm["start_frame"].as<int>();
		if (vm.count("end_frame"))
			end_frame = vm["end_frame"].as<int>();
		if (vm.count("nz"))
			nz = vm["nz"].as<int>();
		if (vm.count("ext"))
			ext = vm["ext"].as<string>();
		if (vm.count("initx"))
			initx = vm["initx"].as<int>();
		if (vm.count("inity"))
			inity = vm["inity"].as<int>();
		if (vm.count("initw"))
			initw = vm["initw"].as<int>();
		if (vm.count("inith"))
			inith = vm["inith"].as<int>();

		ofstream result_file(seq_name+"_PF.txt");
		result_file << initx << "	" << inity << "	" << initw << "	" << inith << endl;
		ofstream fps_file(seq_name+"_PF_FPS.txt");
		PTUsingReIdandPF tracker(3, particles, use_optical_flow_lk, update_online, learning_rate, sigmax, sigmay, sigmaw, exp_coeff_re_id, re_id_weight, hog_particle_expand, thread_num, aspect_ratio);

		cv::Rect initialization(initx, inity, initw, inith);
		cv::Mat image;

		//read first image
		stringstream ss;
		ss << seq_path;
		ss << setfill('0') << setw(nz) << start_frame;
		ss << "." << ext;
		string imgname;
		ss >> imgname;
		image = imread(imgname);

		tic
		tracker.init(image, initialization);

		for (int i = start_frame+1; i <= end_frame; ++i) {

			stringstream ss;
			ss << seq_path;
			ss << setfill('0') << setw(nz) << i;
			ss << "." << ext;
			string imgname;
			ss >> imgname;

			cv::Mat image = cv::imread(imgname);

			cv::Rect rect = tracker.track(image);
			result_file << rect.x << "	" << rect.y << "	" << rect.width << "	" << rect.height << endl;

			cv::rectangle(image, rect, cv::Scalar(0, 0, 255), 2);
			cv::imshow("result", image);
			cv::waitKey(10);
			cv::imwrite(imgname.substr(imgname.size()-8), image);
		}
		double tot_time = toc;
		cout << "FPS: " << 1.*(end_frame - start_frame + 1) / tot_time << endl;
		fps_file << 1.*(end_frame - start_frame + 1) / tot_time;
	}
	catch (invalid_command_line_syntax&)
	{
		cout << "syntax error" << endl;
		cout << desc << endl;
	}
	catch (boost::bad_lexical_cast&)
	{
		cout << "lexical error" << endl;
		cout << desc << endl;
	}
	catch (...)
	{
		cout << "Error caught!" << endl;
	}


}

//#include "tracking_using_re_id_and_pf.h"
//#include "boost/program_options.hpp"
//#include "boost/filesystem.hpp"
//#include "opencv2/opencv.hpp"
//#include "timer.h"
//
//#include <string>
//#include <fstream>
//
////command line
////1. -v E:\0wenwen\HOG_OpticalFLow_Tracking\single_person_xihu2\single_person_xihu2.avi -t single_person_xihu2
////2. -v E:\0wenwen\HOG_OpticalFLow_Tracking\EnterExitCrossingPaths1cor.avi -t EnterExitCrossingPaths1cor
////3. -v E:\0wenwen\HOG_OpticalFLow_Tracking\ETH_pedestrian_dataset.avi -t ETH_pedestrian_dataset
////4. -v E:\0wenwen\HOG_OpticalFLow_Tracking\Lab_ldx\Lab_ldx.avi -t Lab_ldx
////5. -v E:\0wenwen\HOG_OpticalFLow_Tracking\ETH_pedestrian_white_jacket_woman.avi -t ETH_pedestrian_white_jacket_woman
////6. -v E:\0wenwen\dataset\visual_tracker_benchmark\Human9\img\visual_tracker_benchmark_human9.avi -t visual_tracker_benchmark_human9
////7. -v E:\0wenwen\dataset\visual_tracker_benchmark\Human2\img\visual_tracker_benchmark_human2.avi -t visual_tracker_benchmark_human2
////8. -v E:\0wenwen\dataset\visual_tracker_benchmark\Human7\img\visual_tracker_benchmark_human7.avi -t visual_tracker_benchmark_human7
////9. -v E:\0wenwen\dataset\visual_tracker_benchmark\Human8\img\visual_tracker_benchmark_human8.avi -t visual_tracker_benchmark_human8
////10. -t wangwen_beijing -v E:\0wenwen\HOG_OpticalFLow_Tracking\201_beijing_raw_img_left_0.avi -i wangwen_beijing_0.8_init_pos.txt
//
///*visual tracker benchmark*/
///*parameter [learning_rate, re_id_weight, sigma_x, sigma_y, sigma_w]*/
////1.-s E:\0wenwen\dataset\visual_tracker_benchmark\BlurBody ------------------------good
////2.-s E:\0wenwen\dataset\visual_tracker_benchmark\Bolt ----------------------------bad
////3.-s E:\0wenwen\dataset\visual_tracker_benchmark\Bolt2 ---------------------------bad
////4.-s E:\0wenwen\dataset\visual_tracker_benchmark\Couple --------------------------bad
////5.-s E:\0wenwen\dataset\visual_tracker_benchmark\Crossing ------------------------moderate
////6.-s E:\0wenwen\dataset\visual_tracker_benchmark\David3 --------------------------good
////7.-s E:\0wenwen\dataset\visual_tracker_benchmark\Girl2 ---------------------------bad
////8.-s E:\0wenwen\dataset\visual_tracker_benchmark\Gym -----------------------------good [0.5, 0.8, 20, 10, 5]
////9.-s E:\0wenwen\dataset\visual_tracker_benchmark\Human2 --------------------------good
////10.-s E:\0wenwen\dataset\visual_tracker_benchmark\Human3 -------------------------bad
////11.-s E:\0wenwen\dataset\visual_tracker_benchmark\Human4_1 -----------------------error
////12.-s E:\0wenwen\dataset\visual_tracker_benchmark\Human5 -------------------------bad
////13.-s E:\0wenwen\dataset\visual_tracker_benchmark\Human6 -------------------------bad
////15.-s E:\0wenwen\dataset\visual_tracker_benchmark\Human7 -------------------------moderate
////16.-s E:\0wenwen\dataset\visual_tracker_benchmark\Human8 -------------------------good
////17.-s E:\0wenwen\dataset\visual_tracker_benchmark\Human9 -------------------------bad
////18.-s E:\0wenwen\dataset\visual_tracker_benchmark\Jogging_1 ----------------------bad
////19.-s E:\0wenwen\dataset\visual_tracker_benchmark\Singer1 ------------------------moderate
////20.-s E:\0wenwen\dataset\visual_tracker_benchmark\Skater -------------------------good
////21.-s E:\0wenwen\dataset\visual_tracker_benchmark\Skater2 ------------------------moderate
////22.-s E:\0wenwen\dataset\visual_tracker_benchmark\Skating1 -----------------------bad
////23.-s E:\0wenwen\dataset\visual_tracker_benchmark\Skating2_1 ---------------------moderate
////24.-s E:\0wenwen\dataset\visual_tracker_benchmark\Subway -------------------------bad
////25.-s E:\0wenwen\dataset\visual_tracker_benchmark\Walking ------------------------moderate
////26.-s E:\0wenwen\dataset\visual_tracker_benchmark\Walking2 -----------------------bad
////27.-s E:\0wenwen\dataset\visual_tracker_benchmark\Woman --------------------------bad
////28.-s E:\0wenwen\dataset\visual_tracker_benchmark\Human4_2 -----------------------bad
////29.-s E:\0wenwen\dataset\visual_tracker_benchmark\Jogging_2 ----------------------bad
////30.-s E:\0wenwen\dataset\visual_tracker_benchmark\Skating2_2 ---------------------bad
///*visual tracker benchmark*/
//
//using namespace boost::program_options;
//using namespace std;
//using namespace cv;
//
//cv::Point pre_pt(-1,-1);
//cv::Point cur_pt(-1,-1);
//
//void on_mouse(int event, int x, int y, int flags, void* utstc = NULL)
//{
//	using namespace cv;
//	if (event == CV_EVENT_LBUTTONDOWN)
//	{
//		pre_pt = Point(x,y);
//	}
//	else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON))
//	{
//		cur_pt = Point(x,y);
//	}
//}
//
//int main(int argc, char** argv)
//{
//	options_description desc("tracking using re-identification and particle filter");
//	desc.add_options()
//		("help,h", "produce help message")
//		("tar_des,t", value<string>(), "target description")
//		("video,v", value<string>(), "path of video file")
//		("init_pos,i", value<string>(), "path of initialize position file")
//		("sequence,s", value<string>(), "path of sequences of visual tracker benchmark");
//	variables_map vm;
//	string tar_des, video_file, init_pos;
//	Rect init_r;
//	bool use_images;
//	vector<boost::filesystem::path> vp;
//	try
//	{
//		store(parse_command_line(argc, argv, desc), vm);
//		notify(vm);
//		if (vm.count("help"))
//		{
//			cout << desc << endl;
//			return 1;
//		}
//		if (vm.count("sequence"))
//		{
//			use_images = true;
//			string sequence_path = vm["sequence"].as<string>();
//			tar_des = sequence_path.substr(sequence_path.find_last_of("/\\")+1);
//			ifstream groundtruth(sequence_path+"/groundtruth_rect.txt");
//			char tmp;
//			groundtruth >> init_r.x;
//			groundtruth.get(tmp);
//			groundtruth >> init_r.y;
//			groundtruth.get(tmp);
//			groundtruth >> init_r.width;
//			groundtruth.get(tmp);
//			groundtruth >> init_r.height;
//			groundtruth.get(tmp);
//			
//			//groundtruth >> init_r.x >> tmp >> init_r.y >> tmp >> init_r.width >> tmp >> init_r.height >> tmp;
//
//			boost::filesystem::path img_p(sequence_path+"/img");
//			if (!is_directory(img_p)) {cerr << "image path " << endl << sequence_path+"/img" << endl << "doesn't exist" << endl;return 1;}
//			
//			copy(boost::filesystem::directory_iterator(img_p), boost::filesystem::directory_iterator(), back_inserter(vp));
//			sort(vp.begin(), vp.end());
//		}
//		else
//		{
//			use_images = false;
//			if (vm.count("tar_des"))
//			{
//				tar_des = vm["tar_des"].as<string>();
//			}
//			else
//			{
//				cout << "target description needed" << endl;
//				cout << desc << endl;
//				return 1;
//			}
//			if (vm.count("video"))
//			{
//				video_file = vm["video"].as<string>();
//			}
//			else
//			{
//				cout << "video file needed" << endl;
//				cout << desc << endl;
//				return 1;
//			}
//			if (vm.count("init_pos"))
//			{
//				init_pos = vm["init_pos"].as<string>();
//			}
//		}
//		{
//			using namespace cv;
//			using namespace std;
//			
//			VideoCapture cap;
//			if (!use_images) cap.open(video_file);
//			Mat frame;
//			size_t vp_cnt = 0;
//			if (use_images) 
//			{
//				while (".jpg" != vp[vp_cnt].extension() && ".png" != vp[vp_cnt].extension()) vp_cnt++;
//				if (".jpg" == vp[vp_cnt].extension() || ".png" == vp[vp_cnt].extension())
//					frame = imread(vp[vp_cnt++].string());
//			}
//			else
//				cap >> frame;
//			PTUsingReIdandPF pt;
//
//			if (!use_images)
//			{
//				if (init_pos.size() == 0)
//				{
//					namedWindow("initialize");
//					setMouseCallback("initialize", on_mouse);
//					imshow("initialize", frame);
//					Mat tmp; frame.copyTo(tmp);
//					char k = waitKey(2);
//					while (k != 'q' && k != ' ')
//					{
//						if (k == 'c') init_r = Rect();
//						frame.copyTo(tmp);
//						init_r = Rect(pre_pt, cur_pt);
//						rectangle(tmp, init_r, Scalar(0,0,255));
//						imshow("initialize", tmp);
//						k = waitKey(2);
//					}
//				}
//				else
//				{
//					ifstream pos_file(init_pos);
//					pos_file >> init_r.x >> init_r.y >> init_r.width >> init_r.height;
//				}
//			}
//			pt.init(frame, init_r);
//
//			rectangle(frame, init_r, Scalar(0,255,255));
//			imshow("result", frame);
//			char k = waitKey(5);
//			int cnt = 0;
//			char img_name[10];
//			while(k != 'q')
//			{
//				if (use_images)
//				{
//					if (vp_cnt >= vp.size()) break;
//					if (".jpg" == vp[vp_cnt].extension() || ".png" == vp[vp_cnt].extension())
//						frame = imread(vp[vp_cnt++].string());
//					else
//					{
//						vp_cnt++;
//						continue;
//					}
//				}
//				else
//				{
//					if (!cap.read(frame))
//						break;
//				}
//				Rect r;
//				std::vector<cv::Rect> particles;
//				std::vector<cv::Rect> peds;
//				tic
//				r = pt.track(frame);
//				toc
//
//				rectangle(frame, r, Scalar(0,0,255), 4);
//				imshow("result", frame);
//				k = waitKey(2);
//			}
//		}
//	}
//	catch (invalid_command_line_syntax&)
//	{
//		cout << "syntax error" << endl;
//		cout << desc << endl;
//	}
//	catch (boost::bad_lexical_cast&)
//	{
//		cout << "lexical error" << endl;
//		cout << desc << endl;
//	}
//	catch (...)
//	{
//		cout << "Error caught!" << endl;
//	}
//
//
//}