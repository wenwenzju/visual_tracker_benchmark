#include "tracking_using_re_id_and_pf.h"
#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"
#include "opencv2/opencv.hpp"
#include "timer.h"

#include <string>
#include <fstream>
#include <iomanip>

//#define SAVE_IMAGE

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
		("inith", value<int>(), "initial height")
		("rw_ratio", value<double>()->default_value(0.5), "random walk ratio")
		("binL", value<int>()->default_value(16), "bins of histogram of channel L")
		("binU", value<int>()->default_value(16), "bins of histogram of channel U")
		("binV", value<int>()->default_value(4), "bins of histogram of channel V")
		("binHOG", value<int>()->default_value(6), "bins of histogram of hog")
		("init_probe", value<int>()->default_value(5), "initial probes needed")
		("cur_probe", value<int>()->default_value(3), "near current probes needed");
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
	double rwr;
	int binL, binU, binV, binHOG, init_probe, cur_probe;
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
		if (vm.count("rw_ratio"))
			rwr = vm["rw_ratio"].as<double>();
		if (vm.count("binL"))
			binL = vm["binL"].as<int>();
		if (vm.count("binU"))
			binU = vm["binU"].as<int>();
		if (vm.count("binV"))
			binV = vm["binV"].as<int>();
		if (vm.count("binHOG"))
			binHOG = vm["binHOG"].as<int>();
		if (vm.count("init_probe"))
			init_probe = vm["init_probe"].as<int>();
		if (vm.count("cur_probe"))
			cur_probe = vm["cur_probe"].as<int>();

		ofstream result_file(seq_name+"_PF.txt");
		result_file << initx << "	" << inity << "	" << initw << "	" << inith << endl;
		ofstream fps_file(seq_name+"_PF_FPS.txt");
		PTUsingReIdandPF tracker(5, particles, use_optical_flow_lk, update_online, learning_rate, sigmax, sigmay, sigmaw, exp_coeff_re_id, re_id_weight, hog_particle_expand, thread_num, aspect_ratio, rwr, binL, binU, binV, binHOG, init_probe, cur_probe);

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
			char k = cv::waitKey(2);
			if (k=='q') return 0;
			if (k=='p') cv::waitKey(0);
#ifdef SAVE_IMAGE
			stringstream ss1;
			ss1 << setfill('0') << setw(nz) << i;
			ss1 << "." << ext;
			string in; ss1 >> in;
			imwrite(in, image);
#endif

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


//#include "acf_feature_extractor.h"
//#include <string>
//#include "timer.h"
//
//using namespace cv;
//using namespace std;
//int main(int argc, char* argv[])
//{
//	string img_name("D:\\data_seq\\Human7\\0001.jpg");
//	Mat img = imread(img_name);
//	AcfFeatureExtractor afe;
//	vector<Mat> features;
//	cv::Mat fm;
//	tic
//		afe.feature_extract(img, cv::Rect(0,0,img.cols, img.rows), 10, 10, 10, 8, fm);
//	toc
//	tic
//	afe.feature_extract(img, features);
//	toc
//
//	Mat L,U,V;
//	features[0].convertTo(L, CV_8UC1, 255./0.37);
//	imwrite("colorL.jpg", L);
//	features[1].convertTo(U, CV_8UC1, 255);
//	imwrite("colorU.jpg", U);
//	features[2].convertTo(V, CV_8UC1, 255./0.89);
//	imwrite("colorV.jpg", V);
//	Mat gradmag, gradmags;
//	cv::normalize(features[3], gradmag, 0, 255, NORM_MINMAX);
//	gradmag.convertTo(gradmags, CV_8UC1);
//	imwrite("gradmag.jpg", gradmags);
//
//	Mat orient1,orient2, orient3, orient4, orient5, orient6;
//	Mat orient1s,orient2s, orient3s, orient4s, orient5s, orient6s;
//	cv::normalize(features[4], orient1, 0, 255, NORM_MINMAX);
//	orient1.convertTo(orient1s, CV_8UC1);
//	imwrite("orient1.jpg", orient1s);
//
//	cv::normalize(features[5], orient2, 0, 255, NORM_MINMAX);
//	orient2.convertTo(orient2s, CV_8UC1);
//	imwrite("orient2.jpg", orient2s);
//
//	cv::normalize(features[6], orient3, 0, 255, NORM_MINMAX);
//	orient3.convertTo(orient3s, CV_8UC1);
//	imwrite("orient3.jpg", orient3s);
//
//	cv::normalize(features[7], orient4, 0, 255, NORM_MINMAX);
//	orient4.convertTo(orient4s, CV_8UC1);
//	imwrite("orient4.jpg", orient4s);
//
//	cv::normalize(features[8], orient5, 0, 255, NORM_MINMAX);
//	orient5.convertTo(orient5s, CV_8UC1);
//	imwrite("orient5.jpg", orient5s);
//
//	cv::normalize(features[9], orient6, 0, 255, NORM_MINMAX);
//	orient6.convertTo(orient6s, CV_8UC1);
//	imwrite("orient6.jpg", orient6s);
//
//	waitKey(0);
//}