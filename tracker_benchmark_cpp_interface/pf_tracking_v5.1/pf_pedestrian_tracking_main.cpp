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
		("inith", value<int>(), "initial height")
		("rw_ratio", value<double>()->default_value(0.5), "random walk ratio")
		("probe_need", value<int>()->default_value(5), "probe need");
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
	int pn;
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
		if (vm.count("probe_need"))
			pn = vm["probe_need"].as<int>();

		ofstream result_file(seq_name+"_PF.txt");
		result_file << initx << "	" << inity << "	" << initw << "	" << inith << endl;
		ofstream fps_file(seq_name+"_PF_FPS.txt");
		PTUsingReIdandPF tracker(3, particles, use_optical_flow_lk, update_online, learning_rate, sigmax, sigmay, sigmaw, exp_coeff_re_id, re_id_weight, hog_particle_expand, thread_num, aspect_ratio, rwr, pn);

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

//#include "sdalf_re_id.h"
//#include <fstream>
//#include "opencv2/opencv.hpp"
//#include <vector>
//
//using namespace std;
//using namespace cv;
//
//int main()
//{
//	Mat img = imread("D:\\data_seq\\Walking2\\0046.jpg");
//	Rect r(115,112,46,107);
//	Mat showimg;
//	img.copyTo(showimg);
//	rectangle(showimg, r, Scalar(0,0,255),3);
//	imshow("image", showimg);
//	waitKey(0);
//
//	pe_re_id::SdalfPeReId sdalf;
//	std::vector<cv::Mat> g(1);
//	cv::resize(img(r), g[0], cv::Size(64, 128));
//	std::vector<pe_re_id::SdalfFeature> gallery_feature;
//	sdalf.gallery_feautre_extract(g, gallery_feature);
//
//	ofstream template_features("data\\template.txt");
//	sdalf.save_feature(template_features, gallery_feature[0]);
//}