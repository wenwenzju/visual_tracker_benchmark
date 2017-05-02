//#include "tracking_using_re_id_and_pf.h"
//#include "boost/program_options.hpp"
//#include "boost/filesystem.hpp"
//#include "opencv2/opencv.hpp"
//#include "timer.h"
//
//#include <string>
//#include <fstream>
//#include <iomanip>
//
//using namespace boost::program_options;
//using namespace std;
//
//int main(int argc, char** argv)
//{
//	using namespace cv;
//	options_description desc("tracking using re-identification and particle filter");
//	desc.add_options()
//		("help,h", "produce help message")
//		("particles,p", value<int>()->default_value(52), "particles number")
//		("optical_flow,o", value<bool>()->default_value(true), "use optical flow lk")
//		("update_online,u", value<bool>()->default_value(true), "re-id model update online")
//		("learning_rate,l", value<double>()->default_value(0.4), "re-id model learning rate")
//		("sigma_x,x", value<double>()->default_value(20), "sigma of x")
//		("sigma_y,y", value<double>()->default_value(10), "sigma of y")
//		("sigma_w,s", value<double>()->default_value(8), "sigma of width")
//		("exp_coeff_re_id,e", value<double>()->default_value(0.5), "exp coefficient of re-id distance")
//		("re-id_weight,w", value<double>()->default_value(0.8), "weight of re-id in observation model")
//		("hog_particle_expand,d", value<double>()->default_value(1.4), "hog particle expand")
//		("thread_num,t", value<int>()->default_value(4), "thread number")
//		("aspect_ratio,r", value<double>()->default_value(0.43), "aspect ratio")
//
//		("seq_path", value<string>(), "sequence path")
//		("seq_name", value<string>(), "sequence name")
//		("start_frame", value<int>()->default_value(1), "start frame number")
//		("end_frame", value<int>(), "end frame number")
//		("nz", value<int>()->default_value(4), "number of zeros")
//		("ext", value<string>()->default_value("jpg"), "image format")
//		("initx", value<int>(), "initial x")
//		("inity", value<int>(), "initial y")
//		("initw", value<int>(), "initial width")
//		("inith", value<int>(), "initial height");
//	variables_map vm;
//	int particles;
//	bool use_optical_flow_lk;
//	bool update_online;
//	double learning_rate;
//	double sigmax;
//	double sigmay;
//	double sigmaw;
//	double exp_coeff_re_id;
//	double re_id_weight;
//	double hog_particle_expand;
//	int thread_num;
//	double aspect_ratio;
//
//	string seq_path, seq_name;
//	int start_frame, end_frame, nz;
//	string ext;
//	int initx, inity, initw, inith;
//	try
//	{
//		store(parse_command_line(argc, argv, desc), vm);
//		notify(vm);
//		if (vm.count("help"))
//		{
//			cout << desc << endl;
//			return 1;
//		}
//		if (vm.count("particles"))
//			particles = vm["particles"].as<int>();
//		if (vm.count("optical_flow"))
//			use_optical_flow_lk = vm["optical_flow"].as<bool>();
//		if (vm.count("update_online"))
//			update_online = vm["update_online"].as<bool>();
//		if (vm.count("learning_rate"))
//			learning_rate = vm["learning_rate"].as<double>();
//		if (vm.count("sigma_x"))
//			sigmax = vm["sigma_x"].as<double>();
//		if (vm.count("sigma_y"))
//			sigmay = vm["sigma_y"].as<double>();
//		if (vm.count("sigma_w"))
//			sigmaw = vm["sigma_w"].as<double>();
//		if (vm.count("exp_coeff_re_id"))
//			exp_coeff_re_id = vm["exp_coeff_re_id"].as<double>();
//		if (vm.count("re-id_weight"))
//			re_id_weight = vm["re-id_weight"].as<double>();
//		if (vm.count("hog_particle_expand"))
//			hog_particle_expand = vm["hog_particle_expand"].as<double>();
//		if (vm.count("thread_num"))
//			thread_num = vm["thread_num"].as<int>();
//		if (vm.count("aspect_ratio"))
//			aspect_ratio = vm["aspect_ratio"].as<double>();
//		if (vm.count("seq_path")){
//			seq_path = vm["seq_path"].as<string>();
//			if (seq_path[seq_path.size()-1] != '\\' && seq_path[seq_path.size()-1] != '/')
//				seq_path += "/";
//		}
//		if (vm.count("seq_name"))
//			seq_name = vm["seq_name"].as<string>();
//		if (vm.count("start_frame"))
//			start_frame = vm["start_frame"].as<int>();
//		if (vm.count("end_frame"))
//			end_frame = vm["end_frame"].as<int>();
//		if (vm.count("nz"))
//			nz = vm["nz"].as<int>();
//		if (vm.count("ext"))
//			ext = vm["ext"].as<string>();
//		if (vm.count("initx"))
//			initx = vm["initx"].as<int>();
//		if (vm.count("inity"))
//			inity = vm["inity"].as<int>();
//		if (vm.count("initw"))
//			initw = vm["initw"].as<int>();
//		if (vm.count("inith"))
//			inith = vm["inith"].as<int>();
//
//		ofstream result_file(seq_name+"_PF.txt");
//		result_file << initx << "	" << inity << "	" << initw << "	" << inith << endl;
//		ofstream fps_file(seq_name+"_PF_FPS.txt");
//		PTUsingReIdandPF tracker(3, particles, use_optical_flow_lk, update_online, learning_rate, sigmax, sigmay, sigmaw, exp_coeff_re_id, re_id_weight, hog_particle_expand, thread_num, aspect_ratio);
//
//		cv::Rect initialization(initx, inity, initw, inith);
//		cv::Mat image;
//
//		//read first image
//		stringstream ss;
//		ss << seq_path;
//		ss << setfill('0') << setw(nz) << start_frame;
//		ss << "." << ext;
//		string imgname;
//		ss >> imgname;
//		image = imread(imgname);
//
//		tic
//		tracker.init(image, initialization);
//
//		for (int i = start_frame+1; i <= end_frame; ++i) {
//
//			stringstream ss;
//			ss << seq_path;
//			ss << setfill('0') << setw(nz) << i;
//			ss << "." << ext;
//			string imgname;
//			ss >> imgname;
//
//			cv::Mat image = cv::imread(imgname);
//
//			cv::Rect rect = tracker.track(image);
//			result_file << rect.x << "	" << rect.y << "	" << rect.width << "	" << rect.height << endl;
//
//// 			cv::rectangle(image, rect, cv::Scalar(0, 0, 255), 2);
//// 			cv::imshow("result", image);
//// 			cv::waitKey(10);
//		}
//		double tot_time = toc;
//		cout << "FPS: " << 1.*(end_frame - start_frame + 1) / tot_time << endl;
//		fps_file << 1.*(end_frame - start_frame + 1) / tot_time;
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

#include <fstream>
#include "opencv2/opencv.hpp"
#include <vector>

using namespace std;
using namespace cv;

double hog_predict(const std::vector<float>& v)
{
	static vector<float> hogDetector = HOGDescriptor::getDefaultPeopleDetector();
	CV_Assert(v.size()<=hogDetector.size());
	float s = hogDetector[hogDetector.size()-1];
	int k = 0;
	std::for_each(v.begin(),v.end(),[&](const float d){s+=d*hogDetector[k];k++;});
	return s;
}

int main()
{
	Mat img = imread("D:\\data_seq\\Human7\\0001.jpg");
	Rect r(110,111,37,116);
	int width = cvRound(0.41*r.height);
	int x = r.x - (width-r.width)/2;
	r.width = width; r.x = x < 0 ? 0:x;
	Mat showimg;
	img.copyTo(showimg);
	rectangle(showimg, r, Scalar(0,0,255),2);

	int cterx = r.x+r.width/2, ctery = r.y+r.height/2;
	int hw = 1.4*r.width, hh = 1.*hw/0.41;
	Rect hogr(cterx-hw/2, ctery-hh/2, hw, hh);
	Rect hogroi = hogr&Rect(0,0,img.cols, img.rows);
	Mat roi; img(hogroi).copyTo(roi);
	copyMakeBorder(roi, roi, hogroi.y-hogr.y, hogr.br().y-hogroi.br().y, hogroi.x-hogr.x, hogr.br().x-hogroi.br().x, cv::BORDER_REPLICATE);
	rectangle(showimg, hogr, Scalar(255,0,0), 2);
	imshow("image", showimg);
	Mat roi1, roi2;
	resize(img(r), roi1, Size(64, 128));
	imshow("roi1", roi1);
	resize(roi, roi2, Size(64, 128));
	imshow("roi2", roi2);

	HOGDescriptor hog;
	vector<float> descrip1, descrip2;
	hog.compute(roi1, descrip1);
	double score1 = hog_predict(descrip1);
	hog.compute(roi2, descrip2);
	double score2 = hog_predict(descrip2);

	int cx = hogr.x+hogr.width/2, cy = hogr.y+hogr.height/2;
	int hw3 = 1.4*hogr.width, hh3 = 1.*hw3/0.41;
	Rect hogr3(cx-hw3/2, cy-hh3/2, hw3, hh3);
	Rect hogroi3 = hogr3&Rect(0,0,img.cols, img.rows);
	Mat roi3; img(hogroi3).copyTo(roi3);
	copyMakeBorder(roi3, roi3, hogroi3.y-hogr3.y, hogr3.br().y-hogroi3.br().y, hogroi3.x-hogr3.x, hogr3.br().x-hogroi3.br().x, cv::BORDER_REPLICATE);
	Mat roi4; resize(roi3, roi4, Size(64, 128));
	imshow("roi3", roi4);
	vector<float> descrip3;
	hog.compute(roi4, descrip3);
	double score3 = hog_predict(descrip3);

	cout << score1 << endl << score2 << score3 <<endl;
	waitKey(0);
}