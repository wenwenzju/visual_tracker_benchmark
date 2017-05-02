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

//-p 100 -l 0.9 -w 0.8 -y 20 -s 5 -e 0.6 --seq_path D:\data_seq\Human2 --seq_name Human2 --start_frame 1 --end_frame 1128 --nz 4 --ext jpg --initx 198 --inity 249 --initw 95 --inith 325
//-p 100 -l 0.9 -w 0.8 -y 20 -s 5 -e 0.6 --seq_path D:\data_seq\Skating2 --seq_name Skating2 --start_frame 1 --end_frame 473 --nz 4 --ext jpg --initx 289 --inity 67 --initw 64 --inith 236
//-p 100 -l 0.9 -w 0.8 -y 20 -s 5 -e 0.6 --seq_path D:\data_seq\Skating2 --seq_name Skating2 --start_frame 1 --end_frame 473 --nz 4 --ext jpg --initx 347 --inity 58 --initw 103 --inith 251
//-p 100 -l 0.9 -w 0.8 -y 20 -s 5 -e 0.6 --seq_path D:\data_seq\BlurBody --seq_name BlurBody --start_frame 1 --end_frame 334 --nz 4 --ext jpg --initx 400 --inity 48 --initw 87 --inith 319
//-p 100 -l 0.9 -w 0.8 -y 20 -s 5 -e 0.6 --seq_path D:\data_seq\David3 --seq_name David3 --start_frame 1 --end_frame 252 --nz 4 --ext jpg --initx 83 --inity 200 --initw 35 --inith 131
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --seq_path D:\data_seq\Human7 --seq_name Human7 --start_frame 1 --end_frame 250 --nz 4 --ext jpg --initx 110 --inity 111 --initw 37 --inith 116
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --seq_path D:\data_seq\Human8 --seq_name Human8 --start_frame 1 --end_frame 128 --nz 4 --ext jpg --initx 110 --inity 101 --initw 30 --inith 91			OK
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --seq_path D:\data_seq\Human9 --seq_name Human9 --start_frame 1 --end_frame 305 --nz 4 --ext jpg --initx 93 --inity 113 --initw 34 --inith 109
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --seq_path D:\data_seq\Singer1 --seq_name Singer1 --start_frame 1 --end_frame 351 --nz 4 --ext jpg --initx 51 --inity 53 --initw 87 --inith 290
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --seq_path D:\data_seq\Skater --seq_name Skater --start_frame 1 --end_frame 160 --nz 4 --ext jpg --initx 138 --inity 57 --initw 39 --inith 137			NO
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --seq_path D:\data_seq\Skater2 --seq_name Skater2 --start_frame 1 --end_frame 435 --nz 4 --ext jpg --initx 163 --inity 44 --initw 47 --inith 164		NO
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --seq_path D:\data_seq\Skating1 --seq_name Skating1 --start_frame 1 --end_frame 400 --nz 4 --ext jpg --initx 162 --inity 188 --initw 34 --inith 84		NO
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --seq_path D:\data_seq\Subway --seq_name Subway --start_frame 1 --end_frame 175 --nz 4 --ext jpg --initx 16 --inity 88 --initw 19 --inith 51			NO
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --seq_path D:\data_seq\Walking --seq_name Walking --start_frame 1 --end_frame 412 --nz 4 --ext jpg --initx 692 --inity 439 --initw 24 --inith 79		MOD
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --seq_path D:\data_seq\Woman --seq_name Woman --start_frame 1 --end_frame 597 --nz 4 --ext jpg --initx 213 --inity 121 --initw 21 --inith 95
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --seq_path D:\data_seq\Couple --seq_name Couple --start_frame 1 --end_frame 140 --nz 4 --ext jpg --initx 51 --inity 47 --initw 25 --inith 62			OK
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --seq_path D:\data_seq\Crossing --seq_name Crossing --start_frame 1 --end_frame 120 --nz 4 --ext jpg --initx 205 --inity 151 --initw 17 --inith 50		OK
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --seq_path D:\data_seq\Gym --seq_name Gym --start_frame 1 --end_frame 767 --nz 4 --ext jpg --initx 167 --inity 69 --initw 24 --inith 127
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --seq_path D:\data_seq\Human3 --seq_name Human3 --start_frame 1 --end_frame 1698 --nz 4 --ext jpg --initx 264 --inity 311 --initw 37 --inith 69		NO
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --seq_path D:\data_seq\Human4 --seq_name Human4 --start_frame 1 --end_frame 667 --nz 4 --ext jpg --initx 99 --inity 237 --initw 27 --inith 82			NO
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --seq_path D:\data_seq\Human5 --seq_name Human5 --start_frame 1 --end_frame 713 --nz 4 --ext jpg --initx 326 --inity 414 --initw 15 --inith 42			NO
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --seq_path D:\data_seq\Human6 --seq_name Human6 --start_frame 1 --end_frame 792 --nz 4 --ext jpg --initx 340 --inity 358 --initw 18 --inith 55			NO
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --seq_path D:\data_seq\BlurBody --seq_name BlurBody --start_frame 1 --end_frame 334 --nz 4 --ext jpg --initx 400 --inity 48 --initw 87 --inith 319
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --seq_path D:\data_seq\Jogging --seq_name Jogging --start_frame 1 --end_frame 307 --nz 4 --ext jpg --initx 111 --inity 98 --initw 25 --inith 101
//-p 100 -l 0.5 -w 0.8 -y 10 -s 5 -e 0.6 --seq_path D:\data_seq\Jogging --seq_name Jogging --start_frame 1 --end_frame 307 --nz 4 --ext jpg --initx 180 --inity 79 --initw 37 --inith 114
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --seq_path D:\data_seq\Walking2 --seq_name Walking2 --start_frame 1 --end_frame 500 --nz 4 --ext jpg --initx 130 --inity 132 --initw 31 --inith 115
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --seq_path D:\data_seq\Bolt --seq_name Bolt --start_frame 1 --end_frame 350 --nz 4 --ext jpg --initx 336 --inity 165 --initw 26 --inith 61				NO
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --seq_path D:\data_seq\Girl2 --seq_name Girl2 --start_frame 1 --end_frame 1500 --nz 4 --ext jpg --initx 294 --inity 135 --initw 44 --inith 171
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --seq_path D:\data_seq\Doll --seq_name Doll --start_frame 1 --end_frame 3872 --nz 4 --ext jpg --initx 146 --inity 150 --initw 32 --inith 73

//-p 100 -l 0.6 -w 0.8 -y 10 -s 5 -e 0.6 --rw_ratio 0.9 --binL 32 --binU 32 --binV 16 --binHOG 12 --cur_probe 1 --seq_path D:\data_seq\Woman --seq_name Woman --start_frame 1 --end_frame 597 --nz 4 --ext jpg --initx 213 --inity 121 --initw 21 --inith 95
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --rw_ratio 0.9 --binL 32 --binU 32 --binV 16 --binHOG 12 --cur_probe 1 --seq_path D:\data_seq\Human3 --seq_name Human3 --start_frame 1 --end_frame 1698 --nz 4 --ext jpg --initx 264 --inity 311 --initw 37 --inith 69
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --rw_ratio 0.9 --binL 32 --binU 32 --binV 16 --binHOG 12 --cur_probe 1 --seq_path D:\data_seq\Walking --seq_name Walking --start_frame 1 --end_frame 412 --nz 4 --ext jpg --initx 692 --inity 439 --initw 24 --inith 79
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --rw_ratio 0.5 --binL 32 --binU 32 --binV 16 --binHOG 12 --cur_probe 1 --seq_path D:\data_seq\Walking2 --seq_name Walking2 --start_frame 1 --end_frame 500 --nz 4 --ext jpg --initx 130 --inity 132 --initw 31 --inith 115
//-p 100 -l 0.9 -w 0.8 -y 20 -s 5 -e 0.6 --rw_ratio 0.9 --binL 32 --binU 32 --binV 16 --binHOG 12 --cur_probe 1 --seq_path D:\data_seq\David3 --seq_name David3 --start_frame 1 --end_frame 252 --nz 4 --ext jpg --initx 83 --inity 200 --initw 35 --inith 131
//-p 100 -l 0.9 -w 0.8 -y 20 -s 5 -e 0.6 --rw_ratio 0.9 --binL 32 --binU 32 --binV 16 --binHOG 12 --cur_probe 1 --seq_path D:\data_seq\Human2 --seq_name Human2 --start_frame 1 --end_frame 1128 --nz 4 --ext jpg --initx 198 --inity 249 --initw 95 --inith 325
//-p 100 -l 0.9 -w 0.8 -y 20 -s 5 -e 0.6 --rw_ratio 0.9 --binL 32 --binU 32 --binV 16 --binHOG 12 --cur_probe 1 --seq_path D:\data_seq\BlurBody --seq_name BlurBody --start_frame 1 --end_frame 334 --nz 4 --ext jpg --initx 400 --inity 48 --initw 87 --inith 319
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --rw_ratio 0.9 --binL 32 --binU 32 --binV 16 --binHOG 12 --cur_probe 1 --seq_path D:\data_seq\Human7 --seq_name Human7 --start_frame 1 --end_frame 250 --nz 4 --ext jpg --initx 110 --inity 111 --initw 37 --inith 116
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --rw_ratio 0.9 --binL 32 --binU 32 --binV 16 --binHOG 12 --cur_probe 1 --seq_path D:\data_seq\Human8 --seq_name Human8 --start_frame 1 --end_frame 128 --nz 4 --ext jpg --initx 110 --inity 101 --initw 30 --inith 91
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --rw_ratio 0.9 --binL 32 --binU 32 --binV 16 --binHOG 12 --cur_probe 1 --seq_path D:\data_seq\Human9 --seq_name Human9 --start_frame 1 --end_frame 305 --nz 4 --ext jpg --initx 93 --inity 113 --initw 34 --inith 109
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --rw_ratio 0.9 --binL 32 --binU 32 --binV 16 --binHOG 12 --cur_probe 1 --seq_path D:\data_seq\Singer1 --seq_name Singer1 --start_frame 1 --end_frame 351 --nz 4 --ext jpg --initx 51 --inity 53 --initw 87 --inith 290
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --rw_ratio 0.5 --binL 32 --binU 32 --binV 16 --binHOG 12 --cur_probe 1 --seq_path D:\data_seq\Skater --seq_name Skater --start_frame 1 --end_frame 160 --nz 4 --ext jpg --initx 138 --inity 57 --initw 39 --inith 137
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --rw_ratio 0.5 --binL 32 --binU 32 --binV 16 --binHOG 12 --cur_probe 1 --seq_path D:\data_seq\Skater2 --seq_name Skater2 --start_frame 1 --end_frame 435 --nz 4 --ext jpg --initx 163 --inity 44 --initw 47 --inith 164		bad
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --rw_ratio 0.5 --binL 32 --binU 32 --binV 16 --binHOG 12 --cur_probe 1 --seq_path D:\data_seq\Skating1 --seq_name Skating1 --start_frame 1 --end_frame 400 --nz 4 --ext jpg --initx 162 --inity 188 --initw 34 --inith 84		bad
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --rw_ratio 0.5 --binL 32 --binU 32 --binV 16 --binHOG 12 --cur_probe 1 --seq_path D:\data_seq\Walking --seq_name Walking --start_frame 1 --end_frame 412 --nz 4 --ext jpg --initx 692 --inity 439 --initw 24 --inith 79		bad
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --rw_ratio 0.5 --binL 32 --binU 32 --binV 16 --binHOG 12 --cur_probe 1 --seq_path D:\data_seq\Couple --seq_name Couple --start_frame 1 --end_frame 140 --nz 4 --ext jpg --initx 51 --inity 47 --initw 25 --inith 62			bad
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --rw_ratio 0.5 --binL 32 --binU 32 --binV 16 --binHOG 12 --cur_probe 1 --seq_path D:\data_seq\Crossing --seq_name Crossing --start_frame 1 --end_frame 120 --nz 4 --ext jpg --initx 205 --inity 151 --initw 17 --inith 50
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --rw_ratio 0.5 --binL 32 --binU 32 --binV 16 --binHOG 12 --cur_probe 1 --seq_path D:\data_seq\Gym --seq_name Gym --start_frame 1 --end_frame 767 --nz 4 --ext jpg --initx 167 --inity 69 --initw 24 --inith 127
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --rw_ratio 0.5 --binL 32 --binU 32 --binV 16 --binHOG 12 --cur_probe 1 --seq_path D:\data_seq\Human3 --seq_name Human3 --start_frame 1 --end_frame 1698 --nz 4 --ext jpg --initx 264 --inity 311 --initw 37 --inith 69
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --rw_ratio 0.5 --binL 32 --binU 32 --binV 16 --binHOG 12 --cur_probe 1 --seq_path D:\data_seq\Human4 --seq_name Human4 --start_frame 1 --end_frame 667 --nz 4 --ext jpg --initx 99 --inity 237 --initw 27 --inith 82
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --rw_ratio 0.5 --binL 32 --binU 32 --binV 16 --binHOG 12 --cur_probe 1 --seq_path D:\data_seq\Human5 --seq_name Human5 --start_frame 1 --end_frame 713 --nz 4 --ext jpg --initx 326 --inity 414 --initw 15 --inith 42
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --rw_ratio 0.5 --binL 32 --binU 32 --binV 16 --binHOG 12 --cur_probe 1 --seq_path D:\data_seq\Jogging --seq_name Jogging --start_frame 1 --end_frame 307 --nz 4 --ext jpg --initx 111 --inity 98 --initw 25 --inith 101
//-p 100 -l 0.5 -w 0.8 -y 10 -s 5 -e 0.6 --rw_ratio 0.5 --binL 32 --binU 32 --binV 16 --binHOG 12 --cur_probe 1 --seq_path D:\data_seq\Jogging --seq_name Jogging --start_frame 1 --end_frame 307 --nz 4 --ext jpg --initx 180 --inity 79 --initw 37 --inith 114
//-p 100 -l 0.9 -w 0.8 -y 10 -s 5 -e 0.6 --rw_ratio 0.5 --binL 32 --binU 32 --binV 16 --binHOG 12 --cur_probe 1 --seq_path D:\data_seq\Girl2 --seq_name Girl2 --start_frame 1 --end_frame 1500 --nz 4 --ext jpg --initx 294 --inity 135 --initw 44 --inith 171
//-p 100 -l 0.6 -w 0.8 -y 10 -s 5 -e 0.6 --rw_ratio 0.8 --binL 32 --binU 32 --binV 16 --binHOG 12 --cur_probe 1 --seq_path D:\data_seq\Girl2 --seq_name Girl2 --start_frame 1 --end_frame 1500 --nz 4 --ext jpg --initx 294 --inity 135 --initw 44 --inith 171			//good firstn = 30

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
			cv::waitKey(0);
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