//#include <fstream>
//#include <iostream>
//#include <iomanip>
//#include "opencv2/opencv.hpp"
//#include "boost/filesystem.hpp"
//
//using namespace cv;
//using namespace std;
//using namespace boost::filesystem;
//
//int main()
//{
//	string comPath("E:\\work_space\\05body_analysis\\03code\\win64\\visual_tracker_benchmark\\tracker_benchmark_v1.0\\compareVideo\\");
//	char videoNames[][10] = {"BlurBody","david3","Gym","Human7","Human8", "Human9"};
//
//	path toSave("compare_result");
//	if (!is_directory(toSave))
//		create_directory(toSave);
//	for (int i = 0; i < sizeof(videoNames)/sizeof(videoNames[0]); ++i)
//	{
//		if (!is_directory(toSave / videoNames[i]))
//			create_directory(toSave / videoNames[i]);
//	}
//
//	string rankFileName("first_5_rank.txt");
//	vector<string> firstTrackers;
//	std::ifstream rankFile(comPath+rankFileName);
//	int i = 0;
//	while(!rankFile.eof())
//	{
//		char tmp[10];
//		rankFile.getline(tmp,10);
//		firstTrackers.push_back(tmp);
//		cout << firstTrackers[i++] << endl;
//	}
//
//	vector<Scalar> colors;
//	colors.push_back(Scalar(0,0,255));
//	colors.push_back(Scalar(0,255,0));
//	colors.push_back(Scalar(255,0,0));
//	colors.push_back(Scalar(255,255,0));
//	colors.push_back(Scalar(0,255,255));
//
//	int fontFace = FONT_HERSHEY_SIMPLEX;
//	int baseline = 0;
//	int thickness = 2;
//	int textThickness = 1;
//	double fontScale = 0.5;
//	Size textSize = getTextSize("0", fontFace, fontScale, textThickness, &baseline);
//
//	for (int i = 0; i < sizeof(videoNames)/sizeof(videoNames[0]); ++i)
//	{
//		vector<vector<Rect> > oneVideoRects(firstTrackers.size());
//		string rectsName(string(videoNames[i])+"_");
//		for (int j = 0; j < firstTrackers.size(); ++j)
//		{
//			string tmpRectsName(rectsName+firstTrackers[j]);
//			tmpRectsName += ".txt";
//			std::ifstream rects(comPath+tmpRectsName);
//			while(!rects.eof())
//			{
//				char tmp[10];
//				rects.getline(tmp, 10, ',');
//				if (!tmp[0]) continue;
//				int x = atoi(tmp);
//				rects.getline(tmp, 10, ',');
//				int y = atoi(tmp);
//				rects.getline(tmp, 10, ',');
//				int w = atoi(tmp);
//				rects.getline(tmp, 10);
//				int h = atoi(tmp);
//				oneVideoRects[j].push_back(Rect(x,y,w,h));
//			}
//			rects.close();
//		}
//
//		int start_frame = 1, end_frame = oneVideoRects[0].size(), nz = 4;
//		string seq_path("D:\\data_seq\\");
//		seq_path += videoNames[i];
//		seq_path += "\\";
//		for (int j = start_frame; j <= end_frame; ++j) {
//
//			stringstream ss, ss_img;
//			ss << seq_path;
//			ss_img << "compare_result\\" << videoNames[i] << "\\";
//			ss << setfill('0') << setw(nz) << j;
//			ss_img << setfill('0') << setw(nz) << j;
//			ss << ".jpg";
//			ss_img << ".jpg";
//			string imgname;
//			ss >> imgname;
//
//			cv::Mat image = cv::imread(imgname);
//			Point textOrg(0, image.rows-1);
//			for (int jj = 0; jj < firstTrackers.size(); ++jj)
//			{
//				rectangle(image, oneVideoRects[jj][j-1], colors[jj], 2);
//				putText(image, (jj == 0 ? "Our" : firstTrackers[jj]), textOrg+Point(0,-(firstTrackers.size()-1-jj)*textSize.height), fontFace, fontScale, colors[jj], textThickness);
//			}
//			imshow("result", image);
//			waitKey(2);
//			imwrite(ss_img.str(), image);
//			//cv::rectangle(image, rect, cv::Scalar(0, 0, 255), 2);
//			//cv::imshow("result", image);
//			//cv::waitKey(10);
//		}
//	}
//}

#include <fstream>
#include <iostream>
#include <iomanip>
#include "opencv2/opencv.hpp"
#include "boost/filesystem.hpp"

using namespace cv;
using namespace std;
using namespace boost::filesystem;

int main()
{
	string comPath("..\\tracking_results\\");
	char videoNames[][20] = {"walking2", "skating1", "jogging.1", "jogging.2", "Human4.2", "Human5", "Human6", "Human3", 
	"Girl2", "crossing", "divid3", "Blurbody", "Gym", "Human7", "Human8", "Human9", "woman", "singer1", "Human2", "walking",
	"Skater2", "Skater", "Skating2.1", "Skating2.2"};
	//char videoNames[][20] = { "Human3"};

	path toSave("compare_result");
	if (!is_directory(toSave))
		create_directory(toSave);
	int len = sizeof(videoNames)/sizeof(videoNames[0]);
	cout << len << endl;
	for (int i = 0; i < len; ++i)
	{
		if (!is_directory(toSave / videoNames[i]))
			create_directory(toSave / videoNames[i]);
	}

	//string rankFileName("first_5_rank.txt");
	//vector<string> firstTrackers;
	//std::ifstream rankFile(comPath+rankFileName);
	//int i = 0;
	//while(!rankFile.eof())
	//{
	//	char tmp[10];
	//	rankFile.getline(tmp,10);
	//	firstTrackers.push_back(tmp);
	//	cout << firstTrackers[i++] << endl;
	//}
	vector<string> firstTrackers;
	firstTrackers.push_back("MEEM");
	firstTrackers.push_back("CCT");
	firstTrackers.push_back("MUSTer");
	firstTrackers.push_back("DSST");
	firstTrackers.push_back("RWSA");

	vector<Scalar> colors;
	colors.push_back(Scalar(0,255,0));
	colors.push_back(Scalar(255,0,0));
	colors.push_back(Scalar(255,255,0));
	colors.push_back(Scalar(255,0,255));
	colors.push_back(Scalar(0,0,255));

	for (int i = 0; i < sizeof(videoNames)/sizeof(videoNames[0]); ++i)
	{
		vector<vector<Rect> > oneVideoRects(firstTrackers.size());
		string rectsName(videoNames[i]);
		for (int j = 0; j < firstTrackers.size(); ++j)
		{
			string tmpRectsName(firstTrackers[j]+"_"+rectsName);
			tmpRectsName += ".txt";
			std::ifstream rects(comPath+tmpRectsName);
			while(!rects.eof())
			{
				char tmp[10];
				rects.getline(tmp, 10, ' ');
				if (strlen(tmp) == 0) 
				{
					break;
				}
				int x = atoi(tmp);
				rects.getline(tmp, 10, ' ');
				int y = atoi(tmp);
				rects.getline(tmp, 10, ' ');
				int w = atoi(tmp);
				rects.getline(tmp, 10);
				int h = atoi(tmp);
				oneVideoRects[j].push_back(Rect(x,y,w,h));
				cout << tmpRectsName << ": x = " << x << ", y = " << y << ", w = " << w << ", h = " << h << endl;
			}
			rects.close();
		}

		int start_frame = 1, end_frame = oneVideoRects[0].size(), nz = 4;
		string seq_path("D:\\data_seq\\");
		string tmp(videoNames[i]);
		seq_path += tmp.substr(0, tmp.find('.'));
		seq_path += "\\";
		for (int j = start_frame; j <= end_frame; ++j) {

			stringstream ss, ss_img;
			ss << seq_path;
			ss_img << "compare_result\\" << videoNames[i] << "\\";
			ss << setfill('0') << setw(nz) << j;
			ss_img << setfill('0') << setw(nz) << j;
			ss << ".jpg";
			ss_img << ".jpg";
			string imgname;
			ss >> imgname;

			cv::Mat image = cv::imread(imgname);
			for (int jj = 0; jj < firstTrackers.size(); ++jj)
			{
				rectangle(image, oneVideoRects[jj][j-1], colors[jj], 2);
			}
			string frameID("#"); frameID += to_string((long long)j);
			putText(image, frameID, Point(0, image.rows-5), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 2);
			imshow("result", image);
			waitKey(2);
			imwrite(ss_img.str(), image);
			//cv::rectangle(image, rect, cv::Scalar(0, 0, 255), 2);
			//cv::imshow("result", image);
			//cv::waitKey(10);
		}
	}
}