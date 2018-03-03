#include "loadImages.h"

#include "opencv2/core/core_c.h"
#include "opencv2/opencv.hpp"
#include "opencv2/ml/ml.hpp"

#include <filesystem>
#include <string>
#include <vector>
#include <random>
#include <iterator>

using namespace std;
// TODO: use VS2017, which implements C++17
namespace fs = std::experimental::filesystem::v1;
using namespace cv;

bool loadImages(const string& dir_path, int data_num, int var_count, Mat* data, Mat* responses) {
	vector<fs::path> dirs;
	vector<int> class_files_num;
	vector<uniform_int_distribution<int>> class_unifs;

	data->create(data_num, var_count, CV_8UC1);
	responses->create(data_num, 1, CV_8UC1);

	for (auto& p : fs::directory_iterator(dir_path)) {
		if (fs::is_directory(p.path())) {
			dirs.push_back(p.path());

			fs::directory_iterator begin(p.path()), end;
			auto reg_file = static_cast<bool(*)(const fs::path&)>(fs::is_regular_file);
			int count = std::count_if(begin, end, reg_file);
			class_files_num.push_back(count);

			class_unifs.push_back(uniform_int_distribution<int>(0, count - 1));
		}
	}

	printf("Loading %i images.\n", data_num);

	int class_num = dirs.size();
	uniform_int_distribution<int> unif(0, class_num - 1);
	default_random_engine re;

	for (int i = 0; i < data_num; ++i) {
		int dir = unif(re);

		while (class_files_num[dir] == 0) {
			dir = (dir + 1) % class_num; // TODO: do it more safety
		}

		int file_num = class_unifs[dir](re);
		fs::directory_iterator it(dirs[dir]);

		int cur = 0;
		while (cur != file_num) {
			++it;
			++cur;
		}

		string filename = it->path().string();

		Mat img = imread(filename, 0);
		img.convertTo(img, CV_8UC1);

		if (!img.data) {
			return false;
		}

		//Mat element = getStructuringElement(MORPH_CROSS, Size(5, 5), Point(5, 5));
		/// Apply the dilation operation
		//dilate(img, img, element);

		GaussianBlur(img, img, Size(5, 5), 0, 0);
		threshold(img, img, 0, 255, THRESH_BINARY | THRESH_OTSU);
		
		int j = 0;
		for (int r = 0; r < img.rows; ++r) {
			for (int c = 0; c < img.cols; ++c) {
				data->at<unsigned char>(i, j) = img.at<unsigned char>(r, c);
				responses->at<unsigned char>(i, 0) = (unsigned char)dir;
				++j;
			}
		}

		--class_files_num[dir];

		printf("%i/%i\n", i, data_num);
	}

	return true;
}