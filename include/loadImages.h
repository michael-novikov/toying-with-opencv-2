#pragma once

#include "opencv2/core/core_c.h"
#include "opencv2/ml/ml.hpp"

#include <cstdio>
#include <vector>
#include <string>

bool loadImages(const std::string& dir_path, int data_num, int var_count, cv::Mat* data, cv::Mat* responses);