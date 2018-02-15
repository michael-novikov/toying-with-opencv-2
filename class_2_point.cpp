#include "opencv2/core/core_c.h"
#include "opencv2/ml/ml.hpp"

#include <cstdio>
#include <ctime>
#include <vector>

using namespace cv;

static int build_mlp_classifier()
{
	CvMat *data = 0;
	CvMat* responses = 0;
	int n = 1000;
	int a = -20;
	int b = 20;
	const int class_count = 1;

	CvMat train_data;

	CvMat* mlp_response = 0;


	RNG rng(1000);
	//srand(time(NULL));
	data = cvCreateMat(n, 2, CV_32F);
	responses = cvCreateMat(n, class_count, CV_32F);
	Mat d(n, 2, CV_32F);
	Mat r(n, 1, CV_32F);
	for (int i = 0; i < n; i++) {

		d.at<float>(i, 0) = (float)(rng.next() / (b - a)*0.1 + a);
		d.at<float>(i, 1) = (float)(rng.next() / (b - a)*0.1 + a);
		if (d.at<float>(i, 0) < (d.at<float>(i, 1)*d.at<float>(i, 1) + 10)) {
			r.at<float>(i, 0) = 1;
		}
		else {
			r.at<float>(i, 0) = -1;
		}
	}

	*data = d;

	*responses = r;
	int nsamples_all = 0, ntrain_samples = 0;
	int i, j;
	double train_hr = 0, test_hr = 0;
	CvANN_MLP mlp;


	printf("The database  is loaded.\n");
	nsamples_all = data->rows;
	ntrain_samples = (int)(nsamples_all*0.8);

	CvMat* new_responses = cvCreateMat(ntrain_samples, class_count, CV_32F);
	cvGetRows(responses, new_responses, 0, ntrain_samples);

	cvGetRows(data, &train_data, 0, ntrain_samples);
	//	for (int i = 0; i < ntrain_samples; i++){
	//		printf("%f %f %f \n", CV_MAT_ELEM(train_data, float, i, 0), d.at<float>(i, 0), CV_MAT_ELEM(*new_responses, float, i, 0)/*, ( (float*)(data->data.ptr + i*data->step))[0]*/);

	//}
	// 2. train classifier
	int layer_sz[] = { data->cols, 10, 50, 100, class_count };
	CvMat layer_sizes =
		cvMat(1, (int)(sizeof(layer_sz) / sizeof(layer_sz[0])), CV_32S, layer_sz);
	mlp.create(&layer_sizes, CvANN_MLP::GAUSSIAN, 1.0, 1.0);
	printf("Training the classifier (may take a few minutes)...\n");

	//#if 1
	//	int method = CvANN_MLP_TrainParams::BACKPROP;
	//double method_param = 0.001;
	//	int max_iter = 300;
	//#else
	int method = CvANN_MLP_TrainParams::RPROP;
	double method_param = 0.1;
	int max_iter = 1000;
	//#endif
	//printf("%d  %d \n",train_data.cols,new_responses->cols);
	int kk = mlp.train(&train_data, new_responses, 0, 0,
		CvANN_MLP_TrainParams(cvTermCriteria(CV_TERMCRIT_ITER, max_iter, 0.01),
			method, method_param));
	cvReleaseMat(&new_responses);
	printf(" train = %d \n", kk);
	//}

	mlp_response = cvCreateMat(1, class_count, CV_32F);

	// compute prediction error on train and test data
	for (i = 0; i < nsamples_all; i++)
	{
		int best_class;
		CvMat sample;
		cvGetRow(data, &sample, i);
		CvPoint max_loc = { 0, 0 };
		mlp.predict(&sample, mlp_response);
		//	cvMinMaxLoc(mlp_response, 0, 0, 0, &max_loc, 0);
		//	printf(" %f %f \n",/* CV_MAT_ELEM(*mlp_response, float, i, 0))*/ mlp_response->data.fl[i], responses->data.fl[i]);
		best_class = mlp_response->data.fl[0] > 0 ? 1 : -1;

		//int r = fabs((double)best_class - responses->data.fl[i]) < FLT_EPSILON ? 1 : 0;
		int r = best_class == responses->data.fl[i] ? 1 : 0;
		if (i < ntrain_samples)
			train_hr += r;
		else
			test_hr += r;
	}

	test_hr /= (double)(nsamples_all - ntrain_samples);
	train_hr /= (double)ntrain_samples;
	printf("Recognition rate: train = %.1f%%, test = %.1f%%\n",
		train_hr*100., test_hr*100.);




	cvReleaseMat(&mlp_response);
	cvReleaseMat(&data);
	cvReleaseMat(&responses);

	return 0;
}

int main_off(int argc, char *argv[])
{
	build_mlp_classifier();
	system("pause");
	return 0;
}