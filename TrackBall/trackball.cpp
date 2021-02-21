#include "jetson-utils/videoSource.h"
#include "jetson-utils/videoOutput.h"
#include "jetson-inference/detectNet.h"
#include <jetson-utils/cudaOverlay.h>
#include <jetson-utils/cudaMappedMemory.h>
#include <jetson-utils/imageIO.h>
#include "opencv2/opencv.hpp"
#include <signal.h>
#include <typeinfo>

using namespace std;
using namespace cv;

void drawArrow(cv::Mat& img, cv::Point pStart, cv::Point pEnd, int len, int alpha, cv::Scalar& color, int thickness, int lineType){
    const double PI = 3.1415926;
    Point arrow;
    //Calculate the ¦È angle
    double angle = atan2((double)(pStart.y - pEnd.y), (double)(pStart.x - pEnd.x));
    line(img, pStart, pEnd, color, thickness, lineType);
    //Calculate the end position of the other end of the arrow corner
    arrow.x = pEnd.x + len * cos(angle + PI * alpha / 180);
    arrow.y = pEnd.y + len * sin(angle + PI * alpha / 180);
    line(img, pEnd, arrow, color, thickness, lineType);
    arrow.x = pEnd.x + len * cos(angle - PI * alpha / 180);
    arrow.y = pEnd.y + len * sin(angle - PI * alpha / 180);
    line(img, pEnd, arrow, color, thickness, lineType);
}

bool signal_recieved = false;

void sig_handler(int signo){
	if(signo == SIGINT){
		LogVerbose("received SIGINT\n");
		signal_recieved = true;
	}
}

int main(int argc, char** argv){
	//parse command line
	commandLine cmdLine(argc, argv, (const char*)NULL);

	//create input stream
	videoSource* input = videoSource::Create(cmdLine, ARG_POSITION(0));
    if(!input){
		LogError("TrackBall:  failed to create input stream\n");
		return 0;
	}

	//create output stream
	videoOutput* output = videoOutput::Create(cmdLine, ARG_POSITION(1));
	if(!output)
		LogError("TrackBall:  failed to create output stream\n");

    //create detection network
    detectNet* net = detectNet::Create( "../../Dataset/models/detectnet.prototxt",\
										"../../Dataset/models/onnx_file/ball-mix_with_GoogleDataset.onnx",\
										"",\
                                        "../../Dataset/models/onnx_file/labels.txt",\
                                        (float)0.5,\
                                        "input_0",\
										"scores",\
                                        "boxes",\
										DEFAULT_MAX_BATCH_SIZE,\
										TYPE_FASTEST,\
										DEVICE_GPU,\
										(bool)1);
	if(!net){
		LogError("TrackBall:  failed to load detectNet model\n");
		return 0;
	}

	// parse overlay flags
	const uint32_t overlayFlags = detectNet::OverlayFlagsFromStr("box,labels,conf");

	//draw top layer
	Mat TopImg = Mat(input->GetHeight(), input->GetWidth(), CV_8UC4, Scalar(255, 255, 255, 0));
	Point CurP = Point(0, 0);
	Point PreP = Point(0, 0);
	Scalar Color = Scalar(225, 0, 0);
	uchar4* CUDATopImg = NULL;
	uchar4* imgOutput = NULL;
	int2 dimsA = make_int2(input->GetWidth(),input->GetHeight());
	int2 dimsOutput = make_int2(2*dimsA.x, dimsA.y);
	cudaAllocMapped(&imgOutput, dimsOutput.x, dimsOutput.y);
	cudaAllocMapped(&CUDATopImg, dimsA.x, dimsA.y);

	//processing loop
	while( !signal_recieved ){
		// capture next image image
		uchar4* image = NULL;
		TopImg = Scalar(255,255,255,0);

		if(!input->Capture(&image, 1000)){
			// check for EOS
			if(!input->IsStreaming())
				break; 

			LogError("TrackBall:  failed to capture video frame\n");
			continue;
		}

		// detect objects in the frame
		detectNet::Detection* detections = NULL;

		const int numDetections = net->Detect(image, input->GetWidth(), input->GetHeight(), &detections, overlayFlags);

		if(numDetections > 0){
			LogVerbose("%i objects detected\n", numDetections);
		
			for(int n=0; n < numDetections; n++)
			{
				CurP.x = (detections[n].Left + detections[n].Right)/2;
				CurP.y = (detections[n].Top + detections[n].Bottom)/2;
				LogVerbose("detected obj %i  class #%u (%s)  confidence=%f\n", n, detections[n].ClassID, net->GetClassDesc(detections[n].ClassID), detections[n].Confidence);
				LogVerbose("bounding box %i  (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, detections[n].Left, detections[n].Top, detections[n].Right, detections[n].Bottom, detections[n].Width(), detections[n].Height()); 
			}

			circle(TopImg, CurP, 10, Scalar(255,0,0,225),-1);
			drawArrow(TopImg, CurP, (CurP-PreP)*3+CurP, 10, 30, Color, 3, 1);
			PreP.x = CurP.x;
			PreP.y = CurP.y;
		}

		cudaMemcpy2D(CUDATopImg, dimsA.x*sizeof(uchar4), (void*)TopImg.data, TopImg.step, dimsA.x*sizeof(uchar4), dimsA.y, cudaMemcpyHostToDevice);
		CUDA(cudaOverlay(image, dimsA, imgOutput, dimsOutput, 0, 0));
		CUDA(cudaOverlay(CUDATopImg, dimsA, imgOutput, dimsOutput, dimsA.x, 0));

		// render outputs
		if(output != NULL)
		{
			output->Render(imgOutput, dimsOutput.x, dimsOutput.y);

			// update the status bar
			char str[256];
			sprintf(str, "TensorRT %i.%i.%i | %s | Network %.0f FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, precisionTypeToStr(net->GetPrecision()), net->GetNetworkFPS());
			output->SetStatus(str);

			// check if the user quit
			if(!output->IsStreaming())
				signal_recieved = true;
		}

		// print out timing info
		net->PrintProfilerTimes();
	}
	//destroy resources
	LogVerbose("TrackBall:  shutting down...\n");
	
	SAFE_DELETE(input);
	SAFE_DELETE(output);
	SAFE_DELETE(net);

	LogVerbose("TrackBall:  shutdown complete.\n");
	return 0;
}
