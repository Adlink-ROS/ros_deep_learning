/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "ros_compat.h"
#include "image_converter.h"

#include <jetson-inference/detectNet.h>

#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <unordered_map>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cam_lidar_calib/ImageDepth.h>

// globals
detectNet* net = NULL;
uint32_t overlay_flags = detectNet::OVERLAY_NONE;

imageConverter* input_cvt   = NULL;
imageConverter* overlay_cvt = NULL;

Publisher<vision_msgs::Detection2DArray> detection_pub = NULL;
Publisher<sensor_msgs::Image> overlay_pub = NULL;
Publisher<vision_msgs::VisionInfo> info_pub = NULL;

vision_msgs::VisionInfo info_msg;

// filter image and depth info
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, cam_lidar_calib::ImageDepth> SyncPolicy;
std::shared_ptr< message_filters::Subscriber<sensor_msgs::Image> > image_sub;
std::shared_ptr< message_filters::Subscriber<cam_lidar_calib::ImageDepth> > depth_sub;
std::shared_ptr< message_filters::Synchronizer<SyncPolicy> > filter;

std::string label_str[] = {
"unlabeled",
"person",
"bicycle",
"car",
"motorcycle",
"airplane",
"bus",
"train",
"truck",
"boat",
"traffic light",
"fire hydrant",
"street sign",
"stop sign",
"parking meter",
"bench",
"bird",
"cat",
"dog",
"horse",
"sheep",
"cow",
"elephant",
"bear",
"zebra",
"giraffe",
"hat",
"backpack",
"umbrella",
"shoe",
"eye glasses",
"handbag",
"tie",
"suitcase",
"frisbee",
"skis",
"snowboard",
"sports ball",
"kite",
"baseball bat ",
"baseball glove",
"skateboard",
"surfboard",
"tennis racket",
"bottle",
"plate",
"wine glass",
"cup",
"fork",
"knife",
"spoon",
"bowl",
"banana",
"apple",
"sandwich",
"orange",
"broccoli",
"carrot",
"hot dog ",
"pizza",
"donut",
"cake",
"chair",
"couch",
"potted plant",
"bed",
"mirror",
"dining table",
"window",
"desk",
"toilet",
"door",
"tv",
"laptop",
"mouse",
"remote",
"keyboard",
"cell phone",
"microwave",
"oven",
"toaster",
"sink",
"refrigerator",
"blender",
"book",
"clock",
"vase",
"scissors",
"teddy bear",
"hair drier",
"toothbrush"
};

// triggered when a new subscriber connected
void info_callback()
{
	ROS_INFO("new subscriber connected to vision_info topic, sending VisionInfo msg");
	info_pub->publish(info_msg);
}


// publish overlay image
bool publish_overlay( detectNet::Detection* detections, int numDetections )
{
	// get the image dimensions
	const uint32_t width  = input_cvt->GetWidth();
	const uint32_t height = input_cvt->GetHeight();

	// assure correct image size
	if( !overlay_cvt->Resize(width, height, imageConverter::ROSOutputFormat) )
		return false;

	// generate the overlay
	if( !net->Overlay(input_cvt->ImageGPU(), overlay_cvt->ImageGPU(), width, height, 
				   imageConverter::InternalFormat, detections, numDetections, overlay_flags) )
	{
		return false;
	}

	// convert to ros image msg format
	sensor_msgs::Image ros_image_in;
	if( !overlay_cvt->Convert(ros_image_in, imageConverter::ROSOutputFormat) )
		return false;

	// store the timestamp in header field
	ros_image_in.header.stamp = ROS_TIME_NOW();

	// convert to opencv msg format in order to draw something on it
	sensor_msgs::ImagePtr ros_image_ptr;
    cv::Mat cv_image_in = cv_bridge::toCvShare(ros_image_in, ros_image_ptr, "bgr8")->image;
    
    for (auto i = 0; i < numDetections; ++i) {
	    cv::Point2d position(detections[i].Left+(detections[i].Right-detections[i].Left)/2, 
	                        detections[i].Top+(detections[i].Bottom-detections[i].Top)/2);
	    // cv::circle(cv_image_in, position, 10, cv::Scalar(255, 255, 255), -1);
	    cv::putText(cv_image_in, 
	                "Hello, OpenCV!",
	                position,
	                cv::FONT_HERSHEY_DUPLEX,
	                1.0, 
	                CV_RGB(255, 255, 255), 
	                1);
	}

	// convert back to ros image msg format
    sensor_msgs::ImagePtr ros_image_out_ptr = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_image_in).toImageMsg();
                
	// publish the message	
	overlay_pub->publish(ros_image_out_ptr);
	ROS_DEBUG("publishing %ux%u overlay image", width, height);
}

float getDepthByPoint(cv::Point2d point, const cam_lidar_calib::ImageDepthConstPtr &depth_msg)
{
	const float tolerance = 0.025;
	
	for (auto i = 0; i < depth_msg->size; ++i) {
        auto x = abs(point.x - depth_msg->data[i].x) / point.x;
        auto y = abs(point.y - depth_msg->data[i].y) / point.y;
        if (x < tolerance && y < tolerance) {
            return depth_msg->data[i].z;
		}
    }
    return -1.0; // not found
}

bool publish_overlay_with_depth( detectNet::Detection* detections, int numDetections, const cam_lidar_calib::ImageDepthConstPtr &depth_msg)
{
	// get the image dimensions
	const uint32_t width  = input_cvt->GetWidth();
	const uint32_t height = input_cvt->GetHeight();

	// assure correct image size
	if( !overlay_cvt->Resize(width, height, imageConverter::ROSOutputFormat) )
		return false;

	// generate the overlay
	if( !net->Overlay(input_cvt->ImageGPU(), overlay_cvt->ImageGPU(), width, height, 
				   imageConverter::InternalFormat, detections, numDetections, overlay_flags) )
	{
		return false;
	}

	// convert to ros image msg format
	sensor_msgs::Image ros_image_in;
	if( !overlay_cvt->Convert(ros_image_in, imageConverter::ROSOutputFormat) )
		return false;

	// store the timestamp in header field
	ros_image_in.header.stamp = ROS_TIME_NOW();

	// convert to opencv msg format in order to draw something on it
	sensor_msgs::ImagePtr ros_image_ptr;
    cv::Mat cv_image_in = cv_bridge::toCvShare(ros_image_in, ros_image_ptr, "bgr8")->image;
    
    for (auto i = 0; i < numDetections; ++i) {
	    cv::Point2d label_pos(detections[i].Left, detections[i].Top + 20);
	    cv::Point2d center(detections[i].Left+(detections[i].Right-detections[i].Left)/2, 
	                        detections[i].Top+(detections[i].Bottom-detections[i].Top)/2);
	    float depth = getDepthByPoint(center, depth_msg);
	    if (depth < 0) continue;
	    
	    std::string object_str = label_str[detections->ClassID];
	    std::stringstream stream;
	    stream.precision(2);
        stream << std::fixed << depth;
        std::string depth_str = stream.str();
        stream.str(""); // clear
        stream << std::fixed << detections->Confidence*100;
        std::string confidence_str = stream.str();
	    cv::putText(cv_image_in, 
	                object_str + ": " + confidence_str + "%, distance: " + depth_str + "m",
	                label_pos,
	                cv::FONT_HERSHEY_COMPLEX,
	                1.0, 
	                CV_RGB(255, 255, 255), 
	                1);
	}

	// convert back to ros image msg format
    sensor_msgs::ImagePtr ros_image_out_ptr = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_image_in).toImageMsg();
                
	// publish the message	
	overlay_pub->publish(ros_image_out_ptr);
	ROS_DEBUG("publishing %ux%u overlay image", width, height);
}

// input image subscriber callback
void img_callback( const sensor_msgs::ImageConstPtr input )
{
	// convert the image to reside on GPU
	if( !input_cvt || !input_cvt->Convert(input) )
	{
		ROS_INFO("failed to convert %ux%u %s image", input->width, input->height, input->encoding.c_str());
		return;	
	}

	// classify the image
	detectNet::Detection* detections = NULL;

	const int numDetections = net->Detect(input_cvt->ImageGPU(), input_cvt->GetWidth(), input_cvt->GetHeight(), &detections, detectNet::OVERLAY_NONE);

	// verify success	
	if( numDetections < 0 )
	{
		ROS_ERROR("failed to run object detection on %ux%u image", input->width, input->height);
		return;
	}

	// if objects were detected, send out message
	if( numDetections > 0 )
	{
		ROS_INFO("detected %i objects in %ux%u image", numDetections, input->width, input->height);
		
		// create a detection for each bounding box
		vision_msgs::Detection2DArray msg;

		for( int n=0; n < numDetections; n++ )
		{
			detectNet::Detection* det = detections + n;

			ROS_INFO("object %i class #%u (%s)  confidence=%f", n, det->ClassID, net->GetClassDesc(det->ClassID), det->Confidence);
			ROS_INFO("object %i bounding box (%f, %f)  (%f, %f)  w=%f  h=%f", n, det->Left, det->Top, det->Right, det->Bottom, det->Width(), det->Height()); 
			
			// create a detection sub-message
			vision_msgs::Detection2D detMsg;

			detMsg.bbox.size_x = det->Width();
			detMsg.bbox.size_y = det->Height();
			
			float cx, cy;
			det->Center(&cx, &cy);

			detMsg.bbox.center.x = cx;
			detMsg.bbox.center.y = cy;

			detMsg.bbox.center.theta = 0.0f;		// TODO optionally output object image

			// create classification hypothesis
			vision_msgs::ObjectHypothesisWithPose hyp;
			
		#if ROS_DISTRO >= ROS_GALACTIC
			hyp.hypothesis.class_id = det->ClassID;
			hyp.hypothesis.score = det->Confidence;
		#else
			hyp.id = det->ClassID;
			hyp.score = det->Confidence;
		#endif
		
			detMsg.results.push_back(hyp);
			msg.detections.push_back(detMsg);
		}

		// populate timestamp in header field
		msg.header.stamp = ROS_TIME_NOW();

		// publish the detection message
		detection_pub->publish(msg);
	}

	// generate the overlay (if there are subscribers)
	if( ROS_NUM_SUBSCRIBERS(overlay_pub) > 0 )
		publish_overlay(detections, numDetections);
}

void img_depth_callback(const sensor_msgs::ImageConstPtr &image_msg,
						const cam_lidar_calib::ImageDepthConstPtr &depth_msg)
{	
	// convert the image to reside on GPU
	if( !input_cvt || !input_cvt->Convert(image_msg) )
	{
		ROS_INFO("failed to convert %ux%u %s image", image_msg->width, image_msg->height, image_msg->encoding.c_str());
		return;	
	}

	// classify the image
	detectNet::Detection* detections = NULL;

	const int numDetections = net->Detect(input_cvt->ImageGPU(), input_cvt->GetWidth(), input_cvt->GetHeight(), &detections, detectNet::OVERLAY_NONE);

	// verify success	
	if( numDetections < 0 )
	{
		ROS_ERROR("failed to run object detection on %ux%u image", image_msg->width, image_msg->height);
		return;
	}

	// generate the overlay (if there are subscribers)
	if( ROS_NUM_SUBSCRIBERS(overlay_pub) > 0 )
		publish_overlay_with_depth(detections, numDetections, depth_msg);
}

// node main loop
int main(int argc, char **argv)
{
	/*
	 * create node instance
	 */
	ROS_CREATE_NODE("detectnet");

	/*
	 * retrieve parameters
	 */	
 	std::string camera_name  = "port_0";

	std::string model_name  = "ssd-mobilenet-v2";
	std::string model_path;
	std::string prototxt_path;
	std::string class_labels_path;
	
	std::string input_blob  = DETECTNET_DEFAULT_INPUT;
	std::string output_cvg  = DETECTNET_DEFAULT_COVERAGE;
	std::string output_bbox = DETECTNET_DEFAULT_BBOX;
	std::string overlay_str = "box,labels,conf";

	float mean_pixel = 0.0f;
	float threshold  = DETECTNET_DEFAULT_THRESHOLD;

	ROS_DECLARE_PARAMETER("camera_name", camera_name);
	ROS_DECLARE_PARAMETER("model_name", model_name);
	ROS_DECLARE_PARAMETER("model_path", model_path);
	ROS_DECLARE_PARAMETER("prototxt_path", prototxt_path);
	ROS_DECLARE_PARAMETER("class_labels_path", class_labels_path);
	ROS_DECLARE_PARAMETER("input_blob", input_blob);
	ROS_DECLARE_PARAMETER("output_cvg", output_cvg);
	ROS_DECLARE_PARAMETER("output_bbox", output_bbox);
	ROS_DECLARE_PARAMETER("overlay_flags", overlay_str);
	ROS_DECLARE_PARAMETER("mean_pixel_value", mean_pixel);
	ROS_DECLARE_PARAMETER("threshold", threshold);


	/*
	 * retrieve parameters
	 */
	ROS_GET_PARAMETER("camera_name", camera_name);
	ROS_GET_PARAMETER("model_path", model_path);
	ROS_GET_PARAMETER("prototxt_path", prototxt_path);
	ROS_GET_PARAMETER("class_labels_path", class_labels_path);
	ROS_GET_PARAMETER("input_blob", input_blob);
	ROS_GET_PARAMETER("output_cvg", output_cvg);
	ROS_GET_PARAMETER("output_bbox", output_bbox);
	ROS_GET_PARAMETER("overlay_flags", overlay_str);
	ROS_GET_PARAMETER("mean_pixel_value", mean_pixel);
	ROS_GET_PARAMETER("threshold", threshold);

	overlay_flags = detectNet::OverlayFlagsFromStr(overlay_str.c_str());


	/*
	 * load object detection network
	 */
	if( model_path.size() > 0 )
	{
		// create network using custom model paths
		net = detectNet::Create(prototxt_path.c_str(), model_path.c_str(), 
						    mean_pixel, class_labels_path.c_str(), threshold, 
						    input_blob.c_str(), output_cvg.c_str(), output_bbox.c_str());
	}
	else
	{
		// determine which built-in model was requested
		detectNet::NetworkType model = detectNet::NetworkTypeFromStr(model_name.c_str());

		if( model == detectNet::CUSTOM )
		{
			ROS_ERROR("invalid built-in pretrained model name '%s', defaulting to pednet", model_name.c_str());
			model = detectNet::SSD_MOBILENET_V2;
		}

		// create network using the built-in model
		net = detectNet::Create(model, threshold);
	}

	if( !net )
	{
		ROS_ERROR("failed to load detectNet model");
		return 0;
	}


	/*
	 * create the class labels parameter vector
	 */
	std::hash<std::string> model_hasher;  // hash the model path to avoid collisions on the param server
	std::string model_hash_str = std::string(net->GetModelPath()) + std::string(net->GetClassPath());
	const size_t model_hash = model_hasher(model_hash_str);
	
	ROS_INFO("model hash => %zu", model_hash);
	ROS_INFO("hash string => %s", model_hash_str.c_str());

	// obtain the list of class descriptions
	std::vector<std::string> class_descriptions;
	const uint32_t num_classes = net->GetNumClasses();

	for( uint32_t n=0; n < num_classes; n++ )
		class_descriptions.push_back(net->GetClassDesc(n));

	// create the key on the param server
	std::string class_key = std::string("class_labels_") + std::to_string(model_hash);

	ROS_DECLARE_PARAMETER(class_key, class_descriptions);
	ROS_SET_PARAMETER(class_key, class_descriptions);
		
	// populate the vision info msg
	std::string node_namespace = ROS_GET_NAMESPACE();
	ROS_INFO("node namespace => %s", node_namespace.c_str());

	info_msg.database_location = node_namespace + std::string("/") + class_key;
	info_msg.database_version  = 0;
	info_msg.method 		  = net->GetModelPath();
	
	ROS_INFO("class labels => %s", info_msg.database_location.c_str());


	/*
	 * create image converter objects
	 */
	input_cvt = new imageConverter();
	overlay_cvt = new imageConverter();

	if( !input_cvt || !overlay_cvt )
	{
		ROS_ERROR("failed to create imageConverter objects");
		return 0;
	}


	/*
	 * advertise publisher topics
	 */
	ROS_CREATE_PUBLISHER(vision_msgs::Detection2DArray, "detections", 25, detection_pub);
	ROS_CREATE_PUBLISHER(sensor_msgs::Image, "overlay", 2, overlay_pub);
	
	ROS_CREATE_PUBLISHER_STATUS(vision_msgs::VisionInfo, "vision_info", 1, info_callback, info_pub);


	/*
	 * subscribe to image topic
	 */
#if 0
	auto img_sub = ROS_CREATE_SUBSCRIBER(sensor_msgs::Image, "image_in", 5, img_callback);
#else
	/*
	 * message filter for image and depth info
	 */
	std::string camera_in_topic = "/" + camera_name + "/camera/image_raw";
	std::string depthOutTopic = camera_in_topic + "/image_depth";
    image_sub.reset(new message_filters::Subscriber<sensor_msgs::Image>(nh, camera_in_topic, 1));
	depth_sub.reset(new message_filters::Subscriber<cam_lidar_calib::ImageDepth>(nh, depthOutTopic, 1));
	filter.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), *image_sub, *depth_sub));
    filter->registerCallback(boost::bind(&img_depth_callback, _1, _2));
#endif

	/*
	 * wait for messages
	 */
	ROS_INFO("detectnet node initialized, waiting for messages");
	ROS_SPIN();


	/*
	 * free resources
	 */
	delete net;
	delete input_cvt;
	delete overlay_cvt;

	return 0;
}

