import blobconverter

model_path = blobconverter.from_zoo(name="yolop_320x320",
                                    zoo_type="depthai",
                                    shaves=6)

#load model 
import cv2
import depthai as dai
import numpy as np

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(320, 320)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(40)

# Create output
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

# Create a Neural Network that will make object detection
detection_nn = pipeline.createYoloDetectionNetwork()
detection_nn.setConfidenceThreshold(0.5)
detection_nn.setNumClasses(80)
detection_nn.setCoordinateSize(4)

detection_nn.setBlobPath(model_path)