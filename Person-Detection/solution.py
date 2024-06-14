import depthai as dai
import numpy as np
import requests
import time
import cv2 as cv

from depthai_sdk import OakCamera
from numpy import linalg as LA
from robothub import LiveView
from robothub.application import BaseApplication

class DetectionParams:
    def __init__(self):
        self.id = 0
        self.prev_time = 0.0
        self.current_spent_time = 0.0
        self.prev_position = None
        self.crossed_line = False


def getAverage(detectionParamsArray : list):
    accum = 0.0

    for sent_time in detectionParamsArray:
        accum += sent_time

    if accum > 0:
        return accum / len(detectionParamsArray)
    else:
        return 0

class LineCrossingCounter:
    bottom_up = 0
    up_bottom = 0
    current_detections = 0
    previous_positions: dict = {}

    detection_dict = {}

    def process_packets(self, packets):
        frame = packets['color'].getCvFrame()
        color_packet = packets['color']
        nn_packet = packets['3_out;0_video']

        tracklets = nn_packet.daiTracklets.tracklets
        detections = nn_packet.detections

        for key in list(self.detection_dict.keys()):
            if key not in [tracklet.id for tracklet in tracklets]:
                self.detection_dict.pop(key)

        for detection, tracklet in zip(detections, tracklets):
            tracklet_id = tracklet.id
            tracklet_status = tracklet.status

            #print (f"ID: {tracklet_id} Status: {tracklet_status}")
        
            if tracklet_id not in self.detection_dict.keys():
                self.detection_dict[tracklet_id] = DetectionParams()
                self.detection_dict[tracklet_id].id = tracklet_id
                self.detection_dict[tracklet_id].prev_time = time.time()
                self.detection_dict[tracklet_id].current_spent_time = 0.0
                self.detection_dict[tracklet_id].prev_position = None
                self.detection_dict[tracklet_id].crossed_line = False
            elif tracklet_id in self.detection_dict.keys() and tracklet_status == dai.Tracklet.TrackingStatus.REMOVED:
                self.detection_dict.pop(tracklet_id)
                continue
            else:
                self.detection_dict[tracklet_id].current_spent_time += float(time.time() - self.detection_dict[tracklet_id].prev_time)
                self.detection_dict[tracklet_id].prev_time = time.time()


            bbox = [*detection.top_left, *detection.bottom_right]
            h, w = nn_packet.frame.shape[:2]
            normalized_roi = tracklet.roi.denormalize(w, h)
            current_position = self.get_roi_center(normalized_roi)
            previous_position = self.detection_dict[tracklet_id].prev_position

            if previous_position is not None and not self.detection_dict[tracklet_id].crossed_line:
                points = [self.LINE_P1, self.LINE_P2, previous_position, current_position]
                self.calc_ins(points, tracklet_id)
                points = [self.LINE_P3, self.LINE_P4, previous_position, current_position]
                self.calc_outs(points, tracklet_id)
                
            ident = "     " + str(tracklet_id)

            self.detection_dict[tracklet_id].prev_position = current_position

            self.live_view.add_rectangle(rectangle=bbox, label=detection.label + ident)
        
        average = getAverage([detection.current_spent_time for detection in self.detection_dict.values()])

        self.live_view.add_text(f'Out: {self.bottom_up}, In: {self.up_bottom}, Current: {self.bottom_up - self.up_bottom}, Detections: {len(self.detection_dict.keys())}, Average time: {average:.2f}',
                                coords=(100, 100),size=40,color=(50,50,50),thickness=20)
        self.live_view.add_text(f"Detections: {[detection.id for detection in self.detection_dict.values()]}",
                                coords=(100, 200),size=40,color=(50,50,50),thickness=20)
        self.live_view.add_text(f"Seconds: {[f'{detection.current_spent_time:.2f}' for detection in self.detection_dict.values()]}",
                                coords=(100, 240),size=40,color=(50,50,50),thickness=20)
       
        self.live_view.add_line(self.LINE_P1, self.LINE_P2,color=(250,250,250),thickness= 100 )
        self.live_view.add_line(self.LINE_P3, self.LINE_P4,color=(250,250,250),thickness= 100 )
        self.live_view.publish(color_packet.frame)

    def calc_vector_angle(self, point1, point2, point3, point4):
        u = self.create_vector(point1, point2)
        v = self.create_vector(point3, point4)
        return self.angle_between_vectors(u, v)
    
    def calc_ins(self,points, tracklet_id):
        if self.intersect(*points):
            self.detection_dict[tracklet_id].crossed_line = True
            angle = self.calc_vector_angle(*points)
            if angle > 180:
                self.up_bottom += 1

    def calc_outs(self,points, tracklet_id):
        if self.intersect(*points):
            self.detection_dict[tracklet_id].crossed_line = True
            angle = self.calc_vector_angle(*points)
            if angle < 180:
                self.bottom_up += 1

    @staticmethod
    def create_vector(point1, point2):
        return np.array([point2[0] - point1[0], point2[1] - point1[1]])

    @staticmethod
    def angle_between_vectors(u, v):
        i = np.inner(u, v)
        n = LA.norm(u) * LA.norm(v)
        c = i / n
        a = np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))
        return a if np.cross(u, v) < 0 else 360 - a

    def intersect(self, A, B, C, D):
        return self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D)

    @staticmethod
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    @staticmethod
    def get_roi_center(roi):
        return np.array([roi.x + roi.width / 2, roi.y + roi.height / 2])


class Application(BaseApplication):
    line_cross_counter = LineCrossingCounter()

    def setup_pipeline(self, oak: OakCamera):
        color = oak.create_camera(source='color', fps=15, encode='h264')
        detection_nn = oak.create_nn(model='yolov8n_coco_640x352', input=color, tracker=True)
        detection_nn.config_nn(resize_mode='stretch')
        detection_nn.config_tracker(tracker_type=dai.TrackerType.SHORT_TERM_IMAGELESS,
                                    track_labels=[0],  # track cars and motocycles only
                                    assignment_policy=dai.TrackerIdAssignmentPolicy.UNIQUE_ID,
                                    max_obj= 10,
                                    threshold=0.7,
                                    apply_tracking_filter=True,
                                    forget_after_n_frames=15)
        
        self.line_cross_counter.live_view = LiveView.create(device=oak, component=color, name='Line Cross stream', unique_key=f'line_cross_stream',
                                                            manual_publish=True)
        
        oak.sync([color.out.encoded, detection_nn.out.tracker], self.line_cross_counter.process_packets)

if __name__ == "__main__":
    app = Application()
    app.run()
