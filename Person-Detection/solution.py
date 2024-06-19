import depthai as dai
import numpy as np
import requests
import time
import cv2 as cv
from enum import IntEnum

from depthai_sdk import OakCamera
from numpy import linalg as LA
from robothub import LiveView
from robothub.application import BaseApplication


class States(IntEnum):
    IDLE = -1
    IN_LINE = 0
    BEING_ATTENDED = 1

class DetectionParams:
    def __init__(self):
        self.id = 0
        self.prev_time = 0.0
        self.in_line_time = 0.0
        self.being_attended_time = 0.0
        self.current_spent_time = 0.0
        self.state = States.IDLE 

class PersonCounter:
    def __init__(self):
        self.detection_dict : dict = {}
        self.removed_detections : list = []
        self.binary_roi = None
        self.attended_roi = None
        self.elapsed_time = 0.0
        self.prev_time = time.time()
        self.TIME_THRESHOLD = 10.0
        self.averages = [0.0,0.0,0.0]

    def process_packets(self, packets):
        color_packet = packets['color']
        nn_packet = packets['3_out;0_video']

        tracklets = nn_packet.daiTracklets.tracklets
        detections = nn_packet.detections

        for key in list(self.detection_dict.keys()):
            if key not in [tracklet.id for tracklet in tracklets] and self.detection_dict[key].state == States.BEING_ATTENDED:
                self.removed_detections.append(self.detection_dict.pop(key))

        for detection, tracklet in zip(detections, tracklets):
            tracklet_id = tracklet.id
            tracklet_status = tracklet.status

            ### Get relevant information from the tracklet
            bbox = [*detection.top_left, *detection.bottom_right]
            h, w = nn_packet.frame.shape[:2]
            normalized_roi = tracklet.roi.denormalize(w, h)
            current_position = self.get_roi_center(normalized_roi)

            # Check if the current_position lies within the binary_roi
            if self.binary_roi is not None and not self.binary_roi[int(current_position[1]), int(current_position[0])]:
                if tracklet_id in self.detection_dict.keys():
                    self.detection_dict.pop(tracklet_id)
                continue
        
            ### Analyze the tracklet
            if tracklet_id not in self.detection_dict.keys():
                self.detection_dict[tracklet_id] = DetectionParams()
                self.detection_dict[tracklet_id].id = tracklet_id
                self.detection_dict[tracklet_id].prev_time = time.time()
                self.detection_dict[tracklet_id].current_spent_time = 0.0
                self.detection_dict[tracklet_id].state = States.IN_LINE
            elif self.detection_dict[tracklet_id].state == States.IN_LINE and self.attended_roi is not None and not self.attended_roi[int(current_position[1]), int(current_position[0])]:
                 self.detection_dict[tracklet_id].state = States.BEING_ATTENDED
            elif tracklet_status == dai.Tracklet.TrackingStatus.REMOVED:
                self.removed_detections.append(self.detection_dict.pop(tracklet_id))
                continue
            
            ### Calculate the time spent in line and being attended
            if self.detection_dict[tracklet_id].state == States.IN_LINE:
                self.detection_dict[tracklet_id].in_line_time += float(time.time() - self.detection_dict[tracklet_id].prev_time)
            elif self.detection_dict[tracklet_id].state == States.BEING_ATTENDED:
                self.detection_dict[tracklet_id].being_attended_time += float(time.time() - self.detection_dict[tracklet_id].prev_time)  

            self.detection_dict[tracklet_id].current_spent_time += float(time.time() - self.detection_dict[tracklet_id].prev_time)
            self.detection_dict[tracklet_id].prev_time = time.time()    
                
            ident = "     " + str(tracklet_id)

            self.live_view.add_rectangle(rectangle=bbox, label=detection.label + ident)
        
        self.calc_averages()

        self.live_view.add_text("Averages",
                                coords=(100, 100),size=40,color=(50,50,50),thickness=25)
        self.live_view.add_text(f'Total: {self.averages[0]}, In line: {self.averages[1]}, Attended: {self.averages[2]}, Detections: {len(self.removed_detections)}',
                                coords=(100, 100),size=40,color=(50,50,50),thickness=20)
        self.live_view.add_text(f"Detections: {[detection.id for detection in self.detection_dict.values()]}",
                                coords=(100, 200),size=40,color=(50,50,50),thickness=10)
        self.live_view.add_text(f"Seconds: {[f'{detection.current_spent_time:.2f}' for detection in self.detection_dict.values()]}",
                                coords=(100, 240),size=40,color=(50,50,50),thickness=10)
        self.live_view.add_text(f"States: {[detection.state for detection in self.detection_dict.values()]}", 
                                coords=(100, 280),size=40,color=(50,50,50),thickness=10)
        
        self.live_view.publish(color_packet.frame)

        self.elapsed_time = time.time() - self.prev_time

    def calc_averages(self):
        if (self.elapsed_time) / 60 < self.TIME_THRESHOLD:
            return
        
        self.elapsed_time = 0.0
        self.prev_time = time.time()

        avg_spent_time = 0.0
        avg_line_time = 0.0
        avg_attended_time = 0.0

        for detection in self.removed_detections:
            avg_spent_time += detection.current_spent_time
            avg_line_time += detection.in_line_time
            avg_attended_time += detection.being_attended_time
        
        avg_spent_time /= len(self.removed_detections)
        avg_line_time /= len(self.removed_detections)
        avg_attended_time /= len(self.removed_detections)

        self.averages = [avg_spent_time, avg_line_time, avg_attended_time]

        ### INSERT TO DATABASE

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
    person_counter = PersonCounter()

    def setup_pipeline(self, oak: OakCamera):
        color = oak.create_camera(source='color', fps=15, encode='h264')
        detection_nn = oak.create_nn(model='z', input=color, tracker=True)
        detection_nn.config_nn(resize_mode='stretch')
        detection_nn.config_tracker(tracker_type=dai.TrackerType.SHORT_TERM_IMAGELESS,
                                    track_labels=[0],  # track cars and motocycles only
                                    assignment_policy=dai.TrackerIdAssignmentPolicy.UNIQUE_ID,
                                    max_obj= 10,
                                    threshold=0.7,
                                    apply_tracking_filter=True,
                                    forget_after_n_frames=15)
        
        self.person_counter.live_view = LiveView.create(device=oak, component=color, name='Line Cross stream', unique_key=f'line_cross_stream',
                                                            manual_publish=True)
        
        oak.sync([color.out.encoded, detection_nn.out.tracker], self.person_counter.process_packets)

if __name__ == "__main__":
    app = Application()
    app.run()
