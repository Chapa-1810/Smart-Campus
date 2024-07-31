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
    ATTENDEE = 2

class DetectionParams:
    def __init__(self):
        self.id = 0
        self.prev_time = 0.0
        self.in_line_time = 0.0
        self.being_attended_time = 0.0
        self.current_spent_time = 0.0
        self.state = States.IDLE 
        self.bbox = []

class PersonCounter:
    def __init__(self):
        self.detection_dict : dict = {}
        self.removed_detections : list = []
        self.line_mask = None
        self.attention_mask = None
        self.attendee_mask = None
        self.elapsed_time = 0.0
        self.prev_time = time.time()
        self.TIME_THRESHOLD = 10.0
        self.averages = [0.0,0.0,0.0]
        self.load_masks()
        self.IOU_THRESHOLD = 0.5
        self.IOU_THRESHOLD_ATTENDED = 0.1

    def load_masks(self):
        self.line_mask = np.load("line_mask.npy")
        self.attention_mask = np.load("attention_mask.npy")
        self.attendee_mask = np.load("attendee_mask.npy")

    def get_IOU(self, bbox, mask):
        x1, y1, x2, y2 = bbox
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        # Generate a white mask of the bbox dimensions
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            return 0.0

        object_bbox_mask = np.ones((y2 - y1, x2 - x1), dtype=np.uint8)

        # Get area mask of the object in real mask
        object_mask = mask[y1:y2, x1:x2]

        # Calculate the intersection and union
        intersection = np.logical_and(object_bbox_mask, object_mask)
        union = np.logical_or(object_bbox_mask, object_mask)
        iou = np.sum(intersection) / np.sum(union)
        return iou
    
    def get_IOU_bboxes(self, bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        x3, y3, x4, y4 = bbox2
        x3 = int(x3)
        y3 = int(y3)
        x4 = int(x4)
        y4 = int(y4)

        # Get IOU between the two bboxes
        intersection = (min(x2, x4) - max(x1, x3)) * (min(y2, y4) - max(y1, y3))

        if intersection <= 0:
            return 0.0
        
        union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - intersection
        iou = intersection / union

        return iou > self.IOU_THRESHOLD_ATTENDED

    def process_packets(self, packets):
        LINE1, LINE2 = [4, 967],[1033, 512]
        color_packet = packets['color']
        nn_packet = packets['3_out;0_video']

        tracklets = nn_packet.daiTracklets.tracklets
        detections = nn_packet.detections
        ids = [tracklet.id for tracklet in tracklets]
        self.attendees = [self.detection_dict[detection_key] for detection_key in list(self.detection_dict.keys()) if self.detection_dict[detection_key].state == States.ATTENDEE]

        for key in list(self.detection_dict.keys()):
            deleted = DetectionParams()
            if key not in ids:
                deleted = self.detection_dict.pop(key)
            if deleted.state == States.IN_LINE or deleted.state == States.BEING_ATTENDED:
                self.removed_detections.append(deleted.id)
            

        for detection, tracklet in zip(detections, tracklets):
            tracklet_id = tracklet.id
            tracklet_status = tracklet.status

            ### Get relevant information from the tracklet
            bbox = [*detection.top_left, *detection.bottom_right]
            h, w = nn_packet.frame.shape[:2]
            normalized_roi = tracklet.roi.denormalize(w, h)
            current_position = self.get_roi_center(normalized_roi)
            print(f"Tracklet {tracklet_id} is currently at position {current_position}")

            ### Analyze the tracklet
            if tracklet_id not in self.detection_dict.keys():
                self.detection_dict[tracklet_id] = DetectionParams()
                self.detection_dict[tracklet_id].id = tracklet_id
                self.detection_dict[tracklet_id].prev_time = time.time()
                self.detection_dict[tracklet_id].bbox = bbox
            elif self.detection_dict[tracklet_id].state == States.IDLE:
                line_IOU = self.get_IOU(bbox, self.line_mask)
                attendee_IOU = self.get_IOU(bbox, self.attention_mask)
                print(f"Tracklet {tracklet_id} --- line IOU: {line_IOU}  attendee IOU: {attendee_IOU}")
                if line_IOU > self.IOU_THRESHOLD and line_IOU > attendee_IOU:
                    self.detection_dict[tracklet_id].state = States.IN_LINE
                elif attendee_IOU > self.IOU_THRESHOLD and attendee_IOU > line_IOU:
                    self.detection_dict[tracklet_id].state = States.ATTENDEE
            elif self.detection_dict[tracklet_id].state == States.IN_LINE and False: #TODO: Check if the tracklet is being attended
                for attendee in self.attendees:
                    attendee_bbox = attendee.bbox
                    if self.get_IOU_bboxes(bbox, attendee_bbox):
                        self.detection_dict[tracklet_id].state = States.BEING_ATTENDED
            elif tracklet_status == dai.Tracklet.TrackingStatus.REMOVED:
                removed_tracklet = self.detection_dict.pop(tracklet_id) 
                if removed_tracklet.state == States.IN_LINE:
                    self.removed_detections.append(removed_tracklet.id)
                continue
            ### Calculate the time spent in line and being attended
            if self.detection_dict[tracklet_id].state == States.IN_LINE:
                self.detection_dict[tracklet_id].in_line_time += float(time.time() - self.detection_dict[tracklet_id].prev_time)
            elif self.detection_dict[tracklet_id].state == States.BEING_ATTENDED:
                self.detection_dict[tracklet_id].being_attended_time += float(time.time() - self.detection_dict[tracklet_id].prev_time)  

            self.detection_dict[tracklet_id].current_spent_time += float(time.time() - self.detection_dict[tracklet_id].prev_time)
            self.detection_dict[tracklet_id].prev_time = time.time()    
                
            ident = "     " + str(tracklet_id)
            #print (vars(self.detection_dict[tracklet_id]))

            self.live_view.add_rectangle(rectangle=bbox, label=detection.label + ident)
        
        self.calc_averages()

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

    # Redo function 

    def calc_averages(self):
        if (self.elapsed_time) / 60 < self.TIME_THRESHOLD or len(self.removed_detections) == 0:
            return
        
        self.elapsed_time = 0.0
        self.prev_time = time.time()

        avg_spent_time = 0.0
        avg_line_time = 0.0
        avg_attended_time = 0.0

        for detection in self.removed_detections:
            if detection.state == States.IN_LINE:
                avg_spent_time += detection.current_spent_time
                avg_line_time += detection.in_line_time
                avg_attended_time += detection.being_attended_time
        
        
        avg_spent_time /= len(self.removed_detections)
        avg_line_time /= len(self.removed_detections)
        avg_attended_time /= len(self.removed_detections)
        self.removed_detections = []
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
        color = oak.create_camera(source='color', fps=15, encode='h264', resolution='1080p',)
        detection_nn = oak.create_nn(model='yolov8n_coco_640x352', input=color, tracker=True)
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
