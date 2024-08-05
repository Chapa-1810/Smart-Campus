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
        color_packet = packets['color']
        nn_packet = packets['3_out;0_video']

        tracklets = nn_packet.daiTracklets.tracklets
        detections = nn_packet.detections
        ids = [tracklet.id for tracklet in tracklets]
        self.attendees = [self.detection_dict[detection_key] for detection_key in list(self.detection_dict.keys()) if self.detection_dict[detection_key].state == States.ATTENDEE]

        ### Remove tracklets that are not in the current frame
        for key in list(self.detection_dict.keys()):
            deleted = DetectionParams()
            if key not in ids:
                deleted = self.detection_dict.pop(key)
            if deleted.state == States.IN_LINE or deleted.state == States.BEING_ATTENDED:
                self.removed_detections.append(deleted.id)
            
        ### Process the tracklets in the current frame
        for detection, tracklet in zip(detections, tracklets):
            tracklet_id = tracklet.id
            tracklet_status = tracklet.status # dai.Tracklet.TrackingStatus 

            ### Get relevant information from the tracklet
            bbox = [*detection.top_left, *detection.bottom_right]
            h, w = nn_packet.frame.shape[:2]
            normalized_roi = tracklet.roi.denormalize(w, h)
            current_position = self.get_roi_center(normalized_roi)
            print(f"Tracklet {tracklet_id} is currently at position {current_position}")

            ### Analyze the tracklet state
            if tracklet_id not in self.detection_dict.keys(): # New tracklet
                self.detection_dict[tracklet_id] = DetectionParams()
                self.detection_dict[tracklet_id].id = tracklet_id
                self.detection_dict[tracklet_id].prev_time = time.time()
                self.detection_dict[tracklet_id].bbox = bbox
            elif self.detection_dict[tracklet_id].state == States.IDLE: # Tracklet is idle, therefore check if it is in line or being attended
                line_IOU = self.get_IOU(bbox, self.line_mask)
                attendee_IOU = self.get_IOU(bbox, self.attention_mask)
                print(f"Tracklet {tracklet_id} --- line IOU: {line_IOU}  attendee IOU: {attendee_IOU}")
                if line_IOU > self.IOU_THRESHOLD and line_IOU > attendee_IOU: # Tracklet is in line if its IOU is bigger than the threshold and bigger than the attendee IOU
                    self.detection_dict[tracklet_id].state = States.IN_LINE
                elif attendee_IOU > self.IOU_THRESHOLD and attendee_IOU > line_IOU: # Tracklet is being attended if its IOU is bigger than the threshold and bigger than the line IOU
                    self.detection_dict[tracklet_id].state = States.ATTENDEE
            elif self.detection_dict[tracklet_id].state == States.IN_LINE and False: #TODO: Check if the tracklet is being attended with a computer vision algorithm
                for attendee in self.attendees:
                    attendee_bbox = attendee.bbox
                    if self.get_IOU_bboxes(bbox, attendee_bbox):
                        self.detection_dict[tracklet_id].state = States.BEING_ATTENDED
            elif tracklet_status == dai.Tracklet.TrackingStatus.REMOVED: # Tracklet is removed becuase tracklet status is removed
                removed_tracklet = self.detection_dict.pop(tracklet_id) 
                if removed_tracklet.state == States.IN_LINE:
                    self.removed_detections.append(removed_tracklet.id)
                continue
            ### Calculate the time spent in line and being attended
            if self.detection_dict[tracklet_id].state == States.IN_LINE: 
                self.detection_dict[tracklet_id].in_line_time += float(time.time() - self.detection_dict[tracklet_id].prev_time)
            elif self.detection_dict[tracklet_id].state == States.BEING_ATTENDED:
                self.detection_dict[tracklet_id].being_attended_time += float(time.time() - self.detection_dict[tracklet_id].prev_time)  

            self.detection_dict[tracklet_id].current_spent_time += float(time.time() - self.detection_dict[tracklet_id].prev_time) # Calculate the time spent globally for each tracklet
            self.detection_dict[tracklet_id].prev_time = time.time()    
                
            ident = "     " + str(tracklet_id)
            #print (vars(self.detection_dict[tracklet_id]))

            self.live_view.add_rectangle(rectangle=bbox, label=detection.label + ident)
        
        self.calc_averages() # Calculate the averages of the time spent in line and being attended

        self.live_view.add_text(f'Total: {self.averages[0]}, In line: {self.averages[1]}, Attended: {self.averages[2]}, Detections: {len(self.removed_detections)}',
                                coords=(100, 100),size=40,color=(50,50,50),thickness=20)
        self.live_view.add_text(f"Detections: {[detection.id for detection in self.detection_dict.values()]}",
                                coords=(100, 200),size=40,color=(50,50,50),thickness=10)
        self.live_view.add_text(f"Seconds: {[f'{detection.current_spent_time:.2f}' for detection in self.detection_dict.values()]}",
                                coords=(100, 240),size=40,color=(50,50,50),thickness=10)
        self.live_view.add_text(f"States: {[detection.state for detection in self.detection_dict.values()]}", 
                                coords=(100, 280),size=40,color=(50,50,50),thickness=10)
        self.live_view.publish(color_packet.frame)
 
        # self.elapsed_time = time.time() - self.prev_time # Calculate the elapsed time since the last calculation
        self.elapsed_time += float(time.time() - self.prev_time) # Calculate the elapsed time since the last calculation
        self.prev_time = time.time()

    # Redo function 

    def calc_averages(self):
        if (self.elapsed_time / 60) < self.TIME_THRESHOLD or len(self.removed_detections) <= 0: # If the elapsed time is less than the threshold or there are no detections,
            return
        
        self.elapsed_time = 0.0
        # self.prev_time = time.time()

        avg_attended_spent_time = 0.0
        avg_attended_line_time = 0.0
        avg_attended_attended_time = 0.0
        in_line = 0
        attendee = 0

        for detection in self.removed_detections: # Calculate the averages of the time spent in line and being attended
            if detection.state == States.IN_LINE:
                avg_attended_spent_time += detection.current_spent_time
                avg_attended_line_time += detection.in_line_time
                avg_attended_attended_time += detection.being_attended_time
                in_line += 1
            elif detection.state == States.ATTENDEE:
                avg_attended_spent_time += detection.current_spent_time
                avg_attended_line_time += detection.in_line_time
                avg_attended_attended_time += detection.being_attended_time
                attendee += 1
        
        
        avg_attended_spent_time /= in_line
        avg_attended_line_time /= in_line
        avg_attended_attended_time /= in_line
        self.averages = [avg_attended_spent_time, avg_attended_line_time, avg_attended_attended_time]

        self.removed_detections = []


        ### TODO: INSERT TO DATABASE

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
