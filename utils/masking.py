import cv2
import numpy as np
import pathlib

pts = []
base_path = pathlib.Path(__file__).parent.resolve()

video_name = str(input("Enter the name to the video: "))
video_path = base_path / video_name
print(video_path)
video = cv2.VideoCapture(str(video_path))
ret, frame = video.read()

def select_point(event, x, y, flags, param):
    global frame
    global pts
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))
        cv2.circle(frame, (x, y), 8, (0, 0, 255), -1) 

# Manually generate a binary mask for the video with clicked points
filename = str(input("Enter the filename for the mask: "))
filepath = base_path / filename
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):
        pts = []
        ret, frame = video.read()
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.imshow("Mask", mask)
    elif key == ord('s'):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, pts, (255, 255, 255))
        # print(mask.shape)
        # Cconver mask to binary image
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        pts = []

        # # Save the mask as numpy array
        np.save(str(filepath), mask)

        cv2.imshow("Mask", mask)
    elif key == ord('c'):
        pts = []
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.imshow("Mask", mask)
    elif key == ord('l'):
        # Find contours
        print (pts)
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(mask, [pts], isClosed=True, color=(255, 255, 255), thickness=2)
        #gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("Gray", gray)
        thresh = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1]

        # get the (largest) contour
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        big_contour = max(contours, key=cv2.contourArea)

        # draw white filled contour on black background
        result = np.zeros_like(mask)
        cv2.drawContours(result, [big_contour], 0, (255,255,255), cv2.FILLED)

        result = cv2.threshold(result, 128, 255, cv2.THRESH_BINARY)[1]
        # # Save the mask as numpy array
        np.save(str(filepath), result)
        print(result.shape)
        cv2.imshow("Mask", result)
        pts = []
    
    if not ret:
        break

    try:
        mask = np.load("mask.npy")
        cv2.imshow("Mask", mask)
    except:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.imshow("Frame", frame)
    cv2.setMouseCallback("Frame", select_point)

    if key == ord('q'):
        break
