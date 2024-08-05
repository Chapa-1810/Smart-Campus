import cv2 as cv
import numpy as np
import os
import pathlib

BASE_PATH = pathlib.Path(__file__).parent.resolve().absolute()

attendee_mask_file = "attendee_mask.npy"
line_mask_file = "line_mask.npy"
demo_video = "demo.mp4"
output_video = "recorded_demo.mp4"

def load_mask(filepath):
  print(filepath)
  try:
    mask = np.load(filepath)
    return mask
  except:
    return None

def main():
  mask_line = load_mask(os.path.join(BASE_PATH, line_mask_file))
  if mask_line is None:
    print("Line mask not found")
    return
  
  attendee_mask = load_mask(os.path.join(BASE_PATH, attendee_mask_file))
  if attendee_mask is None:
    print("Attende mask not found")
    return
  
  video = cv.VideoCapture(os.path.join(BASE_PATH, "../" + demo_video))
  ret, frame = video.read()

  if not ret:
    print("Error reading video")

  alpha = 0.3
  result = cv.VideoWriter(output_video,  
                         cv.VideoWriter_fourcc(*'mp4v'), 
                         10, frame.shape[:2]) 
  while ret:
    overlay = frame.copy()
    output = frame.copy()

    # apply the mask
    overlay[mask_line == 255] = (0, 255, 0)
    overlay[attendee_mask == 255] = (0, 0, 255)

    # apply the overlay
    cv.addWeighted(overlay, alpha, output, 1 - alpha,
		0, output)

    cv.imshow("Output", output)
    result.write(output)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
      break
    ret, frame = video.read()


if __name__ == "__main__":
  main()