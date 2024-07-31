import numpy as np
import cv2
import os

def load_mask(mask_name):
    try:
        mask = np.load(mask_name)
        return mask
    except:
        return None
    
def main():
    mask_name = "line_mask.npy"
    mask_line = load_mask(mask_name)
    if mask_line is not None:
        print(mask_line.shape)
        cv2.imshow("Mask1", mask_line)
        cv2.waitKey(0)
    else:
        print("Mask not found")

    mask_name = "attendee_mask.npy"
    mask_line = load_mask(mask_name)

    if mask_line is not None:
        print(mask_line.shape)
        cv2.imshow("Mask1", mask_line)
        cv2.waitKey(0)
    else:
        print("Mask not found")

if __name__ == "__main__":
    main()