import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET



def drawRoBBs(fr=None, ann_path=None):
    '''Draws rotated bounding boxes to the corresponding image and returns the annotated image
    This function can be called in two separate ways. 
    
    Args:
    fr: RGB Image in uint8 format
    ann_dir: Path of the annotation file
    
    returns:
    anoted RGB image in uint8 format  
    '''
    fr_out = fr.copy()
    if os.path.exists(ann_path):
        objects = np.loadtxt(ann_path, dtype=str)
        for obj in objects:
            # Get box properties
            x = float(obj[1])*2048
            y = float(obj[2])*2048
            w = float(obj[3])*2048
            h = float(obj[4])*2048
            angle = float(obj[6])*np.pi/180

            c, s = np.cos(angle), np.sin(angle)
            R = np.asarray([[-c, -s], [s, -c]])
            pts = np.asarray([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
            rot_pts = []
            for pt in pts:
                rot_pts.append(([x, y] + pt @ R).astype(int))

            contours = np.array([rot_pts[0], rot_pts[1], rot_pts[2], rot_pts[3]])
            cv2.polylines(fr_out, pts=[contours], color=(0, 0, 255), isClosed=True, thickness=3)
    return fr_out
