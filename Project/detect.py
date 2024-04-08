from PIL import Image, ImageDraw
import math
import os
import sys
from ultralytics import YOLO
import cv2
import torch
from utils import drawRoBBs
import numpy as np
from matplotlib.path import Path
import matplotlib.patches as patches
import shapely.geometry
import shapely.affinity
from matplotlib import pyplot
from descartes import PolygonPatch

def rotate_img(input_path, degrees):

    img=Image.open(input_path)
    rotated=img.rotate(degrees,expand=False)
    return rotated

def draw_inverted_triangle(img, degrees):
    width, height = img.size
    half_angle_rad = math.radians(degrees / 2)
    triangle_half_width = (height / 2) * math.tan(half_angle_rad)
    apex_point = (width / 2, height / 2) 
    left_base_point = (width/2-triangle_half_width, 0)  
    right_base_point = (width/2+triangle_half_width, 0)  

    mask = Image.new('L', (width, height), 0) 
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.polygon([apex_point, left_base_point, right_base_point], fill=255)

    black_fill = Image.new('RGB', (width, height), (0, 0, 0))
    res = Image.composite(img, black_fill, mask)
    return res

def rotate_point(x, y, angle_degrees, center=(0.5, 0.5)):
    angle_rad = np.radians(angle_degrees)
  
    x_translated = x - center[0]
    y_translated = y - center[1]
 
    x_rotated = (x_translated * np.cos(angle_rad)) - (y_translated * np.sin(angle_rad))
    y_rotated = (x_translated * np.sin(angle_rad)) + (y_translated * np.cos(angle_rad))

    x_final = x_rotated + center[0]
    y_final = y_rotated + center[1]
    
    return (x_final, y_final)
  
def process_file(input_path, output_path):
    with open(input_path, 'r') as file:
        lines = file.readlines()

    modified_lines = []
    for line in lines:
        parts = line.split()
        x, y = map(float, parts[1:3])
        angle_degrees = float(parts[-1])
        x_rotated, y_rotated = rotate_point(x, y, angle_degrees)
    
        modified_line = f"{parts[0]} {x_rotated:.6f} {y_rotated:.6f} " + " ".join(parts[3:-1]) + f" {parts[-1]}\n"
        modified_lines.append(modified_line)

    with open(output_path, 'w') as file:
        file.writelines(modified_lines)
def count_lines(filepath):

    with open(filepath, 'r') as file:
        return sum(1 for line in file)

def detect(image_path,yolo_ver, rotation_times,mid_ann_path,final_ann_path, output_path):
  degrees=360/rotation_times
  results=[]
  
  for times in range (rotation_times):
    results.append(draw_inverted_triangle(rotate_img(image_path,degrees*times),degrees))
  
  model = YOLO(yolo_ver) #can be swapped with yolov5su and conf=0.59 for faster performance  or yolov5lu with conf=0.55 for more confidence
  res=model.predict(results,classes=[0],conf=0.25,device='cuda:0')
  i=0
  
  if os.path.exists(mid_ann_path):
    os.remove(mid_ann_path)
  with open(mid_ann_path, "w") as f:
      pass
  for r in res:
    #r.save(f'mid_points/{i}.jpg')
    lines_before = count_lines(mid_ann_path)
    r.save_txt(mid_ann_path, save_conf=True)
    lines_after = count_lines(mid_ann_path)
    
    added_lines_count = lines_after - lines_before
    
    if added_lines_count > 0:
        with open(mid_ann_path, 'r') as file:
            lines = file.readlines()
        for index in range(-added_lines_count, 0):
            lines[index] = lines[index].strip() + f" {i*degrees}\n"
        
        with open(mid_ann_path, 'w') as file:
            file.writelines(lines)
    
    i += 1
  
  if os.path.exists(final_ann_path):
    os.remove(final_ann_path)
  with open(final_ann_path, "w") as f:
      pass
  process_file(mid_ann_path,final_ann_path)
  fr = cv2.imread(image_path)
        
  fr_bb = drawRoBBs(fr=fr, ann_path=final_ann_path)
  cv2.imwrite(output_path, fr_bb)


class RotatedRect:
    def __init__(self, param):
        self.cx = param[1]
        self.cy = param[2]
        self.w = param[3]
        self.h = param[4]
        self.angle = param[6]

    def get_contour(self):
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w/2.0, -h/2.0, w/2.0, h/2.0)
        rc = shapely.affinity.rotate(c, self.angle)
        return shapely.affinity.translate(rc, self.cx, self.cy)

    def intersection(self, other):
        return self.get_contour().intersection(other.get_contour())

def calculate_iou(param1,param2):
  
  r1 = RotatedRect(param1)
  r2 = RotatedRect(param2)
  return  r1.intersection(r2).area/r1.get_contour().area

print(calculate_iou(any,any))

#main program         
if __name__ == "__main__":
  if len(sys.argv) == 2:
      image_path = sys.argv[1]
      print(f"Processing file: {image_path}")

  else:
      print("Usage: python3 detect.py <image_file>")
      sys.exit(1) 
   
  rotation_times=10
  yolo_ver='yolov5su.pt'
  mid_ann_path='mid_points/boxes.txt' 
  final_ann_path='results/annotation.txt'
  output_path='results/result.jpg'
  for i in range(1,1000,50):
    input_path=f'HABBOF/Lab2/{i:06}.jpg'
    output_path=f'results/result{i:06}.jpg'
    detect(input_path,yolo_ver,rotation_times,mid_ann_path,final_ann_path,output_path)
