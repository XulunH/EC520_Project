from PIL import Image, ImageDraw, ImageStat, ImageEnhance
import math
import os
import sys
from ultralytics import YOLO
import cv2
from utils import drawRoBBs
import numpy as np
import shapely.geometry
import shapely.affinity

def rotate_img(img, degrees):
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

def count_lines(filepath):

    with open(filepath, 'r') as file:
        return sum(1 for line in file)
    
def mid_ann_processing(res, mid_ann_path, save_mid_points, degrees):
  i=0
  if os.path.exists(mid_ann_path):
     os.remove(mid_ann_path)
  with open(mid_ann_path, "w") as f:
      pass
  for r in res:
    if(save_mid_points):
     r.save(f'mid_points/{i}.jpg')
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

def rotate_point_cw(x, y, angle_degrees, center):
    angle_rad = np.radians(angle_degrees)
  
    x_translated = x - center[0]
    y_translated = y - center[1]
 
    x_rotated = (x_translated * np.cos(angle_rad)) - (y_translated * np.sin(angle_rad))
    y_rotated = (x_translated * np.sin(angle_rad)) + (y_translated * np.cos(angle_rad))

    x_final = x_rotated + center[0]
    y_final = y_rotated + center[1]
    
    return (x_final, y_final)

def final_ann_processing(input_path, output_path):
    if os.path.exists(output_path):
      os.remove(output_path)
    with open(output_path, "w") as f:
      pass
    with open(input_path, 'r') as file:
        lines = file.readlines()

    modified_lines = []
    for line in lines:
        parts = line.split()
        x, y = map(float, parts[1:3])
        angle_degrees = float(parts[-1])
        x_rotated, y_rotated = rotate_point_cw(x, y, angle_degrees,[0.5,0.5])
    
        modified_line = f"{parts[0]} {x_rotated:.6f} {y_rotated:.6f} " + " ".join(parts[3:-1]) + f" {parts[-1]}\n"
        modified_lines.append(modified_line)

    with open(output_path, 'w') as file:
        file.writelines(modified_lines)



class RotatedRect:
    def __init__(self, param):
        self.cx = param[1]*2048
        self.cy = param[2]*2048
        self.w = param[3]*2048
        self.h = param[4]*2048
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
  return  r1.intersection(r2).area/max(r1.get_contour().area,r2.get_contour().area)

def NMS(final_ann_path, iou_threshold):
    with open(final_ann_path, 'r') as file:
        array = [[float(num) for num in line.split()] for line in file]
    to_delete = set()

    for i in range(len(array)):
        for j in range(i + 1, len(array)):
            if array[i][-1] != array[j][-1] and calculate_iou(array[i], array[j]) > iou_threshold:
                
                loser_index = i if array[i][5] < array[j][5] else j
                to_delete.add(loser_index)
    for index in sorted(to_delete, reverse=True):
        del array[index]
    with open(final_ann_path, 'w') as file:
      for row in array:
        formatted_row = ' '.join(map(str, row)) + '\n'
        file.write(formatted_row)

def adjust_image_brightness(img,conf):
   im = img.convert('L')
   stat = ImageStat.Stat(im)
   brightness_threshold=90
   if(stat.rms[0]<brightness_threshold):
      img_enhancer = ImageEnhance.Brightness(img)
      factor = 3
      enhanced_output = img_enhancer.enhance(factor)
      return enhanced_output, conf-0.25
   else:
      return img, conf






def detect(yolo_ver, device, rotation_times,conf,iou_within_one_subsample,iou_threshold_between_subsamples,save_mid_points, image_path,mid_ann_path,final_ann_path, output_path):
  degrees=360/rotation_times
  results=[]
  img=Image.open(image_path)
  adjusted_img,conf=adjust_image_brightness(img,conf)
  print(conf)
  for times in range (rotation_times):
    results.append(draw_inverted_triangle(rotate_img(adjusted_img,degrees*times),degrees))
  
  model = YOLO(yolo_ver) #can be swapped with yolov5su and conf=0.59 for faster performance  or yolov5lu with conf=0.55 for more confidence
  res=model.predict(results,classes=[0],conf=conf,iou=iou_within_one_subsample,device=device)
  
  mid_ann_processing(res, mid_ann_path, save_mid_points, degrees)
  final_ann_processing(mid_ann_path,final_ann_path)

  fr = cv2.imread(image_path) 
  if(count_lines(final_ann_path)==0):
    cv2.imwrite(output_path, fr)
  elif(iou_threshold_between_subsamples==1 or count_lines(final_ann_path)==1):
    fr_bb = drawRoBBs(fr=fr, ann_path=final_ann_path)
    cv2.imwrite(output_path, fr_bb)
  else:    
    NMS(final_ann_path,iou_threshold_between_subsamples)
    fr_bb = drawRoBBs(fr=fr, ann_path=final_ann_path)
    cv2.imwrite(output_path, fr_bb)


#main program         
if __name__ == "__main__":
  
  rotation_times=10
  conf=0.7
  iou_within_one_subsample=0.33
  iou_threshold_between_subsamples=0.35
  save_mid_points= False
  yolo_ver='best.pt'
  device='cuda:0'
  mid_ann_path='mid_points/boxes.txt' 
  final_ann_path='results/final_annotation.txt'
  HABBOF_path='C:/Users/huang/OneDrive/Desktop/520_Image/HABBOF/'
  

  if len(sys.argv) == 2:
      image_path = HABBOF_path+sys.argv[1]
      save_mid_points= True
      output_path='results/result.jpg'
      detect(yolo_ver,device,rotation_times,conf,iou_within_one_subsample,iou_threshold_between_subsamples,save_mid_points, image_path, mid_ann_path, final_ann_path,output_path)

  else:
    for i in range(1,1000,25):
      image_path=HABBOF_path+f'Lab1/{i:06}.jpg'
      output_path=f'results/trained/result{i:06}.jpg'
      detect(yolo_ver,device,rotation_times,conf,iou_within_one_subsample,iou_threshold_between_subsamples,save_mid_points, image_path, mid_ann_path, final_ann_path,output_path) 
   
  
  
 
  