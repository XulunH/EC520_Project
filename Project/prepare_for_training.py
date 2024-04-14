
import numpy as np
import math
from detect import  draw_inverted_triangle, rotate_img,drawRoBBs
from PIL import Image
import cv2

def rotate_point_ccw(x, y, angle_degrees, center):
    angle_rad = np.radians(angle_degrees)
  
    x_translated = x - center[0]
    y_translated = y - center[1]
 
    x_rotated = (x_translated * np.cos(angle_rad)) + (y_translated * np.sin(angle_rad))
    y_rotated = -(x_translated * np.sin(angle_rad)) + (y_translated * np.cos(angle_rad))

    x_final = x_rotated + center[0]
    y_final = y_rotated + center[1]
    
    return (x_final, y_final)
def convert_annotation(unit_degrees,input_base_path, input_txt_name,output_path_labels):
    f=open(input_base_path+input_txt_name,"r")
    lines=f.readlines()
    f.close()
    bbs=[]
    subimage_indexes=[]
    for line in lines:
        parts=line.split()
        x,y,w,h=map(float,parts[1:5])
        bbs.append((0,x,y,w,h))

    for bb in bbs:
        bb=list(bb)
        angleinrad=math.atan2(-bb[2]+1024,bb[1]-1024)
        angle=90-math.degrees(angleinrad)
        if -90<=angle<0:
            angle+=360
        sub_index=int(((angle+unit_degrees/2)//unit_degrees)%(360/unit_degrees))
        bb[1],bb[2]=rotate_point_ccw(bb[1],bb[2],sub_index*unit_degrees,[1024,1024])
        bb=[x/2048 for x in bb]
        formatted_row = f"{int(bb[0])} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f} {bb[4]:.6f}\n"
        if(sub_index not in subimage_indexes):
           subimage_indexes.append(sub_index)
           with open(output_path_labels+f'{sub_index}'+input_txt_name,"w") as f:
             f.write(formatted_row)
        else:
            with open(output_path_labels+f'{sub_index}'+input_txt_name,"a") as f:
             f.write(formatted_row)
    return subimage_indexes
        
def generate_subimages(unit_degrees,subimage_indexes,input_base_path,input_image_name,output_path_img):
    img=Image.open(input_base_path+input_image_name)
    for sub_index in subimage_indexes:
        subimage=(draw_inverted_triangle(rotate_img(img,unit_degrees*sub_index),unit_degrees))
        subimage.save(output_path_img+f'{sub_index}'+input_image_name)

if __name__=="__main__":
    
    input_base_path='C:/Users/huang/OneDrive/Desktop/520_Image/HABBOF/Lab1/'
    output_path_img='datasets/images/train/'
    output_path_labels='datasets/labels/train/'
    unit_degrees=36
    
    for i in range (1,1001,5):
      input_label_name=f'{i:06}.txt'
      input_image_name=f'{i:06}.jpg'
      subimage_indexes=convert_annotation(unit_degrees, input_base_path, input_label_name,output_path_labels)
      generate_subimages(unit_degrees,subimage_indexes,input_base_path,input_image_name,output_path_img)
      print(f'{i:06}done')
      
    output_path_img='datasets/images/val/'
    output_path_labels='datasets/labels/val/'
    for i in range (1200,1401,5):
      input_label_name=f'{i:06}.txt'
      input_image_name=f'{i:06}.jpg'
      subimage_indexes=convert_annotation(unit_degrees, input_base_path, input_label_name,output_path_labels)
      generate_subimages(unit_degrees,subimage_indexes,input_base_path,input_image_name,output_path_img)
      print(f'{i:06}done')
    output_path_img='datasets/images/test/'
    output_path_labels='datasets/labels/test/'
    for i in range (1500,1701,5):
      input_label_name=f'{i:06}.txt'
      input_image_name=f'{i:06}.jpg'
      subimage_indexes=convert_annotation(unit_degrees, input_base_path, input_label_name,output_path_labels)
      generate_subimages(unit_degrees,subimage_indexes,input_base_path,input_image_name,output_path_img)
      print(f'{i:06}done')
     
    '''
    fr = cv2.imread('datasets/images/train/2000476.jpg')
    fr_bb = drawRoBBs(fr=fr, ann_path='datasets/labels/train/2000476.txt')
    cv2.imwrite('result.jpg', fr_bb)
    '''



    
    

