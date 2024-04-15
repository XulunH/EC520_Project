import os
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
           with open(output_path_labels+f'{sub_index}Meeting2_'+input_txt_name,"w") as f:
             f.write(formatted_row)
        else:
            with open(output_path_labels+f'{sub_index}Meeting2_'+input_txt_name,"a") as f:
             f.write(formatted_row)
    return subimage_indexes
        
def generate_subimages(unit_degrees,subimage_indexes,input_base_path,input_image_name,output_path_img):
    img=Image.open(input_base_path+input_image_name)
    for sub_index in subimage_indexes:
        subimage=(draw_inverted_triangle(rotate_img(img,unit_degrees*sub_index),unit_degrees))
        subimage.save(output_path_img+f'{sub_index}Meeting2_'+input_image_name)



def prepare_for_training(input_path_from_root, unit_degrees, output_path_img, output_path_labels):
    input_base_path = 'C:/Users/huang/OneDrive/Desktop/520_Image/HABBOF/' + input_path_from_root
    files = os.listdir(input_base_path)
    image_files = [f for f in files if f.endswith('.jpg')]

    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0]
        input_label_name = f'{base_name}.txt'
        input_image_name = image_file

        full_label_path = os.path.join(input_base_path, input_label_name)
        full_image_path = os.path.join(input_base_path, input_image_name)

        if os.path.exists(full_image_path) and os.path.exists(full_label_path):
            subimage_indexes = convert_annotation(unit_degrees, input_base_path, input_label_name, output_path_labels)
            generate_subimages(unit_degrees, subimage_indexes, input_base_path, input_image_name, output_path_img)
            print(f'Processing completed for {base_name}')
        else:
            print(f'Missing file for {base_name}, skipping...')


def showboxes():
   fr = cv2.imread('post_processing/Lunch1/images/1Lunch1_001042.jpg')
   fr_bb = drawRoBBs(fr=fr, ann_path='post_processing/Lunch1/labels/1Lunch1_001042.txt')
   cv2.imwrite('result.jpg', fr_bb)

if __name__=="__main__":
    
    input_base_path='C:/Users/huang/OneDrive/Desktop/520_Image/EC520_Project/Project/CEPDOF/'
    output_path_img='datasets/images/train/'
    output_path_labels='datasets/labels/train/'
    unit_degrees=36
    prepare_for_training('Meeting2/',unit_degrees,'post_processing/Meeting2/images/','post_processing/Meeting2/labels/')
    
    
    



    
    

