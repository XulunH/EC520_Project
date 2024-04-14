from PIL import Image
import cv2
from originalutils import drawRoBBs
import numpy as np
import detect


def printimg(image_path,ann_path,output_path):
  rotation_times=10
  degrees=360/rotation_times
  results=[]
  img=cv2.imread(image_path)
  imgwithbb=drawRoBBs(img,ann_path)
  rgb_image = cv2.cvtColor(imgwithbb,cv2.COLOR_BGR2RGB)

# Convert NumPy array to PIL Image
  pil_image = Image.fromarray(rgb_image)
  for times in range (rotation_times):
    result=(detect.draw_inverted_triangle(detect.rotate_img(pil_image,degrees*times),degrees))
    result.save(output_path+f'{times}.jpg')

if __name__=='__main__':
  HABBOF_path='C:/Users/huang/OneDrive/Desktop/520_Image/HABBOF/'
  for i in range(1,1000,50):
    image_path=HABBOF_path+f'Meeting2/{i:06}.jpg'
    ann_path=HABBOF_path+f'Meeting2/{i:06}.txt'
    output_path=f'originalbb/{i:06}'
    printimg(image_path,ann_path,output_path)