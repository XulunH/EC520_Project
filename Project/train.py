from ultralytics import YOLO

model = YOLO('yolov5su.pt')  

if __name__=="__main__":
  results = model.train(data='test.yaml', epochs=20, imgsz=640, device=[0])