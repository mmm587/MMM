import os
import time
# print('nohup python val.py --data data/VOC.yaml --weights runs/train/211107yolov5-FocalLoss-0.5/weights/best.pt --device --name 211107yolov5-FocalLoss-0.5 &') 
# time.sleep(5)
# print('nohup python val.py --data data/VOC.yaml --weights runs/train/211110yolov5-FocalLoss-1.0/weights/best.pt --device --name 211110yolov5-FocalLoss-1.0 &') 
# os.wait
# print('nohup python val.py --data data/VOC.yaml --weights runs/train/211111yolov5-FocalLoss-2.0/weights/best.pt --device --name 211111yolov5-FocalLoss-2.0 &') 

def fun(i=0):
    print('fun%d begin' % i)
    time.sleep(5)
    os.wait
    print('fun end')

if __name__ == '__main__':
    fun(1)
    fun(2)
