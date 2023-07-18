#importing modules
import torch
import cv2
import pytesseract
import numpy as np
import pandas as pd
#import argparse

#loading the model and using it
model = torch.hub.load(r'C:\Users\noman\yolov5', 'custom', path=r"C:\Users\noman\yolov5\runs\train\exp\weights\best.pt", source='local', force_reload=True)
#model.conf=0.5
#model.cuda()

image=cv2.imread(r"C:\Users\noman\Downloads\download (4).jpg")#[..., ::-1] #reading image(converting BGR To RGB)
#cv2.imshow('og image', image)
#cv2.waitKey(0)

result=model(image) #inference
#print(result)

bbox=result.pandas().xyxy[0]#.to_dict(orient='dict') #getting the coordinates of bounding boxes
print(bbox)

#storing the vertices of bounding boxes
xmin=bbox['xmin'][0]
ymin=bbox['ymin'][0]
xmax=bbox['xmax'][0]
ymax=bbox['ymax'][0]

#assigning a variable which holds the cropped immage
cropped_img = image[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
#cv2.imshow('cropped_image', cropped_img)
#cv2.waitKey(0)

#preprocessing

# convert the image to grayscale and flip the foreground
# and background to ensure foreground is now "white" and
# the background is "black"
gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)

#normalization of image(changing the range of intensity of pixels to (0,1))
#img_normalized = cv2.normalize(gray, None, 0, 1.0, cv2.NORM_MINMAX)

# threshold the image, setting all foreground pixels to
# 255 and all background pixels to 0
thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#cv2.imshow("thresh", thresh)
#cv2.waitKey(0)


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
ocr=pytesseract.image_to_string(thresh, config='--psm 6')
print(ocr)


#result.save()
#result.show() #showing the result
#crop = result.crop(save=True)
