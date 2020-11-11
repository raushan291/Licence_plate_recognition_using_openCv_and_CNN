import threading
import time
import os
from PIL import Image
import numpy as np
from number_plate_sepration_openCv import run_number_plate_sepration_openCv
from char_seg import run_char_seg
from char_seg_v2 import run_char_seg_v2
from model import *
import cv2

thread = threading.Thread(target=run_number_plate_sepration_openCv())
thread.start()

while(thread.is_alive()):
    time.sleep(0.5)

thread = threading.Thread(target=run_char_seg_v2())
thread.start()

testImgDir = '/home/rakumar/char_segmentation/output_seg_chars/'
testImgDirList = sorted(os.listdir(testImgDir))
predictions=[]

for dir in testImgDirList:
    test_images = sorted(os.listdir(testImgDir+dir))
    
    pred = ''
    for i, image in enumerate(test_images):
        test_image = testImgDir+dir+'/'+image
        
        test_image = Image.open(test_image)
        
        test_image = test_image.convert('1') # convert a RGB image to binary image
        test_image = test_image.resize((36,10))
        test_image = np.array(test_image)
        test_image = torch.FloatTensor(test_image)
        
        test_image = test_image.unsqueeze(0)
        test_image = test_image.unsqueeze(0)

        outputs = net(test_image)
        _, predicted = torch.max(outputs, 1)
        
        pred=pred+str(mapping[predicted.numpy()[0]])
        # print('Predicted: ', predicted.numpy()[0])
    predictions.append(pred)
print(predictions)