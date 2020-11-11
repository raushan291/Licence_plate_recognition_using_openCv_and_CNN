import cv2
import os
import numpy as np

def getCharcter_v2(image):
    img = cv2.imread(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)

    kernel = np.ones((2,1), np.uint8)
    morph_Closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('morphological-Closing(Dialation+erosion)', morph_Closing)

    ret, thresh = cv2.threshold(morph_Closing, 0, 255, cv2.THRESH_OTSU)
    cv2.imshow('thresh', thresh)
    
    major = cv2.__version__.split('.')[0]
    if major == '3':
        im2, ctrs, hier = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    else:
        ctrs, hier = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    imgCount=0
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)

        ret, thresh_binary_inv = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow('thresh_binary_inv', thresh_binary_inv)
        roi = thresh_binary_inv[y:y + h, x:x + w]

        area = w*h
        
        if 100 < area < 500 and h > 10:   # h>10 will filter a noise having height less than 10px..

            op_folder = image.split('.')[0].split('/')[-1]
            output_folder='/home/rakumar/char_segmentation/output_seg_chars/'+op_folder+'/'
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
            img_name = output_folder+'{0}.png'.format(str(imgCount))
            imgCount += 1

            # cv2.imwrite("ROI_{}.png".format(i), roi)
            cv2.imwrite(img_name, roi)
            rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow('rect', rect)

    # cv2.waitKey(0)

def run_char_seg_v2():
    img_dir = "/home/rakumar/char_segmentation/LP_images/"
    imgList = os.listdir(img_dir)
    for img in imgList:
        image=img_dir+img
        getCharcter_v2(image)

# run_char_seg_v2()