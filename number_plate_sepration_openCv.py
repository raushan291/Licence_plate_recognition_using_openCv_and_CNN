import cv2                  # Importthe Opencv Library
import numpy as np          # Import NumPy, package for scientific computing with Python
import os

def image_show(msg, img):
     cv2.namedWindow(msg,cv2.WINDOW_NORMAL)     # Create a Named window to display image
     cv2.imshow(msg,img)                        # Display the Image

def save_number_plate(NumberPlateCnt, img, image_name):
     x,y,w,h = cv2.boundingRect(NumberPlateCnt)
     if not (NumberPlateCnt is None):
          new_img=img[y:y+h,x:x+w]
          cv2.imwrite("/home/rakumar/char_segmentation/LP_images/"+image_name, new_img)
     else:
          print("No contours found, so no licence plate detected for image {0}".format(image_name))


def get_number_plate(image):
     img = cv2.imread(image)                    # Read the Image
     # img = cv2.resize(img, (100,100))
     image_show("Original Image", img)

     # RGB to Gray scale conversion
     img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
     image_show("1 - Grayscale Conversion", img_gray)

     # Noise removal with iterative bilateral filter(removes noise while preserving edges)
     # noise_removal = cv2.bilateralFilter(img_gray,9,75,75)
     noise_removal = cv2.fastNlMeansDenoising(img_gray,None,10,7,21)
     ## noise_removal = cv2.GaussianBlur(img_gray,(5,5),0) ##
     image_show("2 - Noise Removal(Bilateral Filtering)", noise_removal)

     # Histogram equalisation for better results
     equal_histogram = cv2.equalizeHist(noise_removal)
     image_show("3 - Histogram equalisation",equal_histogram)

     # Morphological opening with a rectangular structure element
     kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))                                # create the kernel
     morph_image = cv2.morphologyEx(equal_histogram,cv2.MORPH_OPEN,kernel,iterations=12)
     image_show("4 - Morphological opening",morph_image)

     # Image subtraction(Subtracting the Morphed image from the histogram equalised Image)
     sub_morp_image = cv2.subtract(equal_histogram,morph_image)
     image_show("5 - Image Subtraction", sub_morp_image)

     # Thresholding the image
     ret,thresh_image = cv2.threshold(sub_morp_image,0,255,cv2.THRESH_OTSU)
     image_show("6 - Thresholding",thresh_image)

     # Applying Canny Edge detection
     canny_image = cv2.Canny(thresh_image,250,255)
     image_show("7 - Canny Edge Detection",canny_image)

     canny_image = cv2.convertScaleAbs(canny_image)

     # Dilation - to strengthen the edges
     kernel = np.ones((3,3), np.uint8)                               # Create the kernel for dilation
     dilated_image = cv2.dilate(canny_image,kernel,iterations=1)     # Carry out Dilation
     image_show("8 - Dilation(closing)", dilated_image)

     # Finding Contours in the image based on edges
     # new,contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
     major = cv2.__version__.split('.')[0]
     if major == '3':
          new,contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
     else:
          contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

     # Sort the contours based on area ,so that the number plate will be in top 10 contours
     contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10]
     print(contours[0].shape)

     NumberPlateCnt = None

     # loop over the contours list
     for c in contours:
          # approximate the contour
          peri = cv2.arcLength(c, True)
          approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # Approximating with 6% error
          # if our approximated contour has four points, then
          # we can assume that we have found our NumberPlate
          if len(approx) == 4:           # Select the contour with 4 corners
               NumberPlateCnt = approx   #assign to NumberPlateCnt when approximate contour found
               break                    # break the loop when Number Plate contour found/approximated
     
     save_number_plate(NumberPlateCnt,img,image_name=image.split('/')[-1])

     try:   # will throw exception if No countours found..
          
          # Drawing the selected contour on the original image
          final = cv2.drawContours(img, [NumberPlateCnt], -1, (0, 255, 0), 3)

          image_show("9 - Approximated Contour",final)

          # SEPARATING OUT THE NUMBER PLATE FROM IMAGE:

          # Masking the part other than the number plate
          mask = np.zeros(img_gray.shape,np.uint8)                            # create an empty black image
          new_image = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1,)       # Draw the contour of number plate on the black image - This is our mask
          new_image = cv2.bitwise_and(img,img,mask=mask)                      # Take bitwise AND with the original image so we can just get the Number Plate from the original image
          image_show("10 - Number Plate Separation",new_image)

          #HISTOGRAM EQUALIZATION FOR ENHANCING THE NUMBER PLATE FOR FURTHER PROCESSING:

          y,cr,cb = cv2.split(cv2.cvtColor(new_image,cv2.COLOR_RGB2YCrCb))        # Converting the image to YCrCb model and splitting the 3 channels
          y = cv2.equalizeHist(y)                                                 # Applying histogram equalisation
          final_image = cv2.cvtColor(cv2.merge([y,cr,cb]),cv2.COLOR_YCrCb2RGB)    # Merging the 3 channels
          image_show("11 - Enhanced Number Plate",final_image)

          # cv2.waitKey(0)                                                           # Wait for a keystroke from the user before ending the program
     except:
          # cv2.waitKey(0) 
          pass

def run_number_plate_sepration_openCv():
     inputPath = '/home/rakumar/char_segmentation/inputs/'
     for image in os.listdir(inputPath):
          img=inputPath+image
          get_number_plate(img)

# run_number_plate_sepration_openCv()