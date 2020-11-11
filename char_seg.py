import cv2
import os

def getCharcter(image):
    # Ordinary license plate value is 0.95, new energy license plate is changed to 0.9
    segmentation_spacing = 0.9

    '''1. read the picture, and do grayscale processing'''
    img = cv2.imread(image)
    # img = cv2.resize(img,(190,37))
    img = cv2.resize(img,(46 if img.shape[1]< 46 else img.shape[1],img.shape[0]))
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


    '''2. Binary the grayscale image'''
    # ret, img_threshold = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    ret, img_threshold = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
    print(img_threshold.shape)

    '''3. Split characters'''
    white = []  # Record the sum of white pixels in each column
    black = []  # Record the sum of black pixels in each column
    height = img_threshold.shape[0]
    width = img_threshold.shape[1]

    '''4. Cycle through the sum of black and white pixels for each column'''
    for i in range(width):
        white_count = 0
        black_count = 0
        for j in range(height):
            if img_threshold[j][i] == 255:
                white_count += 1
            else:
                black_count += 1

        white.append(white_count)
        black.append(black_count)

    white_max = max(white)
    black_max = max(black)

    '''5. Split the image, given the starting point of the character to be split'''
    def find_end(start):
        end = start + 1
        for m in range(start + 1, width - 1):
            if(black[m] > segmentation_spacing * black_max-1):   #segmentation_spacing * black_max ; Added -1 for charcters AA KK MM RR VV WW
                end = m
                break
        return end

    n = 1
    start = 1
    end = 2
    while n < width - 1:
        n += 1
        if(white[n] > (1 - segmentation_spacing) * white_max):
            start = n
            end = find_end(start)
            n = end
            if end - start > 1:      # i.e, (end - start) = width of a character
                print(start, end)
                character = img_threshold[1:height, start:end]
                '''
                # op_folder = image.split('.')[0].split('/')[-1]
                # output_folder='/home/rakumar/char_segmentation/output_seg_chars/'+op_folder+'/'
                # if not os.path.exists(output_folder):
                #     os.mkdir(output_folder)
                # img_name = output_folder+'{0}.png'.format(str(op_folder)+'_'+str(n))
                '''
                
                # op_folder = image.split('_')[-2].split('/')[-1]
                op_folder = image.split('/')[-2]
                # imgName = image.split('.')[0].split('_')[-1]
                imgName = image.split('/')[-1].split('.')[0]
                
                output_folder='/home/rakumar/char_segmentation/output_seg_chars/'+op_folder+'/'
                if not os.path.exists(output_folder):
                    os.mkdir(output_folder)
                img_name = output_folder+'{0}.png'.format(str(imgName)+'-'+str(n))

                cv2.imwrite(img_name, character)  
                # cv2.imshow('character', character)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
    os.remove(image)

def run_char_seg(image):
    # img_dir = "/home/rakumar/char_segmentation/LP_images/"
    # imgList = os.listdir(img_dir)
    # for img in imgList:
    #     image=img_dir+img
    #     getCharcter(image)
    getCharcter(image)

# run_char_seg('/home/rakumar/char_segmentation/LP_images/M.png')