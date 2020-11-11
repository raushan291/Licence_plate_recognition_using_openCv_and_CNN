from PIL import Image, ImageOps
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import cv2

data=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def getListOfFiles(dirName):
    listOfFile = sorted(os.listdir(dirName))
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

allFiles = getListOfFiles('/home/rakumar/char_segmentation/temp/1_NEW_CHAR')
# print(allFiles)


hf = h5py.File('/home/rakumar/char_segmentation/trainingDataset.h5', 'w')

allNpImg = []
allLabels = []
allNames = []
for i, img in enumerate(allFiles):
    im = Image.open(img)
    im = im.resize((36,10))
    np_im = np.array(im)

    allNpImg.append(np_im)

    # allNames.append(np.string_(img.split('/')[-2]))
    allLabels.append(data.index(img.split('/')[-2]))


import random
res = list(zip(allNpImg, allLabels))
random.shuffle(res)
allNpImg, allLabels = zip(*res)


# print(allLabels)
print(allNpImg[0].shape)
print(len(allLabels))
    
hf.create_dataset('image', data=allNpImg)
hf.create_dataset('label', data=allLabels)

hf.close()

### Reading HDF5 files ###
hf = h5py.File('/home/rakumar/char_segmentation/trainingDataset.h5', 'r')
hf.keys()

n1 = hf.get('image')
n2 = hf.get('label')

print(n1.shape)
print(n2.shape)
hf.close()

def load_dataset():
    train_dataset = h5py.File('/home/rakumar/char_segmentation/trainingDataset.h5', 'r')
    train_set_x_orig = np.array(train_dataset['image'][:]) # your train set features
    train_set_y_orig = np.array(train_dataset['label'][:]) # your train set labels

    ## plot img
    rows = 3
    cols = 3
    axes=[]
    fig=plt.figure()

    i =0
    for a in range(rows*cols):
        b = train_set_x_orig[i]
        print(b.shape)
        axes.append( fig.add_subplot(rows, cols, a+1) )

        subplot_title=(str(train_set_y_orig[i]))
        axes[-1].set_title(subplot_title)
         
        i+=1
        # b = b.squeeze(0)
        plt.imshow(b, interpolation='nearest', aspect=15)
    fig.tight_layout()    
    plt.show()

load_dataset()
