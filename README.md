# Licence_plate_recognition_using_openCv_and_CNN
Required Folders:
  (a) inputs --> Put car/vehicle images here.
  (b) LP_images --> Cropped licence plate will be saved here.
  (c) output_seg_chars --> Final segmented characters will be saved here.

Required files:
  (a) main.py ==> This file will execute Number_plate_sepration_openCv, char_seg_v2, and model script.
  (b) number_plate_sepration_openCv.py ==> This file have script to generate LP images (Using openCv).
  (c) char_seg_v2.py ==> This file have character segmentation script (Using openCv).
  (d) model.py ==> Defines CNN model and predict output for new image based on saved model (model.pth).
 
Other files:
  (a) char_seg.py ==> Alternative for char_seg_v2.py file.
  (b) model_training.py ==> Script to train CNN model.
  (c) create_hdf5_files_trainingset.py ==> Script to create training datasets in .hdf format
 
  (*) trainingDataset.h5 ==> Generated training datasets in .hdf format
  (*) model.pth  ==> Saved pyTorch Model
