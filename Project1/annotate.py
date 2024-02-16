import cv2
from roipoly import RoiPoly
import matplotlib.pyplot as plt
import numpy as np
import os

# Create a directory for masks if it doesn't exist
os.makedirs('orangemasks', exist_ok=True)

# List of your images
imagepath = 'trainingimgs'

# Defines class labels
class_labels = ["OrangeCone", "OtherClass"]

# Loop over each image
for imagefile in os.listdir(imagepath):
    imagefilepath = os.path.join(imagepath, imagefile)
    
    # Read and convert image
    image = cv2.imread(imagefilepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # For each class label, show the image and ask the user to annotate it
    for class_label in class_labels:
        plt.imshow(image)
        plt.title(f'Annotate for {class_label}: {imagefile}')
        plt.show(block=False)

        # Let the user draw the polygon for the current class
        roi = RoiPoly(color='r')
        roi.display_roi()

        # Create the mask from the ROI
        mask = roi.get_mask(image[:,:,0])

        # Save the mask to a .npy file
        mask_filename = f"{os.path.basename(imagefile).replace('.png', '')}_{class_label}_mask.npy"
        mask_filepath = os.path.join('orangemasks', mask_filename)
        np.save(mask_filepath, mask)