# Extracting images and masks from the saved files
import json
from create_gui import GUI
import random
import os
from PIL import Image
import tensorflow as tf
import numpy as np
import labelstudio
import shutil

BASE_DIR = "UNetPredictions/detections/"
IMAGE_DIR = BASE_DIR + "images"
MASK_DIR = BASE_DIR + "masks"

RESULT_DIR = "UNetPredictions/usableImages/"

def setup():
    try:
        os.mkdir(RESULT_DIR)
        os.mkdir(RESULT_DIR + "images")
        os.mkdir(RESULT_DIR + "masks")
    except:
        print("Already Created")

def launch(class_dict):
    num_masks = 0
    num_mask_accepted = 0
    print("GUI Stage Launched")
    
    for image in os.scandir(IMAGE_DIR):
      # Currently displaying the crop and the mask on top of it - this could help to confirm whether the cropped image is suitable, as well as the segmentation
        base_image = None
        crops = []
        masks = []
        crops_info = []
        labels = []
        for crop in os.scandir(image.path):
            if crop.is_dir():
                continue
            if crop.name == "base_image.jpg":
                base_image = crop
                continue
            
            crop_image = Image.open(crop.path)
            #if not user_accepts_crop(crop_image): # Send the crop to be approved by the user
                #print("crop not accepted")
                #continue
    
            crop_mask = np.load(crop.path.replace("images", "masks").replace(".jpg", ".npy"))
            with open(crop.path.replace("images", "masks").replace(".jpg", ".txt"), 'r') as crop_info:
                label, x1, y1, x2, y2 = crop_info.readline().split(" ")
            
            crops.append(crop_image)
            masks.append(crop_mask)
            crops_info.append([x1, y1, x2, y2])
            labels.append(label)
            
        gui = GUI(crops, masks, crops_info, labels)
        gui.construct_gui()
        gui.window.destroy()
        usable_masks = gui.usable_masks
        usable_crops = gui.usable_crops
            
        base_PIL_image = Image.open(base_image.path)
        image_dims = np.array(base_PIL_image).shape
        base_image_mask = np.zeros((image_dims[0], image_dims[1], 1))
        num_masks += len(usable_masks)
        for index, mask in enumerate(usable_masks):
            if type(mask) == type(None):
                continue
            num_mask_accepted += 1
            mask = np.array(mask)
            mask = np.reshape(mask, (len(mask), len(mask[0]), 1))
            label, crop_info = usable_crops[index]
            x1,y1,x2,y2 = crop_info
            mask_pad = tf.constant([[int(y1), image_dims[0]-int(y2)], [int(x1), image_dims[1]-int(x2)], [0,0]])
            resized_mask = tf.pad(mask, mask_pad, "CONSTANT")
            base_image_mask = np.where(resized_mask, resized_mask, base_image_mask)
                        
        full_image_gui = GUI([base_PIL_image], [base_image_mask], None, None)
        full_image_gui.construct_gui()
        full_image_gui.window.destroy()
        full_inaccurate = not full_image_gui.usable_masks
    
        detections = []
        for index, crop_info in enumerate(usable_crops):
            # crop.save(RESULT_DIR + f"images/{image.name}_{index}.jpg")
    
            # Rework the mask to only store the elements within the bounding box
            crop_mask = usable_masks[index]
            crop_mask_list = np.ndarray.tolist(np.array(crop_mask))
            detections.append({"label": crop_info[0], "mask": crop_mask_list, "bounding_box": crop_info[1] }) # Stored in xyxy format
    
        if detections:
            crop_dict = {
                          "filename": f"{image.name}.jpg", 
                          "ground_truth": {"detections": detections},
                          "skip_full_mask": full_inaccurate #Signifies whether the detections in ground_truth can be sed to formulate a full image to feed into the model
                        }
            with open(RESULT_DIR + f'masks/{image.name}.json', 'w') as mask_file:
                json.dump(crop_dict, mask_file)
            base_PIL_image = Image.open(base_image.path)
            base_PIL_image.save(RESULT_DIR + f"images/{image.name}.jpg") # Reader can use the bounding values to extract the cropped image from the base image for training
        
    
    print(f"User accepted {num_mask_accepted} masks out of {num_masks}")
    # Select a random image in detections directory
    available_images = os.listdir(IMAGE_DIR)
    random_image = available_images[random.randrange(len(available_images))]
    shutil.copy(f"{IMAGE_DIR}/{random_image}/base_image.jpg", "./chosen_image.jpg")
    labelstudio.launch(class_dict)