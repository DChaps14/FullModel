import os
from PIL import Image
import numpy as np
import tensorflow as tf
import re
import matplotlib.pyplot as plt
import shutil
import yolov5.detect as detect

BASE_DIR = "UNetPredictions/"
RESULT_DIR = BASE_DIR + "detections/"
MASK_DIR = RESULT_DIR + "masks/"
IMAGE_DIR = RESULT_DIR + "images/"

def establish_directories():
    """ Creates the directories to store usable images if not already created """    
    dirs = [BASE_DIR, RESULT_DIR, IMAGE_DIR, MASK_DIR]
    for dir in dirs:
        try:
            os.mkdir(dir)
        except FileExistsError:
            print("Files already created")    


def detect_detector(weights_location, min_confidence):
    """ Runs the detection algorithm for the YOLOv5 model 
    Inputs:
    weights_location - the path to the weights we want to use for this detection run
    min_confidence - the minimum confidence level of images that we want to look at
    """
    detect.run(weights=weights_location, conf_thres=min_confidence, source="UNetPredictions/testImages/", save_txt=True, project="runs/detections")

    
def create_mask(pred_mask):
    """Creates a mask based on a prediction from the model that can be displayed on top of an image"""
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask    

def sort_images(val):
    """ Function to sort a list of images by their area """
    return (val[2]-val[0])*(val[3]-val[1])

def detect_segmenter(input_dir, flipped_dict, input_size):
    """ Runs a detection pipeline on U-Net
    Feeds each image's crops through the U-Net for detection, and saves each result to the file system 
    Inputs:
    input_dir - where the image crops are currently stored
    flipped_dict - a class dictionary of type {class_to_find: label_integer}
    input_size - the input size of the U-Net model
    """
    
    class_dict = {}
    for item in flipped_dict.items():
        class_dict[item[1]] = item[0]

    try:
        label_dir = os.scandir(os.path.join(input_dir, "labels")) # Find the directory that houses all of the crops
    except:
        print("No instances found within the provided image")
        return
        
    model = keras.models.load_model("model.h5")

    for crops in label_dir:
        if crops.is_dir():
            continue
        image_crops = []
        image_name = f"{crops.name[:-4]}"
        image = np.array(Image.open(crops.path.replace("/labels/", "/").replace("txt", "jpg")))
        image_height, image_width, _ = image.shape
        crops_file = open(crops.path, 'r')
        # Add details about each crop stored in the file to a list
        for crop in crops_file.readlines():
            class_int, x1, y1, x2, y2 = crop.split(" ")    
            image_crops.append((int(class_int), float(x1), float(y1), float(x2), float(y2)))
    
        # Sort the crops by their area
        image_crops = sorted(image_crops, key = sort_images)
    
        # Create directories for the image and masks if they haven't been already
        try:
            os.mkdir(os.path.join(IMAGE_DIR, image_name))
        except:
            print("Image directory already created")
        try:
            os.mkdir(os.path.join(MASK_DIR, image_name))
        except:
            print("Image directory already created")
    
        # Copy across the base image to the file structure
        base_image_path = os.path.join(input_dir, image_name + ".jpg")
        shutil.copy(base_image_path, os.path.join(IMAGE_DIR, image_name, "base_image.jpg"))
    
        for index, crop in enumerate(image_crops):
            class_int = crop[0]
            crop_x1 = round(crop[1])
            crop_y1 = round(crop[2])
            crop_x2 = round(crop[3])
            crop_y2 = round(crop[4])
    
            original_width = crop_x2-crop_x1
            original_height = crop_y2-crop_y1
            crop_array = image[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Resize each crop to the model's input size, and get a viewable prediction of each
            resized_crop = np.array([tf.image.resize(crop_array, [input_size, input_size])])
            prediction = model.predict(resized_crop)[0]
            pred_mask = create_mask(prediction)
            crop_image = tf.image.resize(resized_crop[0], [original_height, original_width])
            crop_mask = tf.image.resize(pred_mask, [original_height,original_width])
    
            crop_image = tf.keras.utils.array_to_img(crop_image)
            
            # Save the crop, its mask, and its details
            crop_image.save(os.path.join(IMAGE_DIR, image_name, "file{index}.jpg"))
            np.save(os.path.join(MASK_DIR, image_name, "file{index}.npy"), crop_mask)
            with open(os.path.join(MASK_DIR, image_name, "file{index}.txt"), 'w') as label_file:
                label_file.write(f"{class_dict.get(class_int)} {crop_x1} {crop_y1} {crop_x2} {crop_y2}")