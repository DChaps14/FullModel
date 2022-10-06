import subprocess
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
    dirs = [BASE_DIR, RESULT_DIR, IMAGE_DIR, MASK_DIR]
    for dir in dirs:
        try:
            os.mkdir(dir)
        except FileExistsError:
            print("Files already created")    


def detect_yolo(weights_location, min_confidence):
    detect.run(weights=weights_location, conf_thres=min_confidence, source="UNetPredictions/testImages/", save_txt=True, project="runs/detections")

    
def create_mask(pred_mask):
    """Creates a mask based on a prediction from the model that can be displayed on top of an image"""
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask    

# Return the area of the image to sort the crop collection by
def sort_images(val):
    return (val[2]-val[0])*(val[3]-val[1])

def detect_unet(input_dir, flipped_dict, input_size):
    class_dict = {}
    for item in flipped_dict.items():
        class_dict[item[1]] = item[0]

    try:
        label_dir = os.scandir(os.path.join(input_dir, "labels"))
    except:
        print("No instances found within the provided image")
        
    model = keras.models.load_model("model.h5")

    for crops in label_dir:
        if crops.is_dir():
            continue
        # This is a folder for all the instances within the images
        image_crops = []
        image_name = f"{crops.name[:-4]}"
        image = np.array(Image.open(crops.path.replace("/labels/", "/").replace("txt", "jpg")))
        image_height, image_width, _ = image.shape
        crops_file = open(crops.path, 'r')
        for crop in crops_file.readlines():
            class_int, x1, y1, x2, y2 = crop.split(" ")    
            image_crops.append((int(class_int), float(x1), float(y1), float(x2), float(y2)))
    
        image_crops = sorted(image_crops, key = sort_images)
    
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
            
            resized_crop = np.array([tf.image.resize(crop_array, [input_size, input_size])])
            prediction = model.predict(resized_crop)[0]
            pred_mask = create_mask(prediction)
            crop_image = tf.image.resize(resized_crop[0], [original_height, original_width])
            crop_mask = tf.image.resize(pred_mask, [original_height,original_width])
    
            padding = tf.constant([[crop_y1, image_height-crop_y2], [crop_x1, image_width-crop_x2], [0,0]])
            full_crop = tf.pad(crop_image, padding, 'CONSTANT')
            full_mask = tf.pad(crop_mask, padding, "CONSTANT")
    
            crop_image = tf.keras.utils.array_to_img(crop_image)
    
            crop_image.save(os.path.join(IMAGE_DIR, image_name, "file{index}.jpg"))
            np.save(os.path.join(MASK_DIR, image_name, "file{index}.npy"), crop_mask)
            with open(os.path.join(MASK_DIR, image_name, "file{index}.txt"), 'w') as label_file:
                label_file.write(f"{class_dict.get(class_int)} {crop_x1} {crop_y1} {crop_x2} {crop_y2}")