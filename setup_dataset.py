""" A module to be used to download sample data from FiftyOne
Should only ever really be executed once at the start of the pipeline """
import fiftyone.zoo as foz
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as pt
import shutil
import random

def extract_usables(class_dict):
    """ Extracts the images from the usableImages folder, and returns them to be used by the training pipeline
    Inputs: 
    class_dict - a dictionary of classes to find, of format {class_description: label_integer}"""
    BASE_DIR = "UNetPredictions/usableImages/"
    IMAGE_DIR = BASE_DIR + "images"
    MASK_DIR = BASE_DIR + "masks"

    CROP_DIR = "UNetPredictions/yolov5/images/"
    CROP_INFO_DIR = "UNetPredictions/yolov5/labels/"

    usable_images = []
    usable_masks = []
    
    # Seed the rng early to ensure all subsequent calls are using the same rng
    random.seed()

    for index, image in enumerate(os.scandir(IMAGE_DIR)):
        if image.is_dir():
            continue
        if random.random() < 0.2: # Make 20% of the available images into a validation dataset
            split = "val"
        else:
            split = "train"
        mask_info = json.load(open(image.path.replace("/images/", "/masks/").replace(".jpg", ".json")))
        skip_full_image = mask_info.get("skip_full_image") # The user has indicated that it is not suitable to create a full image using the masks we have
        
        base_image = Image.open(image.path)
        # Convert the image to RGB mode if it isn't already
        if base_image.mode != "RGB":
            base_image = image.convert("RGB")
        image_array = np.array(Image.open(image.path))
        image_name = image.name.split('/')[-1]
        image_height, image_width, _ = image_array.shape
        
        if not skip_full_image:
            base_image_mask = np.zeros((image_height, image_width, 1))
        
        # Get the name of the new detections file, and remove it if it exists - we'll create a new one with all of the currently usable images
        detections = mask_info["ground_truth"]["detections"]
        detection_filename = f"{CROP_INFO_DIR}/{split}/{image_name[:-3]}txt"
        if os.path.exists(detection_filename):
            os.remove(detection_filename)

        detection_file = open(detection_filename, "w")
        for detection in detections:
            # For each detection within the image, extract its information and write it to the detection file for the yolo model
            class_label = detection["label"]
            class_int = class_dict.get(class_label)
            x1, y1, x2, y2 = detection["bounding_box"]
            x,y,w,h = ((int(x1)+int(x2))/2)/image_width, ((int(y1)+int(y2))/2)/image_height, (int(x2)-int(x1))/image_width, (int(y2)-int(y1))/image_height # Normalised xywh format
            detection_file.write(f"{class_int} {x} {y} {w} {h}\n")

            # Extract each mask from the detection information, and generate a array from it
            mask = detection.get("mask")
            if mask is not None:
                mask = np.array(mask)
                cropped_image = image_array[int(y1):int(y2), int(x1):int(x2)]
                usable_images.append(cropped_image)
                usable_masks.append(mask)          

                if not skip_full_image:
                    mask_paddings = tf.constant([[int(y1), image_height-int(y2)], [int(x1), image_width-int(x2)], [0,0]])
                    full_mask = tf.pad(mask, mask_paddings, "CONSTANT")
                    base_image_mask = np.where(full_mask, full_mask, base_image_mask)
        
        base_image.save(f"{CROP_DIR}/{split}/{image_name}")
        detection_file.close()

        if not skip_full_image:
            usable_images.append(image_array)
            usable_masks.append(base_image_mask)

    return usable_images, usable_masks


def install_dataset(class_dict, num_samples, num_trainable, yaml_filename):
    """ Installs the COCO dataset for sample data use
    Inputs:
    class_dict - a dictionary of classes to find, of format {class_description: label_integer}
    num_samples - the number of samples we want to download from the dataset
    num_trainable - the number of training images we want to download
    yaml_filename - the filename of the yaml file that we want to use to feed the yolo model
    """

    # Create the appropriate directories
    try:
        os.mkdir("UNetPredictions/")
        os.mkdir("UNetPredictions/yolov5")
        os.mkdir("UNetPredictions/yolov5/images")
        os.mkdir("UNetPredictions/yolov5/labels")
        os.mkdir("UNetPredictions/yolov5/images/train")
        os.mkdir("UNetPredictions/yolov5/labels/train")
        os.mkdir("UNetPredictions/yolov5/images/val")
        os.mkdir("UNetPredictions/yolov5/labels/val")
        os.mkdir("UNetPredictions/usableImages/")
        os.mkdir("UNetPredictions/usableImages/images")
        os.mkdir("UNetPredictions/usableImages/masks")
        os.mkdir("UNetPredictions/testImages")
    except FileExistsError:
        print("Directories already created")
    
    BASE_DIR = "UNetPredictions/usableImages/"
    IMAGE_DIR = BASE_DIR + "images"
    MASK_DIR = BASE_DIR + "masks"
    TEST_DIR = "UNetPredictions/testImages"
    
    classes = []
    for key in class_dict.keys():
        classes.append(key)
    coco_training_dataset = foz.load_zoo_dataset("coco-2017", dataset_dir="./coco_data", split="train", label_types=["segmentations"], classes=classes, max_samples=num_samples)
    
    for index, sample in enumerate(coco_training_dataset):
        image_filepath = sample['filepath']
        image_name = image_filepath.split("/")[-1]
        image = Image.open(image_filepath)
        # Some images in the COCO dataset aren't RGB, which may mess up the model - convert them here to be safe
        if image.mode != "RGB":
            image = image.convert("RGB")
    
        if index >= num_trainable:
            image.save(f"{TEST_DIR}/{image_name}") # Automatically save the image to the test directory if we have enough training data
        else:
            image_array = np.array(image)
            image_width = len(image_array[0])
            image_height = len(image_array)
            detections = sample['ground_truth']['detections']
    
            # Create a semantic bitmap that indicates where the pixels of the instances exist in the crops
            detections_for_json = []
            for detection in detections:
                label = detection["label"]
                label_int = class_dict.get(label)
                if label_int == None:
                    continue
                else:
                    label_int += 1 # Account for the 'background' class
                bbox = detection["bounding_box"]
                mask = detection["mask"]
                x1, y1 = round(bbox[0]*image_width), round(bbox[1]*image_height)
                width, height = len(mask[0]), len(mask)
                crop_mask = np.where(mask, label_int, 0)
                crop_mask = np.reshape(crop_mask, (len(crop_mask), len(crop_mask[0]), 1))
    
                # Add the crop and its mask to a json file
                json_detection = {"bounding_box": [x1, y1, x1+width, y1+height],
                                  "mask": np.ndarray.tolist(crop_mask),
                                  "label": label}
                detections_for_json.append(json_detection)
                
    
            image.save(f"{IMAGE_DIR}/{image_name}")
            mask_json = {
                "ground_truth": {"detections": detections_for_json},
                "filename": image_name
            }
            with open(f'{MASK_DIR}/{image_name.replace(".jpg", ".json")}', 'w') as mask_file:
                    json.dump(mask_json, mask_file)
    
    
    # Fill out a yaml file with information the YOLOv5 model can use to find the training images
    info_file = open(f"yolov5/data/{data_yaml}.yaml", 'w')
    info_file.write("train: ../UNetPredictions/yolov5/images/train\n")
    info_file.write("val: ../UNetPredictions/yolov5/images/val\n")
    info_file.write(f"\nnc: {len(class_dict)}\n")
    info_file.write(f"\nnames: {classes}")
    info_file.close()