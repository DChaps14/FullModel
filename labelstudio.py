import webbrowser, os
import shutil
import time
import json
import numpy as np
from PIL import Image, ImageDraw
import random
from bs4 import BeautifulSoup
import re

def launch(class_dict):
    """ Launches an html file for the user to label """
    
    html = open("label_studio.html", 'r', encoding="utf-8")
    
    # Parse the text file into html format
    soup = BeautifulSoup(html, "html.parser")
    label_script = soup.find(id="label_studio_script")
    label_string = label_script.contents[0]
    
    # Generate random labels to be used within the html
    colour_string = "0123456789abcdef"
    random.seed()
    new_labels = '<PolygonLabels name="tag" toName="img">\n'
    for index, label in enumerate(class_dict.keys()):
        colour = "".join(random.choices(colour_string, k=6))
        html_string = f"<Label background='#{colour}' value='{label}'></Label>\n"
        new_labels += html_string
    new_labels += "</PolygonLabels>"
    
    # Substitute the new labels and random colours into the html
    updated_script = re.sub("<PolygonLabels name=\"tag\" toName=\"img\">[\s\S]+</PolygonLabels>", new_labels, label_string)
    label_script.string = updated_script
    
    html = open("label_studio.html", 'w')
    html.write(str(soup))
    html.close()
    
    # Open the label_studio instance
    webbrowser.open('file://' + os.path.realpath("label_studio.html"))
    
    # Retrieve the path of the current file, and use this to find the downloaded annotations
    current_path = os.path.dirname(os.path.abspath(__file__))
    path = current_path.split("\\")[:3] # Extracts the user's directory
    download_path = "\\".join(path) + "\\Downloads\\annotation.txt" # Finds the annotation within the user's downloads
    
    # Wait for the user to submit the annotation
    while not os.path.exists(download_path):
        time.sleep(5)
        print("waiting")
    
    # Copy the annotation to this directory, and remove it from downloads
    shutil.move(download_path, "./newAnnotation.txt")
    os.remove(download_path)
    
    # Open the image and extract its details
    image = Image.open("chosen_image.jpg")
    image_array = np.array(image)
    image_height, image_width, _ = image_array.shape
    full_image_mask = np.zeros((len(image_array), len(image_array[0]), 1))
    detections = []
    
    # Open and read the annotation information supplied by the user
    annotation_file = open("newAnnotation.txt")
    annotation_info = annotation_file.readlines()
    for annot_json in annotation_info:
        annot = json.loads(annot_json)
        
        # Extract the points from each of the user's annotations
        points = annot["value"]["points"]
        label = annot["value"]["polygonlabels"][0]
        label_int = class_dict.get(label) + 1
        mask_points = {}
        minx, miny, maxx, maxy = round(points[0][0]*image_width*0.01), round(points[0][1]*image_height*0.01), round(points[0][0]*image_width*0.01), round(points[0][1]*image_height*0.01)
        
        # Find the maximum and minimum x and y points, to determine the bounding box
        for index, point in enumerate(points):
            point = (round(point[0]*image_width*0.01), round(point[1]*image_height*0.01))
            points[index] = point
            if point[0] < minx:
                minx = point[0]
            elif point[0] > maxx:
                maxx = point[0]
            if point[1] < miny:
                miny = point[1]
            elif point[1] > maxy:
                maxy = point[1]
        
        # Generate a dummy image, and draw the mask polygon on it
        drawing_image = Image.new("L", (image_width, image_height), 0)
        polygon_draw = ImageDraw.Draw(drawing_image)
        polygon_draw.polygon(points, fill=1, outline=1)
        
        # Extract the mask from the dummy image, and add it to the full image mask
        full_mask = np.array(drawing_image)
        full_mask = np.resize(full_mask, (image_height, image_width, 1))
        full_mask = np.where(full_mask, label_int, 0)
        full_image_mask = np.where(full_mask, label_int, full_image_mask)
        
        # Create the cropped image and mask, and add a new detection for the instance
        crop_image = image_array[miny:maxy, minx:maxx]
        crop_mask = full_mask[miny:maxy, minx:maxx]
        detection = {"label": label, "mask": np.ndarray.tolist(crop_mask),
                     "bounding_box": [minx, miny, maxx, maxy]}
        detections.append(detection)
        
    # Generate a psuedo-random filename, and save the image and json as a usableImage
    filename = str(abs(hash(time.localtime())))
    image.save(f"UNetPredictions/usableImages/images/{filename}.jpg")
    mask_json = {"filename": filename+".jpg", "ground_truth": {"detections": detections},
            "skip_full_mask": False}
    with open(f"UNetPredictions/usableImages/masks/{filename}.json", 'w') as mask_file:
        json.dump(mask_json, mask_file)