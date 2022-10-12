import tkinter as tk
from PIL import Image, ImageTk
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import numpy as np

""" A class used to construct a GUI and display a number of images that can either be approved or denied """
class GUI:
    
    def __init__(self, images, masks, info, labels):
        """
        Inputs:
        images - the list of images that the user needs to approve or deny
        masks - the list of segmentation masks that correspond to the images
        info - the coordinates of the cropped images within the base image
        labels - what each cropped image was identified as by the object detector
        """
        self.window = tk.Tk()
        self.checking_mask = False
        self.images = images
        self.masks = masks
        self.crop_info = info
        self.labels = labels
        self.current_index = 0
        self.check_crop_str = "Is the image below a suitable bounding box around one or more '{crop_class}'?"
        self.check_mask_str = "Are all trainable classes suitably segmented in the image below?"
        
        # Set a default alpha value for masks to utilise
        self.mask_alpha = 0.4
        
        # Stores the usable images
        self.usable_crops = []
        self.usable_masks = []
        
        # Initialise the variables for the components for the GUI
        self.instruction_label = None
        self.image_label = None
        self.confirm_button = None
        self.reject_button = None
        
    def process_images(self, images):
        """Converts the PIL image to a matplotlib plot, then back to a PIL image to provide a version of the image that displays nicely on the GUI.
        Applies any masks that are supplied alongside it on top of the original image"""
        
        fig = plt.figure()
        # Show the base image, then any masks that apply to it, on the plot
        plt.imshow(images[0])
        for index in range(1, len(images)):
            plt.imshow(images[index], alpha=self.mask_alpha)
        plt.axis('off')
        
        # Save the plot to a buffer, then open it as a PIL image
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png')
        image = Image.open(img_buf)
        
        # Create a tkinter-compatible image from the PIL image
        new_image = ImageTk.PhotoImage(image)
        img_buf.close()
        plt.close('all')
        
        return new_image
        
    
    def confirm_image(self):
        """ Called when the user selects the 'Suitable' button 
        If checking the mask, saves the mask as usable
        If checking the crop, saves the crop as usable, and then displays the corresponding mask
        """
        if self.checking_mask:
            self.checking_mask = False
            self.usable_masks.append(self.masks[self.current_index])
            self.move_to_next_crop()
        else:
            self.checking_mask = True
            # Save the co-ordinates of the crop within the full image, alongside the class
            crop_info = [self.labels[self.current_index], self.crop_info[self.current_index]]
            self.usable_crops.append(crop_info)
            
            self.instruction_label.configure(text = self.check_mask_str)
            
            # Generate the mask for this image, and display it on the GUI
            mask = self.masks[self.current_index] 
            image = self.images[self.current_index]
            image_with_mask = self.process_images([image, mask])
            
            self.image_label.configure(image = image_with_mask)
            self.image_label.image = image_with_mask
        
        
    def move_to_next_crop(self):
        """ Displays the next crop in the list
        Called if the user clicks the 'Not Suitable' button
        Quits the GUI if all images have been displayed """
        if self.checking_mask:
            # If we were checking the mask of an image, save a 'None' mask to indicate that the corresponding crop does not have a suitable image
            self.usable_masks.append(None)
            self.checking_mask = False
        self.current_index += 1
        if self.current_index >= len(self.images):
            self.window.quit()
        else:
            # Update the label to match the current image
            self.instruction_label.configure(text = self.check_crop_str.format(crop_class = self.labels[self.current_index]))
            
            # Extract the new image and display it on screen
            image = self.process_images([self.images[self.current_index]])
            self.image_label.configure(image = image)
            self.image_label.image = image
    
    def construct_gui(self):
        """ Initialises and places each of the components within the GUI 
        Starts the tkinter infinite loop until the user has labelled each image """
        image = self.images[self.current_index]
        if self.crop_info is not None:
            # We're checking the suitability of individual images
            self.instruction_label = tk.Label(text=self.check_crop_str.format(crop_class = self.labels[self.current_index]))
            image = self.process_images([image])
            self.image_label = tk.Label(image = image)
        else:
            # We're checking the suitability of the full image
            self.instruction_label = tk.Label(text=self.check_mask_str)
            self.checking_mask = True
            mask = self.masks[self.current_index]

            # Place the mask on the image, and display it on screen
            image_with_mask = self.process_images([image, mask])            
            self.image_label = tk.Label(image = image_with_mask)          
            
        # Create the buttons
        self.confirm_button = tk.Button(self.window, text="Suitable", command=self.confirm_image)
        self.reject_button = tk.Button(self.window, text="Not Suitable", command=self.move_to_next_crop)
        
        # Pack each element into the GUI system
        self.instruction_label.pack()
        self.image_label.pack()
        self.confirm_button.pack(side=tk.LEFT)
        self.reject_button.pack(side=tk.RIGHT)
        tk.mainloop()
