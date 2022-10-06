import tkinter as tk
from PIL import Image, ImageTk
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import numpy as np

# Called once per set of crops on an image - alternatively, could call with a list of lists of crops, and once a list has been exhausted, construct the full image from it
class GUI:
    
    def __init__(self, images, masks, info, labels):
        self.window = tk.Tk()
        self.checking_mask = False
        self.images = images
        self.masks = masks
        self.crop_info = info
        self.labels = labels
        self.current_index = 0
        self.check_crop_str = "Is the image below a suitable bounding box around one or more '{crop_class}'?"
        self.check_mask_str = "Are all trainable classes suitably segmented in the image below?"
        self.mask_alpha = round(255*0.4)
        
        # Stores the usable images
        self.usable_crops = []
        self.usable_masks = []
        
        # Components for the GUI
        self.instruction_label = None
        self.image_label = None
        self.confirm_button = None
        self.reject_button = None
        
    def process_images(self, images):
        """Converts the PIL image to a matplotlib plot, then back to a PIL image to provide a nicely-resized version of the image
        Applies any masks that are supplied alongside it on top of the original image"""
        fig = plt.figure()
        plt.imshow(images[0])
        for index in range(1, len(images)):
            plt.imshow(images[index], alpha=0.4)
        plt.axis('off')
        
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png')
        image = Image.open(img_buf)
                    
        new_image = ImageTk.PhotoImage(image)
        img_buf.close()
        plt.close('all')
        
        return new_image
        
    
    def confirm_image(self):
        if self.checking_mask:
            self.checking_mask = False
            self.usable_masks.append(self.masks[self.current_index])
            self.move_to_next_crop()
        else:
            self.checking_mask = True
            crop_info = [self.labels[self.current_index], self.crop_info[self.current_index]]
            self.usable_crops.append(crop_info)
            self.instruction_label.configure(text = self.check_mask_str)
            mask = self.masks[self.current_index] 
            image = self.images[self.current_index]
            image_with_mask = self.process_images([image, mask])
            
            self.image_label.configure(image = image_with_mask)
            self.image_label.image = image_with_mask
        
        
    def move_to_next_crop(self):
        if self.checking_mask:
            self.usable_masks.append(None)
            self.checking_mask = False
        self.current_index += 1
        if self.current_index >= len(self.images):
            self.window.quit()
        else:
            self.instruction_label.configure(text = self.check_crop_str.format(crop_class = self.labels[self.current_index]))
            
            image = self.process_images([self.images[self.current_index]])
            self.image_label.configure(image = image)
            self.image_label.image = image
    
    def construct_gui(self):
        image = self.images[self.current_index]
        if self.crop_info is not None:
            self.instruction_label = tk.Label(text=self.check_crop_str.format(crop_class = self.labels[self.current_index]))
            image = self.process_images([image])
            self.image_label = tk.Label(image = image)
        else:
            # We're checking the suitability of the full image
            self.instruction_label = tk.Label(text=self.check_mask_str)
            self.checking_mask = True
            mask = self.masks[self.current_index]

            image_with_mask = self.process_images([image, mask])            
            self.image_label = tk.Label(image = image_with_mask)          
            
        self.confirm_button = tk.Button(self.window, text="Suitable", command=self.confirm_image)
        self.reject_button = tk.Button(self.window, text="Not Suitable", command=self.move_to_next_crop)
        self.instruction_label.pack()
        self.image_label.pack()
        self.confirm_button.pack(side=tk.LEFT)
        self.reject_button.pack(side=tk.RIGHT)
        tk.mainloop()
