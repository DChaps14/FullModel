import pipeline as pipeline
import run_model as run
import model_detection as detect
import sys
from unittest.mock import patch
import unittest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tensorflow import keras
matplotlib.use("TkAgg")


#####################################################
# Testing the command line arguments work as expected
#####################################################
class TestCommandLineParsing(unittest.TestCase):
    
    def test_custom_data_without_classes(self):
        """ Checks that the system won't use custom data if a class list isn't supplied """
        args = run.parse_args(["--demo_data", "False", "--source", "new_yaml"])
        self.assertFalse(run.check_args(args))
            
    def test_custom_data_without_yaml(self):
        """Checks that the system won't use custom data without a supplied yaml file"""
        args = run.parse_args(["--demo_data", "False", "--classes", "cat", "dog"])
        self.assertFalse(run.check_args(args))        
            
    def test_epoch_bounds(self):
        """ Checks that a user can't enter a zero or negative number of epochs """
        args = run.parse_args(["--epochs", "0"])
        self.assertFalse(run.check_args(args))               
            
        args = run.parse_args(["--epochs", "-10"])
        self.assertFalse(run.check_args(args))  

            
    def test_confidence_lower_bound(self):
        """ Checks that the confidence can't be lower than 0 """
        args = run.parse_args(["--confidence", "-0.0000001"])
        self.assertFalse(run.check_args(args))  
            
        args = run.parse_args(["--confidence", "-10"])
        self.assertFalse(run.check_args(args))  
    
    def test_confidence_upper_bound(self):
        """ Checks that the confidence can't be lower than 0 """
        args = run.parse_args(["--confidence", "1.0000001"])
        self.assertFalse(run.check_args(args))          

        args = run.parse_args(["--confidence", "10"])
        self.assertFalse(run.check_args(args))  
            
    def test_input_size_bounds(self):
        """ Checks that the input size of the UNet model cannot be less than 0 """ 
        args = run.parse_args(["--input_size", "0"])
        self.assertFalse(run.check_args(args))
        
        args = run.parse_args(["--input_size", "-1"])
        self.assertFalse(run.check_args(args))
            

#####################################################
# Testing the pipeline helper functions
#####################################################
def dummy_pipeline():
    return pipeline.UNetTrainingWithFeedback([], [], 1, 1, 256)

def generate_images(width, height, mask=False, colour=False, num_images=1):
    image_list = []
    channels = 1 if mask else 3
    for i in range(num_images):
        new_image = np.random.randint(0, 255, (height, width, channels)) if colour else np.zeros((height, width, channels))
        image_list.append(new_image)
    return image_list

class TestPipelineHelpers(unittest.TestCase):

    def test_pipeline_allows_small_image_size(self):
        small_image = generate_images(10, 10)
        small_mask = generate_images(10, 10, mask=True)
        
        pipeline= dummy_pipeline()
        image, mask = pipeline.preprocess_image(small_image, small_mask)
        self.assertEqual(image[0].shape, (256, 256, 3))
        self.assertEqual(mask[0].shape, (256, 256, 1))
        
    def test_pipeline_allows_medium_image_size(self):
        medium_image = generate_images(500, 500)
        medium_mask = generate_images(500, 500, mask=True)
        
        pipeline= dummy_pipeline()
        image, mask = pipeline.preprocess_image(medium_image, medium_mask)
        self.assertEqual(image[0].shape, (256, 256, 3))
        self.assertEqual(mask[0].shape, (256, 256, 1))
        
    def test_pipeline_allows_large_image_size(self):
        large_image = generate_images(1024, 1080)
        large_mask = generate_images(1024, 1080, mask=True)
        
        pipeline= dummy_pipeline()
        image, mask = pipeline.preprocess_image(large_image, large_mask)
        self.assertEqual(image[0].shape,(256, 256, 3))
        self.assertEqual(mask[0].shape, (256, 256, 1))
        
        
    def test_pipeline_applies_greyscale_correctly(self):
        print("Greyscale test")
        colour_image = generate_images(256, 256, colour=True)
        
        plt.imshow(keras.utils.array_to_img(colour_image[0]))
        plt.show()
        
        grey_image = dummy_pipeline().apply_greyscale(colour_image[0])
        
        plt.imshow(keras.utils.array_to_img(grey_image))
        plt.show()
        
    def test_pipeline_applies_shift_correctly(self):
        print("Random shift tests")
        images = generate_images(256, 256, colour=True, num_images=3)
        masks = generate_images(256, 256, mask=True, num_images=3)
        
        pipeline = dummy_pipeline()
        
        for index, image in enumerate(images):
            shift_image, _ = pipeline.apply_random_shift(image, masks[index])
            plt.imshow(keras.utils.array_to_img(shift_image))
            plt.show()
            
    def test_pipeline_applies_saturation_correctly(self):
        print("Saturation Tests")
        images = generate_images(256, 256, colour=True, num_images=3)
        masks = generate_images(256, 256, mask=True, num_images=3)
        
        pipeline = dummy_pipeline()
        
        for index, image in enumerate(images):
            shift_image = pipeline.apply_saturation(image, (index, index))
            plt.imshow(keras.utils.array_to_img(shift_image))
            plt.show()
            
            
    def test_pipeline_adds_augmented_data_from_arrays(self):
        image_arrays = generate_images(256, 256, colour=True, num_images=12)
        mask_arrays = generate_images(256, 256, num_images=12, mask=True)
        
        pipeline = dummy_pipeline()
        image_augments, mask_augments = pipeline.generate_data_lists(image_arrays, mask_arrays)
        self.assertEqual(len(image_augments), 12*4)
        self.assertEqual(len(mask_augments), 12*4)


#####################################################
# Test detection helpers
#####################################################
class TestDetectHelper(unittest.TestCase):
    def test_sort_images_by_area(self):
        large_image = (0, 0, 256, 256)
        medium_image = (0, 0, 100, 100)
        small_image = (0, 0, 50, 50)
        xsmall_image = (0, 0, 10, 10)
        xlarge_image = (0, 0, 500, 500)
        
        unsorted_list = [medium_image, xsmall_image, xlarge_image, large_image, small_image]
        sorted_list = sorted(unsorted_list, key=detect.sort_images)
        self.assertEqual(sorted_list, [xsmall_image, small_image, medium_image, large_image, xlarge_image])




if __name__ == "__main__":
    unittest.main()
