import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import random

""" A class utilised to manipulate the training data, and train the U-Net model """
class UNetTrainingWithFeedback():

    def __init__(self, image_dataset, mask_dataset, batch_size, epochs, input_size):
        """
        Inputs:
        image_dataset - a dataset housing the images we can use for training
        mask_dataset - a dataset housing the masks we can use for training
        batch_size - the batch size we want to use for the training data
        epochs - the number of epochs we want to run training for
        input_size - the input_size of the model
        """
        self.batch_size = batch_size
        self.buffer_size = 1000
        self.steps_per_epoch = round(0.8*len(image_dataset) / self.batch_size) # Set the total number of steps that will be done per epoch       
        self.epochs = epochs
        self.val_steps = round(0.2*len(image_dataset) / self.batch_size)
        self.input_size = input_size
        # Establish the batches for training and testing
        self.train_batches, self.validation_batches = self.establish_datasets(image_dataset, mask_dataset)


    def preprocess_image(self, image, mask):
        """ Converts the image and mask to tensors, resizes them to the model's input size, and resizes the colour values to be decimal """
        image_tens = tf.convert_to_tensor(image, dtype=tf.float32)
        image_tens = tf.image.resize(image_tens, [self.input_size, self.input_size])
        image_tens = image_tens / 255.0

        mask_tens = tf.convert_to_tensor(mask, dtype=tf.float32)
        mask_tens = tf.image.resize(mask_tens, [self.input_size, self.input_size])

        return image_tens, mask_tens

    def apply_greyscale(self, image):
        """Convert the image to greyscale"""
        mod_image = tf.image.rgb_to_grayscale(image)
        # Convert the image to 3-channel
        mod_image = tf.repeat(mod_image, repeats=[3], axis=2)
        return mod_image

    def apply_random_shift(self, image, mask):
        """Shift the image a random number of bits vertically and horizontally"""
        image = np.array(image)
        mask = np.array(mask)
        image_width = len(image[0])
        image_height = len(image)

        # Get a random integer between 0 and half the images width/height. Then, randomly choose whether to move it to the left or right and up or down
        ver_shift = random.randrange(image_height//4, image_height//2)
        hor_shift = random.randrange(image_width//4, image_width//2)
        shift_right = random.randrange(2)
        shift_up = random.randrange(2)

        # Depending on how the shift is occuring, extract the pixels to retain, then determine the padding to use
        if shift_up and shift_right:
            horizontal_change = image_width-hor_shift
            mod_image = image[ver_shift:, :horizontal_change]
            mod_mask = mask[ver_shift:, :horizontal_change]
            height_padding = [0, ver_shift]
            width_padding = [hor_shift, 0]
        elif shift_up:
            mod_image = image[ver_shift:, hor_shift:]
            mod_mask = mask[ver_shift:, hor_shift:]
            height_padding = [0, ver_shift]
            width_padding = [0, hor_shift]
        elif shift_right:
            vertical_change = image_height-ver_shift
            horizontal_change = image_width-hor_shift
            mod_image = image[:vertical_change, :horizontal_change]
            mod_mask = mask[:vertical_change, :horizontal_change]
            height_padding = [ver_shift, 0]
            width_padding = [hor_shift, 0]
        else:
            vertical_change = image_height-ver_shift
            mod_image = image[:vertical_change, hor_shift:]
            mod_mask = mask[:vertical_change, hor_shift:]
            height_padding = [ver_shift, 0]
            width_padding = [0, hor_shift]

        mod_image = tf.convert_to_tensor(mod_image)
        mod_mask = tf.convert_to_tensor(mod_mask)

        # Pad the extracted pixels to full input size
        paddings = tf.constant([height_padding, width_padding, [0,0]])
        mod_image = tf.pad(mod_image, paddings, 'CONSTANT')
        mod_mask = tf.pad(mod_mask, paddings, 'CONSTANT')

        return mod_image, mod_mask

    def apply_saturation(self, image, seed):
        """Adjusts the saturation of the image to a random value
        Inputs:
        image - the image to augment
        seed - a seed for the random generator of shape (2,)"""
        mod_image = tf.image.stateless_random_saturation(image, lower=0.1, upper=0.9, seed=seed)
        return mod_image

    def generate_data_lists(self, image_arrays, mask_arrays):
        """Creates a list of tensors that can be converted to a tf Dataset to feed into the model
        Takes each image in the list and sends it to each of the augment functions, then appends each of these into the overall image list """
        image_list, mask_list = [], []
        for index, image in enumerate(image_arrays):
            image, mask = self.preprocess_image(image, mask_arrays[index])
            sat_image = self.apply_saturation(image, (index, index))
            shift_image, shift_mask = self.apply_random_shift(image, mask)
            grey_image = self.apply_greyscale(image)
            image_list += [image, sat_image, shift_image, grey_image]
            mask_list += [mask, mask, shift_mask, mask]

        return (image_list, mask_list)

    def establish_datasets(self, image_list, mask_list):
        """Split the images and masks into training and validation splits, and generate dataset values for each of them"""
        
        # Shuffle each of the lists the same way to achieve a new split of validation
        seed = random.random()
        random.shuffle(image_list, lambda: seed)
        random.shuffle(mask_list, lambda: seed)
        split_index = round(0.8*len(image_list))
        train_images = image_list[:split_index]
        train_masks = mask_list[:split_index]
        val_images = image_list[split_index:]
        val_masks = mask_list[split_index:]


        training_dataset = tf.data.Dataset.from_tensor_slices(self.generate_data_lists(train_images, train_masks))
        validation_dataset = tf.data.Dataset.from_tensor_slices(self.generate_data_lists(val_images, val_masks))

        training_batches = (
            training_dataset
            .cache()
            .shuffle(self.buffer_size)
            .batch(self.batch_size)
            .repeat() # Makes the batch set infinite - hence we need to specify the steps_per_epoch value
            .prefetch(buffer_size=tf.data.AUTOTUNE))

        validation_batches = (
            validation_dataset
            .cache()
            .shuffle(self.buffer_size)
            .batch(self.batch_size)
            .repeat()
            .prefetch(buffer_size=tf.data.AUTOTUNE))

        return training_batches, validation_batches

    def train(self, model):
        """ Train the model, using the pipelines pre-defined variables """
        callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
        model.fit(self.train_batches, epochs=self.epochs, steps_per_epoch=self.steps_per_epoch, 
                          validation_data=self.validation_batches, validation_steps=self.val_steps, callbacks=callback)
        return model