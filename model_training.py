from tensorflow import keras
from pipeline import UNetTrainingWithFeedback
from UNet import UNet_Named_Layers
from setup_dataset import extract_usables
import subprocess
import yolov5.train as train

def first_train_unet(epochs, class_dict, input_size):
    """ Runs a training pipeline on U-Net
    This is executed the first time training is done on the U-Net model 
    Inputs:
    epochs - the number of epochs the training should run for
    class_dict - a class dictionary in the format {class_description: label_integer} 
    input_size - the input size of the model"""
    
    # Get the images that we can use to train the model
    usable_images, usable_masks = extract_usables(class_dict)
    
    model = UNet_Named_Layers("base_model", decoder_names=False).create_model((input_size, input_size, 3), len(class_dict)+1)
    # Load the transfer learning weights in, and freeze the decoder and bottleneck
    model.load_weights("model_weights.h5", by_name=True)
    for layer_num in range(19):
        model.layers[layer_num].trainable = False
    
    # Construct the training pipeline, and train it once initially
    training_pipeline = UNetTrainingWithFeedback(usable_images, usable_masks, len(usable_images)//10, epochs, input_size)
    model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(), 
            metrics=["accuracy"]
    )
    model = training_pipeline.train(model)
    
    # Train the model again with fine-tuning
    model.trainable = True
    model.compile(
          optimizer=keras.optimizers.Adam(learning_rate=0.0001),
          loss=keras.losses.SparseCategoricalCrossentropy(),
          metrics=["accuracy"]
    )
    model = training_pipeline.train(model)
    model.save("model.h5")
    
def further_train_unet(epochs, class_dict):
    """ Runs the U-Net model through a training iteration
    This is to be called on every other occasion but the first training instance 
    Inputs:
    epochs - the number of epochs to train the model for
    class_dict - a class dictionary of the format {class_description: label_integer} 
    """
    usable_images, usable_masks = extract_usables(class_dict)
    training_pipeline = UNetTrainingWithFeedback(usable_images, usable_masks, len(usable_images)//10, epochs, class_dict, input_size)
    model = keras.models.load_model('model.h5')
    model = training_pipeline.train(model)
    model.save("model.h5")
    
def train_yolo(epochs, data_yaml, weights_location):
    """ Runs the YOLOv5 model through a training iteration
    Inputs:
    epochs - the number of epochs to run the training for
    data_yaml - the name of the yaml outlining where the data is stored
    weights_location - the path to the file of weights that we want to use for this training run """
    train.run(data=f"{data_yaml}.yaml", batch=32,  epochs=epochs, project='runs/train', weights=weights_location)