from tensorflow import keras
from pipeline import UNetTrainingWithFeedback
from UNet import UNet_Named_Layers
from setup_dataset import extract_usables
import subprocess
import yolov5.train as train

def first_train_unet(epochs, class_dict, input_size):
    usable_images, usable_masks = extract_usables(class_dict)
    
    model = UNet_Named_Layers("base_model", decoder_names=False).create_model((input_size, input_size, 3), len(class_dict)+1)
    model.load_weights("model_weights.h5", by_name=True)
    for layer_num in range(19):
        model.layers[layer_num].trainable = False
    
    training_pipeline = UNetTrainingWithFeedback(usable_images, usable_masks, len(usable_images)//10, epochs, input_size)
    model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(), 
            metrics=["accuracy"]
    )
    model = training_pipeline.train(model)
    
    model.trainable = True
    model.compile(
          optimizer=keras.optimizers.Adam(learning_rate=0.0001),
          loss=keras.losses.SparseCategoricalCrossentropy(),
          metrics=["accuracy"]
    )
    model = training_pipeline.train(model)
    model.save("model.h5")
    
def further_train_unet(epochs, class_dict):
    usable_images, usable_masks = extract_usables(class_dict)
    training_pipeline = UNetTrainingWithFeedback(usable_images, usable_masks, len(usable_images)//10, epochs, class_dict, input_size)
    model = keras.models.load_model('model.h5')
    model = training_pipeline.train(model)
    model.save("model.h5")
    
def train_yolo(epochs, data_yaml, weights_location):
    train.run(data=f"{data_yaml}.yaml", batch=32,  epochs=epochs, project='runs/train', weights=weights_location)