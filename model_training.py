from tensorflow import keras
from pipeline import UNetTrainingWithFeedback
from UNet import UNet_Named_Layers
from setup_dataset import extract_usables
import subprocess

def first_train_unet(epochs, class_dict):
    usable_images, usable_masks = extract_usables(class_dict)
    
    model = UNet_Named_Layers("base_model", decoder_names=False).create_model((256, 256, 3), len(class_dict)+1)
    model.load_weights("model_weights.h5", by_name=True)
    for layer_num in range(19):
        model.layers[layer_num].trainable = False
    
    training_pipeline = UNetTrainingWithFeedback(usable_images, usable_masks, len(usable_images)//10, epochs)
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
    training_pipeline = UNetTrainingWithFeedback(usable_images, usable_masks, len(usable_images)//10, epochs, class_dict)
    model = keras.models.load_model('model.h5')
    model = training_pipeline.train(model)
    model.save("model.h5")
    
def train_yolo(epochs, data_yaml, weights_location):
    command = f"python3 yolov5/train.py -- batch 10 --epochs {epochs} -- data {data_yaml} -- weights {weights_location} -- project 'runs/train'"
    process = subprocess.run(command)
    print(process)