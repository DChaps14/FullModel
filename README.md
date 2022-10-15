# SICLOPS: Sets of Images Created and Labelled using Object Perception and Segmentation

# Result of Pipeline
After the completion of a number of iterations of the pipeline can provide both an instance-segmented dataset, as well as an object detector and semantic segmenter that have been suitably trained to generate this dataset. The user can either extract the dataset to train additional models, or extract either of the models to utilise their already trained weights.

# Running the pipeline
## Prior Dependencies
The weights for the transfer learning can be found in the following OneDrive folder: https://1drv.ms/u/s!AhDiTpIEQVOS_XjTor-DALb3HzB2?e=CFhQTo

Apart from this repository, the operation of the pipeline requires a clone of the modified YOLOv5 model: https://github.com/DChaps14/yolov5. This repository needs to be cloned within the 'FullModel' directory, housing the model's python scripts.

To install the required dependecies for the pipeline, run the following commands

``cd FullModel``

``python3 -m pip install -r requirements.txt``

``python3 -m pip install -r yolov5/requirements.txt``

The weights that are utilised by the segmenter are located within a 7zip file 

## Setting up images for training
To utilise our own dataset, the training images need to be established in a particular way, such that they can be accessed by the YOLOv5 model.
The YOLOv5 model needs a yaml file that outlines how many classes need to be trained, what these classes are. The format should be as follows:

``train: ../UNetPredictions/yolov5/images/train``

``val: ../UNetPredictions/yolov5/images/val``

``nc: <the number of classes to find in this pipeline>``

``names: [<each of the names of the classes seperated by commas>]``

This file needs to be stored within the 'data' directory of the 'yolov5' model.

The images that we want to utilise for initial training data needs to be stored in the following directory path in jpg format: ``FullModel\UNetPredictions\usableImages\images``. 

The masks for these images need to be stored in the following json format:

``{filename: <The name of the file that the mask relates to>, ground_truth: {detections: [{bounding_box: <The bounding box of the detection in xyxy format>, mask: <An array of shape (bbox_height, bbox_width, 1) containing the mask of the instances within the bounding box>, label: <The label of the bounding box>}...]}}``

The masks should have the same filename as the image they relate to, and should be stored in the ``FullModel\UNetPredictions\usableImages\images`` directory.

Any test images to be used need to be placed in the following directory, in jpg format: ``FullModel\UNetPredictions\testImages``

### Sidenote: Mask format

A mask within the .json file should look something like the following:

``[[[0], [0], [1], ..., [0]],``

``[[0], [0], [1], ..., [1]],``

``...,``

``[[0], [0], [2], ..., [1]]]``

where [0] denotes a pixel with the background label, [1] denotes a pixel with the label of class 1, and so on.

## Command Line Arguments
To run the pipeline with sample data, execute the following command in the command line of the FullModel directory:
``python3 run_model.py``

Additional information can be passed to the pipeline to utilise your own images, or change elements of the pipeline. The following arguments can be appended to the above command

| Flag | Description | Available Values | Notes |
| ---- | ----------- | ---------------- | -------------- |
| --classes | A list of the classes that will be trained | Pass in as many space-seperated names as there are classes in the training dataset | Will have no effect if --demo_data is not ``False`` |
| --demo_data | Whether to use demonstration data for the pipeline | ``True`` or ``False`` | If ``False``, requires --classes, --source | 
| --source | The name of the yaml file stored in yolov5/data that holds the YOLOv5 training data | Any | Will have no effect if --demo_data is not ``False`` |
| --epochs | The number of epochs the pipeline will train each model for | Any positive integer (recommended above 5) | Code will exectue if value is below 5, but higher epochs may allow better results |
| --confidence | The minimum confidence for the YOLOv5 predictions that are returned | Any floating point between 0 and 1 | None |
| --input_size | The square size the pipeline will reshape the UNet model's input images to | Any integer | Higher values may increase execution time and memory cost |

# Further Considerations
## Utilising Alternate Models
The pipeline has been designed in a way that aims to allow for easy extension with alternate models - if a different segmentation model is desired, or a different object detection model is needed, then the developer should only need to modify the methods within model_detection and model_training to fit with these new models. Therefore, as long as the new models are established to utilise the current structure of the training data, this pipeline should theoretically allow for plug-and-play model switching.

## Future Additions
The initial extension that I would recommend is the deployment of this pipeline to a cloud service provider. A large amount of storage space is necessary to store the images that the model utilises for both prediction and training, and the pipeline requires GPU intensive operations to train each model that any user's computer might not be able to accomplish. Hence, to ensure that the pipeline is widely available and usable by any client, an instance may need to be established on-cloud for wider use.

A Generative Adversarial Network may be beneficial if added to the pipeline. As GANs attempts to train a classifier by creating images that should theoretically be indistinguishable from real images, such a model could be implemented into the pipeline in an attempt to generate more images to be used to train the model.

Finally, a correction funcionality could be added to the GUI - currently, the user does not have a way to provide feedback to the model other than whether or not it was suitable. Such a mechanism could allow a user to rectify any mistakes that the model has made in its predictions, allowing for a faster completion and generation of the dataset. Three options for correction stand out to me at this point
- An image is pushed to labelstudio if the pipeline doesn't provide suitable results from it after a set number of iterations
- Buttons are added to the GUI that the user can utilise to inform the model on why the image is not suitable. This feedback can help the pipeline to either improve its training processes, or inform the user on the best steps forward.
- The user is able to 'touch up' the images presented to them, by shifting or adding bounding boxes, or removing or adding areas of classification to the images that are presented to them.

These additions may make the pipeline more of a tool to help generate a dataset, instead of a tool to completely generate the dataset for the user. However, while they may increase the amount of work the user has to do, it may also decrease the time or images needed to generate a new dataset.

## Design Decisions
- The validation split is randomly generated for each iteration of the pipeline. This was done to ensure that the pipeline saw as many images in its training dataset as possible across each iteration.
- After completing a Labelstudio operation, the user does not have to download the instance manually and add it to the training dataset. Instead, the image is downloaded automatically by simulating a click and extracting it from the Downloads folder of the current user. This was done to allow smooth operation of the pipeline, as well as to decrease the amount of human interaction.
- The notebooks within the following Google Drive folder should outline the decision making process in why techniques or models were added to the pipeline: https://drive.google.com/drive/folders/1Ily8QIiTOLN2Cy7dTw2pGh8_NBIzGF_u?usp=sharing
