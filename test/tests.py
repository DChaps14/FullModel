from .. import pipeline as pipeline
from .. import run_model as run
from .. import model_detection as detect


#####################################################
# Testing the command line arguments work as expected
#####################################################





#####################################################
# Testing the pipeline can take images of any size and feed them to the model
#####################################################


# Pipeline helper functions work

# Preprocess image converts an image to a fixed size and converts it to a tensor

# Apply greyscale applies greyscale

# Apply random_shift

# Apply apply saturation

# Generate data lists should produce a list of 4 times the size, with images all in aspect ratio 256x256

# establish_datasets should create two datasets of (roughly) len 0.8 and 0.2 the size of the original


#####################################################
# Test detection helpers
#####################################################

# sort_images