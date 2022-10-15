import setup_dataset
import model_training
import model_detection
import execute_gui
import multiprocessing
import argparse
import sys

def parse_args(args):
    """ Extracts the information passed from the command line through flags """ 
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', type=str, nargs="+", help="A list of classes that will be trained", default=[])
    parser.add_argument('--demo_data', type=str, default="True", help='Use demonstration data for the pipeline')
    parser.add_argument('--source', type=str, default="default_yaml", help='Name of the YAML file in which the YOLOv5 training info is stored')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs the model will attempt to train for')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold for returned YOLOv5 predictions')
    parser.add_argument('--input_size', type=int, default=256, help='Square size of the UNet model\'s input layer')
    
    return parser.parse_args(args)

def check_args(args):
    """ Checks whether the arguments parsed from the command line are valid, and commands are used in parallel with each other where they should be """
    use_demo_data = True
    if args.demo_data.upper() != "TRUE":
        use_demo_data = False
        
    if not use_demo_data and (args.classes == [] or args.source == "default_yaml"):
        print("Demo data not used, but no additional information supplied")
        return False
    
    if args.epochs <= 0:
        print("Cannot train on a non-positive number of epochs")
        return False
    elif args.epochs <= 5:
        print("Training on a low number of epochs may limit effectiveness")
    
    if args.confidence < 0 or args.confidence > 1:
        print("Confidence value must be between 0 and 1")
        return False
        
    if args.input_size <= 0:
        print("Input size must be a positive number")
        return False
    
    return use_demo_data, args

def run():
    """ Execute the pipeline. Will run in an infinite loop """
    args = parse_args(sys.argv[1:])
    use_demo_data, args = check_args(args)
    
    if use_demo_data:
        class_dict = {"cat": 0}
         # Install data from the COCO dataset for a demonstration
        setup_dataset.install_dataset(class_dict, 200, 100, args.source)    
    else:
        class_dict = {}
        for target_class in args.classes:
            class_dict[target_class] = len(class_dict)
    
    #Set up the directories needed by parts of the pipeline
    model_detection.establish_directories()
    execute_gui.setup()
    print("Directories established: Beginning pipeline")
    
    num_iterations = 1
    
    # Train the detector and segmentor with the initial training images, after applying transfer learning weights
    # Run with multiprocessing to help with clearing up the GPU RAM as the pipeline executes its steps
    p = multiprocessing.Process(target=model_training.first_train_segmenter(args.epochs, class_dict, input_size))
    p.start()
    p.join()
    p.close()
    p = multiprocessing.Process(target=model_training.train_detector(args.epochs, args.source, "yolov5s.pt"))
    p.start()
    p.join()
    p.close()
    
    while True:
        # DETECTION PORTION
        weight_dir = "exp" if num_iterations == 1 else f"exp{num_iterations}"
        model_detection.detect_detector(f"runs/train/{weight_dir}/weights/best.pt", args.confidence)
        model_detection.detect_segmenter(f"runs/detections/{weight_dir}", class_dict, input_size)
        
        # USER APPROVAL PORTION
        print("Launching GUI")
        execute_gui.launch(class_dict)
        
        print(f"Training and Detection Iteration {num_iterations} completed")
        num_iterations += 1
        
        # FURTHER TRAINING PORTION
        p = multiprocessing.Process(target=model_training.further_train_segmenter(args.epochs, class_dict))
        p.start()
        p.join()
        p.close()    
        p = multiprocessing.Process(target=model_training.train_detector(args.epochs, args.source, f"runs/train/{weight_dir}/weights/best.pt"))
        p.start()
        p.join()
        p.close()
        

if __name__ == "__main__":
    run()