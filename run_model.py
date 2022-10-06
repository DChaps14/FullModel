import setup_dataset
import model_training
import model_detection
import execute_gui
import multiprocessing
import argparse
import sys

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', type=str, nargs="+", help="A list of classes that will be trained", default=[])
    parser.add_argument('--demo_data', type=str, default="True", help='Use demonstration data for the pipeline')
    parser.add_argument('--source', type=str, default="default_yaml", help='Name of the YAML file in which the YOLOv5 training info is stored')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs the model will attempt to train for')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold for returned YOLOv5 predictions')
    
    return parser.parse_args()

def run():
    args = parse_opt()
    use_demo_data = True
    if args.demo_data.upper() != "TRUE":
        use_demo_data = False
        
    if not use_demo_data and (args.classes == [] or args.source == "default_yaml"):
        print("Demo data not used, but no additional information supplied")
        sys.exit()
    
    if use_demo_data:
        class_dict = {"cat": 0}
         # Install data from the COCO dataset as a demonstration
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
    
    p = multiprocessing.Process(target=model_training.first_train_unet(args.epochs, class_dict))
    p.start()
    p.join()
    p.close()
    p = multiprocessing.Process(target=model_training.train_yolo(args.epochs, args.source, "yolov5s.pt"))
    p.start()
    p.join()
    p.close()
    
    while True:
        weight_dir = "exp" if num_iterations == 1 else f"exp{num_iterations}"
        model_detection.detect_yolo(f"runs/train/{weight_dir}/weights/best.pt", args.confidence)
        model_detection.detect_unet(f"runs/detections/{weight_dir}", class_dict)
        
        print("Launching GUI")
        execute_gui.launch(class_dict)
        
        print(f"Training and Detection Iteration {num_iterations} completed")
        num_iterations += 1
        
        p = multiprocessing.Process(target=model_training.further_train_unet(args.epochs, class_dict))
        p.start()
        p.join()
        p.close()    
        p = multiprocessing.Process(target=model_training.train_yolo(args.epochs, args.source, f"runs/train/{weight_dir}/weights/best.pt"))
        p.start()
        p.join()
        p.close()
        

if __name__ == "__main__":
    run()