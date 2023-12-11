import cv2 as cv
import numpy as np
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite
# from utils import visualize, create_label_dict
import time
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml


'''Detection model'''
# Path to tflite model
model_path = 'models/yolov8n_int8_def.tflite'
# Model input size
input_size = (640, 640)

'''Detection score threshold'''
score_thres = 0.5
iou_thres = 0.5

'''Path to id to label file'''
labelmap_path = 'labelmap.txt'

'''Camera type and orientation'''
res = (640, 480)
flip = True


class YoloV8Detector:


    def __init__(self, 
                 model_path, 
                 input_size,
                 camera_res) -> None:

        self.model_path = model_path
        self.input_size = input_size
        self.camera_res = camera_res 

        '''Setting up and configure the camera'''
        # Camera
        self.cam = Picamera2()
        config = self.cam.create_preview_configuration(main={"size": camera_res, "format": "BGR888"})
        self.cam.configure(config)
        
        self.classes = yaml_load(check_yaml('coco128.yaml'))['names']
        # Color Palette
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        


    def create_detector(self, model_path):
        # Initialize the object detector
        detector = tflite.Interpreter(model_path)
        # Allocate memory for the model's input `Tensor`s
        detector.allocate_tensors()
        return detector


    def start_live_detection(self, 
                             score_thres,
                             iou_thres,
                             flip=False):
        
        # Creating detector
        self.detector = self.create_detector(self.model_path)  
        self.detector_output = self.detector.get_output_details()
        self.detector_input = self.detector.get_input_details()[0]

        # Create dictionary to map class ID to class name
        # id2name_dict = create_label_dict(labelmap_path)

        # Picamera
        self.cam.start()
        print ('Camera is running')

    
        while(True):

            try:
                t1 = time.time()

                '''Capture'''
                # picamera
                frame_ori = self.cam.capture_array()

                # Flip
                if flip:
                    frame_ori = cv.rotate(frame_ori, cv.ROTATE_180)

                frame = frame_ori.copy()

                ''' Preprocess '''
                # Convert BGR to RGB
                # Resize the frame to match the model input size
                frame = cv.resize(frame, input_size).astype('int8')
                frame = cv.normalize(frame, None, -128, 127, cv.NORM_MINMAX, dtype=cv.CV_8S)
                frame = np.expand_dims(frame, axis=0)

                # ''' Run object detection '''
                self.detector.set_tensor(self.detector_input['index'], frame)
                self.detector.invoke()
                
                # Output handling
                output = self.detector.get_tensor(self.detector_output[0]['index'])
                
                frame = self.postprocess(frame, output, score_thres, iou_thres)

                frame_ori = frame_ori[:,:,::-1]

                # Display the resulting frame
                cv.imshow('frame', frame)

                # the 'q' button is set as the
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

                t2 = time.time()
                #print (f'frame_time: {t2-t1}')

            except Exception as e:
                print (e)
                # On error, release the camera object
                self.cam.stop()
                break

        # Destroy all the windows
        cv.destroyAllWindows()


    def postprocess(self, image, output, score_thres, iou_thres):

        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.camera_res[0] / self.input_size[0]
        y_factor = self.camera_res[1] / self.input_size[1]

        # Iterate over each row in the outputs array
        for i in range(rows):

            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]
            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)
            # Normalize the max_score
            max_score = (max_score-(-128))/255

            # If the maximum score is above the confidence threshold
            if max_score >= score_thres:

                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)
                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv.dnn.NMSBoxes(boxes, scores, score_thres, iou_thres)

        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # Draw the detection on the input image
            image = self.draw_detections(image, box, score, class_id)

        # Return the modified input image
        return image


    def draw_detections(self, image, box, score, class_id):

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box
        # Retrieve the color for the class ID
        color = self.color_palette[class_id]
        # Draw the bounding box on the image
        cv.rectangle(image, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
        # Create the label text with class name and score
        label = f'{self.classes[class_id]}: {score:.2f}'
        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        # Draw a filled rectangle as the background for the label text
        cv.rectangle(image, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                      cv.FILLED)
        # Draw the label text on the image
        cv.putText(image, label, (label_x, label_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)


if __name__ == '__main__':
    
    # Initiate detector
    yolo_detector = YoloV8Detector(model_path=model_path,
                                   input_size=input_size,
                                   camera_res=res)
    # Start detection
    yolo_detector.start_live_detection(score_thres=score_thres,
                                       iou_thres=iou_thres,
                                       flip=flip)
    print ('end')