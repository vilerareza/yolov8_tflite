import cv2 as cv
import numpy as np
import tflite_runtime.interpreter as tflite
import time
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml
import math
import gc
from tqdm import tqdm

'''Video path'''
video_path = 'videos/2.mp4'

'''Detection model'''
# Path to tflite model
model_path = 'models/yolov8n_int8_def.tflite'
# Model input size
input_size = (640, 640)

'''Detection score threshold'''
score_thres = 0.3
iou_thres = 0.3


class YoloV8Detector:


    def __init__(self, 
                 model_path, 
                 input_size) -> None:

        self.model_path = model_path
        self.input_size = input_size

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
                             video_path,
                             score_thres,
                             iou_thres):
        
        # Retrieve metadata 
        # fps
        cap = cv.VideoCapture(video_path)
        fps = math.ceil(cap.get(cv.CAP_PROP_FPS))
        n_frame = math.ceil(cap.get(cv.CAP_PROP_FRAME_COUNT))
        # w x h
        ret, test_frame = cap.read(0)
        video_w = test_frame.shape[1]
        video_h = test_frame.shape[0]

        # # Writer object
        writer = cv.VideoWriter('output.avi', cv.VideoWriter_fourcc(*'MJPG'), fps, (video_w, video_h)) 

        # Creating detector
        self.detector = self.create_detector(self.model_path)  
        self.detector_output = self.detector.get_output_details()
        self.detector_input = self.detector.get_input_details()[0]

        # Frame processing and writing the result to temp file
        # for idx, frame_ori in tqdm(enumerate(video_gen)):
        for i in tqdm(range(500)):

            ret, frame_ori = cap.read()

            

            if ret:
                frame = frame_ori.copy()
                # BGR to RGB
                frame = frame[:,:,::-1]
                ''' Preprocess '''
                # Resize the frame to match the model input size
                frame = cv.resize(frame, input_size)
                frame = cv.normalize(frame, None, -128, 127, cv.NORM_MINMAX, dtype=cv.CV_8S)
                frame = np.expand_dims(frame, axis=0)

                # ''' Run object detection '''
                self.detector.set_tensor(self.detector_input['index'], frame)
                self.detector.invoke()
                
                # Output handling
                output = self.detector.get_tensor(self.detector_output[0]['index'])            
                frame_ori = self.postprocess(frame_ori, output, score_thres, iou_thres)

                #frame_ori = frame_ori[:,:,::-1]

                writer.write(frame_ori)

                del frame_ori
                gc.collect()
            
            else:
                break

        cap.release() 
        writer.release() 


    def postprocess(self, image, output, score_thres, iou_thres):

        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

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
                x = (x+128)/255 ; y = (y+128)/255 ; w = (w+128)/255  ; h = (h+128)/255 
                left = int((x - w / 2) * image.shape[1])
                top = int((y - h / 2) * image.shape[0])
                width = int(w * image.shape[1])
                height = int(h * image.shape[0])

                # if left<0:
                #     left=0
                # elif left>image.shape[1]:
                #     left=image.shape[1]
                # if top<0:
                #     top=0
                # elif top>image.shape[0]:
                #     top=image.shape[0]
                # if width<0:
                #     width=0
                # elif width>image.shape[1]:
                #     width=image.shape[1]
                # if height<0:
                #     height=0
                # elif height>image.shape[0]:
                #     height=image.shape[0]
                # print (left, top, width, height)
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
        return image


if __name__ == '__main__':
    
    # Initiate detector
    yolo_detector = YoloV8Detector(model_path=model_path,
                                   input_size=input_size)
    # Start detection
    yolo_detector.start_live_detection(video_path=video_path,
                                       score_thres=score_thres,
                                       iou_thres=iou_thres)
    print ('end')