import cv2 as cv
import numpy as np
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite
# from utils import visualize, create_label_dict
import time


'''Detection model'''
# Path to tflite model
model_path = 'models/yolov8_int8.tflite'
# Model input size
input_size = (416, 416)

'''Detection score threshold'''
det_score_thres = 0.6

'''Path to id to label file'''
labelmap_path = 'labelmap.txt'

'''Camera type and orientation'''
res = (640, 480)
is_picamera = True
flip = True


def start_live_detection(cam, 
                         model_path,
                         input_size,
                         score_thres,
                         labelmap_path, 
                         is_picamera=True, 
                         flip=False):
    
    # Creating detector
    detector = create_detector(model_path)  
    detector_output = detector.get_output_details()
    detector_input = detector.get_input_details()[0]

    # Create dictionary to map class ID to class name
    # id2name_dict = create_label_dict(labelmap_path)

    # Picamera
    cam.start()
    print ('Camera is running')

    
    while(True):

        try:
            t1 = time.time()

            '''Capture'''
            # picamera
            frame_ori = cam.capture_array()

            # Flip
            if flip:
                frame_ori = cv.rotate(frame_ori, cv.ROTATE_180)

            frame = frame_ori.copy()

            ''' Preprocess '''
            # Convert BGR to RGB
            # frame = frame[:,:,::-1]
            # Resize the frame to match the model input size
            frame = cv.resize(frame, input_size).astype('int8')
            #frame = np.transpose(frame, (2, 0, 1)) 
            #print (frame.shape)
            frame = np.expand_dims(frame, axis=0)

            # ''' Run object detection '''
            detector.set_tensor(detector_input['index'], frame)
            detector.invoke()
            # Bounding boxes coordinates
            # output = detector_output[1]# ['index']
            # print (output)
            outputs = detector.get_tensor(detector_output[1]['index'])
            # outputs = np.transpose(np.squeeze(outputs[0]))
            # rows = outputs.shape[0]
            # print (rows)
            # classes_scores = outputs[0][4:]
            # print (classes_scores)
            # Detected objects class ID
            #class_ids = detector.get_tensor(detector_output[3]['index'])[0]
            # Detection scores
            #scores = detector.get_tensor(detector_output[0]['index'])[0]
            #scores = [round(score, 2) for score in scores]

            # if len(bboxes) > 0:

            #     # Draw the detection result
            #     frame_ori = visualize(frame_ori, 
            #                         bboxes, 
            #                         class_ids, 
            #                         scores, 
            #                         score_thres, 
            #                         id2name_dict, 
            #                         color='rgb',
            #                         model_type='efficientdet')
        
            frame_ori = frame_ori[:,:,::-1]

            # Display the resulting frame
            cv.imshow('frame', frame_ori)

            # the 'q' button is set as the
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            t2 = time.time()
            print (f'frame_time: {t2-t1}')

        except Exception as e:
            print (e)
            # On error, release the camera object
            cam.stop()
            break

    # After the loop release the cap object
    if not is_picamera:
        cam.release()
    # Destroy all the windows
    cv.destroyAllWindows()


def create_detector(model_path):
    # Initialize the object detector
    detector = tflite.Interpreter(model_path)
    # Allocate memory for the model's input `Tensor`s
    detector.allocate_tensors()
    return detector


if __name__ == '__main__':
    
    '''Setting up and configure the camera'''
    # Picamera
    cam = Picamera2()
    config = cam.create_preview_configuration(main={"size": res, "format": "BGR888"})
    cam.configure(config)

    # Start camera
    start_live_detection(cam, 
                         model_path, 
                         input_size, 
                         det_score_thres, 
                         labelmap_path,
                         flip=flip)
    print ('end')