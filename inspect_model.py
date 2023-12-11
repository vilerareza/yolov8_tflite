import cv2 as cv
import numpy as np
import tflite_runtime.interpreter as tflite
# from utils import visualize, create_label_dict
import time


'''Detection model'''
# Path to tflite model
model_path = 'models/yolov8n_int8_def.tflite'

# Show input output details
interpreter = tflite.Interpreter(model_path)
input_type = interpreter.get_input_details()[0]
print('input: ', input_type)
output_type = interpreter.get_output_details()
print('output: ', output_type)