import cv2
import numpy as np
import sys
sys.path.append('../')
import constants
from utils import convert_pixel_distance_to_meters, convert_meters_to_pixel_distance

class MiniCourt():
    def __init__(self,frame):
        


    
    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        