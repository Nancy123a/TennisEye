import numpy as np

def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    center_x = int((x1 + x2)/2)
    center_y = int((y1 + y2)/2)
    return (center_x,center_y)

## eculidean distance (measure distance between 2 objects) 
def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)

## point: position of foot of player
## keypoints: all the court keypoints
## keypoints_indices: keypoints that u need to use to calculate the distance
def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    closest_distance = float('inf')
    key_point_ind = keypoint_indices[0]
    
    for keypoint_indix in keypoint_indices:
        keypoint = keypoints[keypoint_indix * 2], keypoints[keypoint_indix * 2 + 1]
        distance = abs(point[1] - keypoint[1])
        
        if distance < closest_distance:
            closest_distance = distance
            key_point_ind = keypoint_indix
    
    return key_point_ind


## get the player height which is the difference between the upper and lower y position of player bounding box
def get_height_of_bbox(bbox):
    return bbox[3]-bbox[1]

## return x and y position between the foot of player and closest keypoint
def measure_xy_distance(p1,p2):
    return abs(p1[0]-p2[0]), abs(p1[1]-p2[1])

## get the center of bbox x and y position
def get_center_of_bbox(bbox):
    return (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))


