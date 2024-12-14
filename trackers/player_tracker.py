from ultralytics import YOLO 
import cv2
import pickle
import sys
from utils import measure_distance, get_center_of_bbox
sys.path.append('../')

class PlayerTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0] ## the first frame 
        ## court keypoints: 14 keypoints, player_detections_first_frame is the first frame of the players
        ### the function return the 2 players with minimum distances
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        ## loop through the choosen users (id of players)
        ## player detections are the players detected (id, bounding box)
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    ## player_dixt has the id of the player + bounding box of the player 
    def choose_players(self, court_keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            for i in range(0,len(court_keypoints),2): ## each point is a x and y coordinate 
                court_keypoint = (court_keypoints[i], court_keypoints[i+1]) ## court keypoint is the x and y of each keypoint
                ## measure distance between player center and court keypoint
                distance = measure_distance(player_center, court_keypoint) 
                if distance < min_distance:
                    min_distance = distance
            ## id and distance between each player and the minimum keypoint in the court
            distances.append((track_id, min_distance))
        
        # sort the distances in ascending order
        distances.sort(key = lambda x: x[1])
        # Choose the first 2 tracks
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players


    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        ## read from stub if we already detect the players then use it, if not then do object detection 
        player_detections = []

        if read_from_stub and stub_path is not None: ## stub path is in the pickles file
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f) ## load the pickle file 
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict) ## append the frames to the array
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f) ## dump the frames in the pickle file
        
        return player_detections

    def detect_frame(self,frame):
        results = self.model.track(frame, persist=True)[0] ## model is yolo, tracking return: bounding box, id ##presist to give same id for same bounding box
        id_name_dict = results.names ## names of classes

        player_dict = {} ## key is id and value is result
        for box in results.boxes: ## get the bounding boxes
            track_id = int(box.id.tolist()[0]) ## get the tracking id
            result = box.xyxy.tolist()[0] ## get x y position of the bounding box
            object_cls_id = box.cls.tolist()[0] ## class id
            object_cls_name = id_name_dict[object_cls_id] ## class name
            if object_cls_name == "person":
                player_dict[track_id] = result ## id of the player + its result = xy direction
        
        return player_dict

    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames


    