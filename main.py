from utils import (read_video, save_video)
from trackers import PlayerTracker,BallTracker
from court_line_detector import CourtLineDetector
#from mini_court import MiniCourt
import cv2 

def main():
    #read video
    input_video_path = 'input/input_video.mp4'
    video_frames = read_video(input_video_path)
    
    # detect players and Ball
    player_tracker = PlayerTracker(model_path="yolov8x")
    ball_tracker = BallTracker(model_path="models/best.pt")

    player_detections = player_tracker.detect_frames(video_frames , read_from_stub=True , 
                                                     stub_path="tracker_stubs/player_detections.pkl")

    ball_detections = ball_tracker.detect_frames(video_frames,read_from_stub=True,stub_path="tracker_stubs/ball_detections.pkl")
    ## ball detections is a dictionary of id, and the coordinates of the bounding box

    ball_detections=ball_tracker.interpolate_ball_positions(ball_detections)

    court_model_path = "models/keypoints_model.pth"
    court_line_detector= CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    player_detections = player_tracker.choose_and_filter_players(court_keypoints,player_detections)

    # draw bboxes
    output_video_frames = player_tracker.draw_bboxes(video_frames,player_detections)
    output_video_frames= ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    ## Draw court keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames,court_keypoints)

    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame,f"Frame: {i}",(10,110),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    save_video(output_video_frames,"output/output_main.avi")


if __name__ == "__main__":
    main()