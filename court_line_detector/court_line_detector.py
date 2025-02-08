import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
from torchvision.models import ResNet50_Weights
import numpy as np

class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2) 
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        # Step 1: Preprocess and predict keypoints
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        keypoints = outputs.squeeze().cpu().numpy()
        
        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0
    
        # Step 2: Harris corner detection
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_float = np.float32(gray_image)
        harris_response = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
    
        # Step 3: Find strong corners
        harris_response = cv2.dilate(harris_response, None)
        threshold = 0.01 * harris_response.max()
        corners = np.argwhere(harris_response > threshold)  # Corners are (y, x)
        corners_xy = corners[:, [1, 0]]
    
        # Step 4: Calculate minimum distances and closest corners
        closest_corners = []  # To store the closest corner for each keypoint
        
        # Loop through each keypoint (x, y)
        for i in range(0, len(keypoints), 2):
            kp_x = keypoints[i]
            kp_y = keypoints[i + 1]
            
            # Calculate Euclidean distance from this keypoint to all corners
            distances = np.sqrt((corners_xy[:, 0] - kp_x)**2 + (corners_xy[:, 1] - kp_y)**2)  # corners[:, 1] = y, corners[:, 0] = x
            
            # Find the index of the minimum distance
            min_idx = np.argmin(distances)
            
            # Get the closest corner coordinates
            closest_corner = corners_xy[min_idx]
            
            # Append the closest corner to the list
            closest_corners.append(closest_corner)
    
        return closest_corners


    def draw_keypoints(self, image, keypoints):
        # Plot keypoints on the image
        for index,(x,y) in enumerate(keypoints):
            cv2.putText(image, str(index), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 8, (0, 0, 255), -1)
        return image

    
    def draw_keypoints_on_video(self, video_frames, keypoints,closest_keypoints):
        output_video_frames = []
        for frame_num,frame in enumerate(video_frames):
            frame=self.draw_keypoints(frame,keypoints)
            output_video_frames.append(frame)
        return output_video_frames

    def draw_closest_keypoint(self,video_frames,keypoints):
        for frame_num,frame in enumerate(video_frames):
            keypoints[frame_num] = [keypoints[frame_num]]
            for close_x,close_y in keypoints[frame_num]:
                cv2.circle(frame, (close_x, close_y), 8, (0, 255, 0), -1)
        return video_frames


    