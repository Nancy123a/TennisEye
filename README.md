# TennisEye
## Group Members: Nancy Bou Kamel

## Motivation/Idea:
An advanced analytical platform that utilizes computer vision to study ball dynamics and player interactions, providing deep insights for enhanced game performance. This Tennis Analysis System is designed to provide real-time object detection for key elements of a tennis match, including the ball and two players. The system captures and analyzes vital metrics such as ball and shot speed, average player speed for both competitor. Moreover, the system detects the tennis court's background and applies homography mapping to maintain spatial awareness and improve accuracy, even in situations where parts of the court are occluded or when players or the ball are temporarily hidden from view. This capability ensures reliable data collection and performance insights, regardless of occlusions.

## Related Work

The Tennis Analysis System leverages advancements in computer vision and sports analytics, utilizing real-time object detection frameworks like YOLO and Faster R-CNN to track players and the ball during matches. Insights from user interface design create an intuitive dashboard for easy access to performance metrics.

This project aims to implement this system and enhance it by incorporating occlusion management techniques. This will improve accuracy even when players or the ball are temporarily obscured, allowing for more reliable gameplay insights. 

### Training Datasets: 
- UCF YouTube Action Data Set https://www.kaggle.com/datasets/pypiahmad/ucf-youtube-action-data-set?resource=download
- Youtube video will be downloa as mp4:
  - https://www.youtube.com/shorts/ycklQ9Z9E0s
  - https://www.youtube.com/shorts/MI2yKJFcM6I
  - https://www.youtube.com/shorts/Ncrmzf0rgUA

### Planned Implementation / Evaluation
- Player Detection Using YOLOv8: Leverages YOLOv8 to accurately identify players in tennis footage, employing predictive algorithms to maintain tracking even when players are briefly occluded.

- Enhanced Ball Detection: Refines the YOLO model for precise tennis ball detection in various conditions, using trajectory estimation techniques to predict the ball's path during occlusions.

- Speed Calculation: Implements algorithms to calculate the speeds of players and the ball, ensuring robust measurements by interpolating speeds based on previous and subsequent positions during occlusions.

- Dynamic Miniature Court Representation: Creates a scaled-down model of the tennis court that reflects player movements in real-time, adjusting the display when occlusions occur to maintain an accurate representation.

- Key Point Extraction with ResNet50: Customizes ResNet50 to extract key points on the court, utilizing historical data to infer locations during occlusions, thereby enabling continuous analysis of player strategies.
