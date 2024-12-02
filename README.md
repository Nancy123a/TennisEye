# TennisEye
## Group Members: Nancy Bou Kamel

## Motivation/Idea:
An advanced analytical platform that utilizes computer vision to study ball dynamics and player interactions, providing deep insights for enhanced game performance. This Tennis Analysis System is designed to provide real-time object detection for key elements of a tennis match, including the ball and two players. The system captures and analyzes vital metrics such as ball and shot speed, average player speed for both competitor. Moreover, the system detects the tennis court's background and applies homography mapping to maintain spatial awareness and improve accuracy. 

## Related Work
Messelodi et al. developed a low-cost vision-based system for real-time tennis analysis using four synchronized cameras to track players and the ball. The system detects events like shots, bounces, strokes, and line calls, offering live feedback and post-session reviews. Field tests in Italy showed high reliability, with 99.7% shot detection, 97.1% stroke classification accuracy, and 99.5% line-calling accuracy, with a response time of 152 ms, demonstrating its effectiveness for accessible tennis analytics. (Link: https://www.researchgate.net/publication/335605903_A_Low-Cost_Computer_Vision_System_for_Real-Time_Tennis_Analysis).

### Training Datasets: 
- Tennis ball detection: https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection
- Tennis court detector : https://github.com/yastrebksv/TennisCourtDetector

### Planned Implementation / Evaluation
- Player Detection Using YOLOv8: Leverages YOLOv8 to accurately identify players in tennis footage, employing predictive algorithms to maintain tracking.
- Speed Calculation: Implements algorithms to calculate the speeds of players and the ball.
- Employ PyTorch to train a Convolutional Neural Network (CNN) for keypoint extraction.
- Implement object trackers to monitor objects across multiple frames.
- Work with OpenCV (CV2) to handle video reading, manipulation, and saving.
- Evaluate detection data and use a data-driven approach for feature development.
- Combine the outputs of these ML/DL models into a cohesive project with tangible results.
- Dynamic Miniature Court Representation: Creates a scaled-down model of the tennis court that reflects player movements in real-time.

### Requirements:
- python3.8
- ultralytics
- pytroch
- pandas
- numpy
- opencv
