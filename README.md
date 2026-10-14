# Raspberry Pi Animal Detection
This Python program captures pictures on a Raspberry Pi 5, identifies animals using 3 pretrained AI models, and saves images that pass an image-quality check. 

# Equipment: 

  Hardware: Raspberry Pi 5, Camera Module 3
  
  Software: Raspberry Pi OS, Python, Picamera2, OpenCV, TorchVision, PyTorch
  
# Workflow:
1. Capture one frame from the camera.
2. Analyze the same frame sequentially with RetinaNet, Faster R-CNN MobileNetV3, and      MobileNetV3 Large.
3. Accept the frame as containing an animal when at least two models vote "YES".
4. Check sharpness within the detected animal region.
5. Save the image if it passes the sharpness threshold and the save interval requirement

# Functionalities:
1. 3-model voting system
2. Continuous autofocus and live camera preview window
3. Image sharpness filtering mechanism
4. Configurable detection thresholds and save intervals
5. CPU temperature monitoring and will stop working if it reaches 70 Celcus. 
6. Press "E + Enter" to stop new captures while allowing the current frame to finish         processing, subject to quality and safety checks

# AI Usage:
AI tools assisted with code debugging and development. 

# Setup Note:
1. The repository contains the main application script. It requires a separate RetinaNet project containing references. utils, configuration files, label files, and compatible model weights.
2. Install the required libraries in a compatible Raspberry Pi environment.
3. Set RETINANET_ROOT to the local RetinaNet project directory.
4. Set SAVE_FOLDER to the desired output directory.
5. Use a desktop session that supports the QTGL camera preview.

# Limitations:
Detection depends on lighting and the animal categories that are supported by the models. False positives and missed detections are possible. 

# Acknowledgments:
This project uses pretrained models and third-party libraries. Their code and model weights remain subject to their respective licenses and terms.


