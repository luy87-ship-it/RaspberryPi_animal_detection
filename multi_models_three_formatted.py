# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

import os
import sys
import time
import argparse
import threading
import warnings
from datetime import datetime
import cv2
import torch
import numpy as np
from PIL import Image
from picamera2 import Picamera2, Preview
from libcamera import controls
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

RETINANET_ROOT = '/home/yiranspi5/camera_project/retinanet_pet_detector'
RETINANET_CONFIG = os.path.join(RETINANET_ROOT, 'config', 'resnet34.yaml')
SAVE_FOLDER = '/home/yiranspi5/camera_project/multi_model/Saved_Pics'


# ------------------------------------------------------------
# RETINANET PATH FIX
# ------------------------------------------------------------

if RETINANET_ROOT not in sys.path:
    sys.path.insert(0, RETINANET_ROOT)
# RetinaNet utilities load resources relative to the project root.
os.chdir(RETINANET_ROOT)
from references.utils import get_model, get_preds, load_yaml_config


# ------------------------------------------------------------
# GENERAL SETTINGS
# ------------------------------------------------------------

warnings.filterwarnings('ignore')
DEVICE = torch.device('cpu')
os.makedirs(SAVE_FOLDER, exist_ok=True)


# ------------------------------------------------------------
# CAMERA SETTINGS
# ------------------------------------------------------------

MAIN_WIDTH = 1280
MAIN_HEIGHT = 720
DETECT_WIDTH = 1280
DETECT_HEIGHT = 720
DETECTION_INTERVAL = 1.5


# ------------------------------------------------------------
# MODEL SETTINGS
# ------------------------------------------------------------

RETINANET_THRESHOLD = 0.3
FASTER_RCNN_THRESHOLD = 0.3
MOBILENET_ANIMAL_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
MAX_DETECTIONS = 100
REQUIRED_VOTES = 2

# ------------------------------------------------------------
# IMAGE QUALITY
# ------------------------------------------------------------

BLUR_THRESHOLD = 25.0

# ------------------------------------------------------------
# SAVE SETTINGS
# ------------------------------------------------------------

SAVE_INTERVAL = 2.0

# ------------------------------------------------------------
# TEMPERATURE SETTINGS
# ------------------------------------------------------------

MAX_TEMP = 70.0
TEMP_CHECK_INTERVAL = 4.0

# ------------------------------------------------------------
# GLOBAL EVENTS
# ------------------------------------------------------------

capture_stop_event = threading.Event()
thermal_stop_event = threading.Event()
total_saved = 0

# ------------------------------------------------------------
# COCO ANIMAL CLASSES
# ------------------------------------------------------------

COCO_ANIMALS = {'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'}

# ------------------------------------------------------------
# IMAGENET ANIMAL KEYWORDS
# ------------------------------------------------------------

ANIMAL_KEYWORDS = {'dog', 'terrier', 'retriever', 'shepherd', 'spaniel', 'poodle', 'hound',
    'collie', 'pinscher', 'schnauzer', 'mastiff', 'bulldog', 'chihuahua', 'pug', 'corgi',
    'malamute', 'husky', 'samoyed', 'dalmatian', 'papillon', 'pekinese', 'rottweiler', 'doberman',
    'boxer', 'beagle', 'whippet', 'greyhound', 'cat', 'tabby', 'siamese', 'persian', 'wolf', 'fox',
    'coyote', 'jackal', 'lion', 'tiger', 'leopard', 'jaguar', 'cheetah', 'cougar', 'lynx', 'bear',
    'horse', 'pony', 'zebra', 'cow', 'ox', 'bull', 'bison', 'buffalo', 'sheep', 'ram', 'goat',
    'pig', 'hog', 'boar', 'rabbit', 'hare', 'hamster', 'guinea pig', 'mouse', 'rat', 'squirrel',
    'chipmunk', 'marmot', 'beaver', 'otter', 'weasel', 'ferret', 'mink', 'mongoose', 'skunk',
    'monkey', 'ape', 'gorilla', 'chimpanzee', 'orangutan', 'baboon', 'macaque', 'gibbon',
    'elephant', 'rhinoceros', 'hippopotamus', 'camel', 'llama', 'alpaca', 'giraffe', 'deer',
    'antelope', 'gazelle', 'moose', 'elk', 'dolphin', 'whale', 'seal', 'sea lion', 'walrus', 'bird',
    'eagle', 'hawk', 'falcon', 'owl', 'parrot', 'macaw', 'cockatoo', 'penguin', 'duck', 'goose',
    'swan', 'chicken', 'hen', 'rooster', 'turkey', 'peacock', 'ostrich', 'flamingo', 'pelican',
    'pigeon', 'dove', 'sparrow', 'finch', 'snake', 'python', 'cobra', 'viper', 'boa', 'lizard',
    'gecko', 'iguana', 'chameleon', 'alligator', 'crocodile', 'turtle', 'tortoise', 'frog', 'toad',
    'salamander', 'newt', 'fish', 'shark', 'ray', 'eel', 'goldfish', 'salmon', 'trout', 'tuna',
    'crab', 'lobster', 'shrimp', 'spider', 'scorpion', 'butterfly', 'bee', 'beetle'}

# ------------------------------------------------------------
# CPU TEMPERATURE
# ------------------------------------------------------------

def get_cpu_temperature():
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as file:
            temperature = float(file.read().strip())
            return temperature / 1000.0
    except Exception:
        return None

# ------------------------------------------------------------
# TEMPERATURE MONITOR
# ------------------------------------------------------------

def temperature_monitor():
    while not thermal_stop_event.is_set():
        temperature = get_cpu_temperature()
        if temperature is not None:
            print(f'Pi Temperature: {temperature:.1f} C')
            if temperature >= MAX_TEMP:
                print()
                print('==========================================')
                print('TEMPERATURE SAFETY STOP')
                print('==========================================')
                print(f'Temperature reached {temperature:.1f} C')
                thermal_stop_event.set()
                capture_stop_event.set()
                break
        if capture_stop_event.is_set():
            break
        thermal_stop_event.wait(TEMP_CHECK_INTERVAL)

# ------------------------------------------------------------
# KEYBOARD CONTROL
# ------------------------------------------------------------

def keyboard_monitor():
    print()
    print('Type E + Enter to stop NEW picture acquisition.')
    while not capture_stop_event.is_set() and (not thermal_stop_event.is_set()):
        try:
            command = input().strip().lower()
        except EOFError:
            break
        if command == 'e':
            print()
            print('Camera stop requested.')
            print('No new pictures will be acquired.')
            print('Current picture analysis will continue.')
            capture_stop_event.set()
            break

# ------------------------------------------------------------
# IMAGENET ANIMAL CHECK
# ------------------------------------------------------------

def is_animal_imagenet_class(class_name):
    class_name = class_name.lower().strip()
    for keyword in ANIMAL_KEYWORDS:
        if keyword in class_name:
            return True
    return False

# ------------------------------------------------------------
# LOAD RETINANET
# ------------------------------------------------------------

def load_retinanet():
    print('Loading RetinaNet...')
    conf_dict = load_yaml_config(RETINANET_CONFIG)
    conf_dict['score_thres'] = RETINANET_THRESHOLD
    conf_dict['nms_thres'] = IOU_THRESHOLD
    conf_dict['max_detections'] = MAX_DETECTIONS
    conf = argparse.Namespace(**conf_dict)
    model = get_model(conf)
    print('RetinaNet loaded.')
    return model

# ------------------------------------------------------------
# LOAD R-CNN
# ------------------------------------------------------------

def load_faster_rcnn():
    print('Loading Faster R-CNN...')
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    model.to(DEVICE)
    model.eval()
    categories = weights.meta['categories']
    print('Faster R-CNN loaded.')
    return (model, categories)

# ------------------------------------------------------------
# LOAD MOBILENET V3
# ------------------------------------------------------------

def load_mobilenet():
    print('Loading MobileNetV3...')
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights)
    model.to(DEVICE)
    model.eval()
    preprocess = weights.transforms()
    categories = weights.meta['categories']
    print('MobileNetV3 loaded.')
    return (model, preprocess, categories)

# ------------------------------------------------------------
# RETINANET DETECTION
# ------------------------------------------------------------

def run_retinanet(model, image_rgb):
    boxes, classes, scores = get_preds(model, image_rgb)
    best_score = 0.0
    best_box = None
    for box, score in zip(boxes, scores):
        score = float(score)
        if score >= RETINANET_THRESHOLD and score > best_score:
            best_score = score
            best_box = box
    detected = best_box is not None
    return (detected, best_score, best_box)

# ------------------------------------------------------------
# FASTER R-CNN DETECTION
# ------------------------------------------------------------

def run_faster_rcnn(model, categories, image_rgb):
    image_rgb = np.ascontiguousarray(image_rgb)
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.to(DEVICE)
    with torch.inference_mode():
        output = model([image_tensor])[0]
    best_score = 0.0
    best_label = None
    best_box = None
    for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
        score_value = float(score.item())
        if score_value < FASTER_RCNN_THRESHOLD:
            break
        label_index = int(label.item())
        if label_index >= len(categories):
            continue
        label_name = categories[label_index].lower()
        if label_name in COCO_ANIMALS:
            if score_value > best_score:
                best_score = score_value
                best_label = label_name
                best_box = box.detach().cpu().numpy()
    detected = best_label is not None
    return (detected, best_score, best_label, best_box)

# ------------------------------------------------------------
# MOBILENET DETECTION
# ------------------------------------------------------------

def run_mobilenet(model, preprocess, categories, image_rgb):
    pil_image = Image.fromarray(image_rgb)
    input_tensor = preprocess(pil_image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
    animal_probability = 0.0
    for index, class_name in enumerate(categories):
        if is_animal_imagenet_class(class_name):
            animal_probability += float(probabilities[index].item())
    detected = animal_probability >= MOBILENET_ANIMAL_THRESHOLD
    return (detected, animal_probability)

# ------------------------------------------------------------
# THREE-MODEL DETECTION
# ------------------------------------------------------------

# Run all three models on the same frame, then vote.
def detect_animal(retinanet_model, faster_model, faster_categories, mobilenet_model,
    mobilenet_preprocess, mobilenet_categories, image_rgb):
    retina_yes, retina_score, retina_box = run_retinanet(retinanet_model, image_rgb)
    if thermal_stop_event.is_set():
        return None
    faster_yes, faster_score, faster_label, faster_box = run_faster_rcnn(faster_model,
        faster_categories, image_rgb)
    if thermal_stop_event.is_set():
        return None
    mobile_yes, mobile_probability = run_mobilenet(mobilenet_model, mobilenet_preprocess,
        mobilenet_categories, image_rgb)
    if thermal_stop_event.is_set():
        return None
    votes = int(retina_yes) + int(faster_yes) + int(mobile_yes)
    final_detected = votes >= REQUIRED_VOTES
    selected_box = None
    if faster_box is not None:
        selected_box = faster_box
    elif retina_box is not None:
        selected_box = retina_box
    return (final_detected, votes, selected_box, retina_yes, retina_score, faster_yes, faster_score,
        faster_label, mobile_yes, mobile_probability)

# ------------------------------------------------------------
# SCALE DETECTION BOX
# ------------------------------------------------------------

def scale_box_to_main(box):
    if box is None:
        return None
    x_scale = MAIN_WIDTH / DETECT_WIDTH
    y_scale = MAIN_HEIGHT / DETECT_HEIGHT
    x1, y1, x2, y2 = box
    return (int(x1 * x_scale), int(y1 * y_scale), int(x2 * x_scale), int(y2 * y_scale))

# ------------------------------------------------------------
# EXTRACT ROI
# ------------------------------------------------------------

def extract_roi(frame, box):
    if box is None:
        return frame
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return frame
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame
    return roi

# ------------------------------------------------------------
# SHARPNESS CHECK
# ------------------------------------------------------------

# Measure sharpness inside the detected animal region.
def calculate_sharpness(frame, box):
    roi = extract_roi(frame, box)
    if len(roi.shape) == 3 and roi.shape[2] == 4:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(sharpness)

# ------------------------------------------------------------
# SAVE CURRENT FRAME
# ------------------------------------------------------------

# Save the analyzed frame using the current camera channel order.
def save_frame(frame, votes, sharpness, recognition_time):
    global total_saved
    if thermal_stop_event.is_set():
        return (None, None)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f'animal_{timestamp}_votes_{votes}_sharp_{sharpness:.1f}.jpg'
    save_path = os.path.join(SAVE_FOLDER, filename)
    success = cv2.imwrite(save_path, frame)
    if success:
        save_finished_time = time.monotonic()
        recognition_to_save = save_finished_time - recognition_time
        total_saved += 1
        return (save_path, recognition_to_save)
    return (None, None)

# ------------------------------------------------------------
# CLOSE CAMERA
# ------------------------------------------------------------

def close_camera(camera):
    if camera is None:
        return
    try:
        camera.stop_preview()
    except Exception:
        pass
    try:
        camera.stop()
    except Exception:
        pass
    try:
        camera.close()
    except Exception:
        pass

# ------------------------------------------------------------
# MAIN PROGRAM
# ------------------------------------------------------------

def main():

    # Show the active settings and stop behavior.
    print()
    print('==========================================')
    print('LIVE MULTI-MODEL ANIMAL SYSTEM')
    print('==========================================')
    print('RetinaNet + Faster R-CNN + MobileNetV3')
    print('Voting Rule: 2 / 3')
    print('Processing: Sequential Single Frame')
    print('Burst Capture: OFF')
    print('RAM Frame Buffer: OFF')
    print('E Stop Behavior:')
    print('Stop new acquisition -> finish current analysis/save')
    print(f'Save Folder: {SAVE_FOLDER}')
    print(f'Sharpness Threshold: {BLUR_THRESHOLD:.1f}')
    print(f'Save Interval: {SAVE_INTERVAL:.1f} s')
    print(f'Temperature Display: Every {TEMP_CHECK_INTERVAL:.0f} seconds')
    print()

    # Check the initial CPU temperature before loading the models.
    temperature = get_cpu_temperature()
    if temperature is not None:
        print(f'Pi Temperature: {temperature:.1f} C')
        if temperature >= MAX_TEMP:
            print('Temperature too high.')
            return

    # Load all three models once and reuse them for every captured frame.
    retinanet_model = load_retinanet()
    faster_model, faster_categories = load_faster_rcnn()
    mobilenet_model, mobilenet_preprocess, mobilenet_categories = load_mobilenet()
    print()
    print('All 3 models loaded.')

    # Keep cleanup safe if camera initialization fails.
    camera = None
    try:

        # Create the camera and configure the preview and detection streams.
        camera = Picamera2()
        camera_config = camera.create_preview_configuration(main={'size': (MAIN_WIDTH, MAIN_HEIGHT),
            'format': 'XRGB8888'}, lores={'size': (DETECT_WIDTH, DETECT_HEIGHT),
            'format': 'RGB888'}, buffer_count=6)
        camera.configure(camera_config)

        # Start the desktop preview and enable continuous autofocus.
        camera.start_preview(Preview.QTGL)
        camera.start()
        camera.set_controls({'AfMode': controls.AfModeEnum.Continuous})
        print()
        print('QTGL Preview: ON')
        print('Continuous Autofocus: ON')
        print(f'Resolution: {MAIN_WIDTH}x{MAIN_HEIGHT}')
        print(f'Detection Interval: {DETECTION_INTERVAL:.1f} s')
        print('Burst Capture: OFF')
        print('RAM Frame Buffer: OFF')

        # Allow exposure and autofocus to settle before the first capture.
        time.sleep(3)

        # Monitor CPU temperature in a background thread.
        temp_thread = threading.Thread(target=temperature_monitor, daemon=True)
        temp_thread.start()

        # Listen for E + Enter without blocking image processing.
        keyboard_thread = threading.Thread(target=keyboard_monitor, daemon=True)
        keyboard_thread.start()

        # Track detection starts and successful saves separately.
        last_detection_time = 0.0
        last_save_time = 0.0
        print()
        print('Live Detection Started')
        print()

        # Process one frame at a time until a stop is requested.
        while not thermal_stop_event.is_set():

            if capture_stop_event.is_set():
                print()
                print('No more pictures will be acquired.')
                break

            # Wait until the minimum interval between detection starts has elapsed.
            current_time = time.monotonic()
            if current_time - last_detection_time < DETECTION_INTERVAL:
                time.sleep(0.03)
                continue

            if capture_stop_event.is_set():
                print()
                print('No more pictures will be acquired.')
                break
            last_detection_time = current_time

            # A normal stop finishes the current frame; a thermal stop cancels processing.
            detection_frame = camera.capture_array('lores')
            detection_frame = np.ascontiguousarray(detection_frame)

            # Run RetinaNet, Faster R-CNN and MobileNetV3 sequentially on this frame.
            # The detection function combines their results using a 2/3 vote.
            inference_start = time.monotonic()
            detection_result = detect_animal(retinanet_model, faster_model, faster_categories,
                mobilenet_model, mobilenet_preprocess, mobilenet_categories, detection_frame)

            # A thermal stop during inference cancels the remaining processing.
            if detection_result is None:
                print()
                print('Analysis cancelled by thermal safety.')
                break

            # Record inference duration and show the voting and model results.
            inference_time = time.monotonic() - inference_start
            detected, votes, detection_box, retina_yes, retina_score, faster_yes, faster_score, faster_label, mobile_yes, mobile_probability = detection_result

            # Discard frames that do not receive enough animal votes.
            if not detected:
                print('Animal Detected: NO')
                print(f'Votes: {votes}/3')
                print(f'Inference Time: {inference_time:.3f} s')
                print()
                if capture_stop_event.is_set():
                    print('Current picture analysis finished.')
                    print('Camera will now stop.')
                    break
                continue

            # This timer excludes model inference.
            recognition_time = time.monotonic()
            print('Animal Detected: YES')
            print(f'Votes: {votes}/3')
            print(f'Inference Time: {inference_time:.3f} s')

            # Display each model result 
            print(f"RetinaNet: {('YES' if retina_yes else 'NO')} ({retina_score:.2f})")
            if faster_label is not None:
                print(f"Faster R-CNN: {('YES' if faster_yes else 'NO')} ({faster_score:.2f}, {faster_label})")
            else:
                print(f"Faster R-CNN: {('YES' if faster_yes else 'NO')} ({faster_score:.2f})")
            print(f"MobileNetV3: {('YES' if mobile_yes else 'NO')} ({mobile_probability:.2f})")

            # Check for a thermal stop before checking image quality.
            if thermal_stop_event.is_set():
                print('Processing cancelled by thermal safety.')
                break

            # Prepare the ROI box for the sharpness check.
            frame_box = scale_box_to_main(detection_box)
            sharpness = calculate_sharpness(detection_frame, frame_box)
            print(f'Sharpness: {sharpness:.2f}')

            # Reject blurry pics; finish this frame before a normal stop.
            if sharpness < BLUR_THRESHOLD:
                print('Quality: FAIL')
                print('Picture Saved: NO')
                print('Reason: Image is too blurry.')
                print()
                if capture_stop_event.is_set():
                    print('Current picture analysis finished.')
                    print('Camera will now stop.')
                    break
                continue
            print('Quality: PASS')

            # Skip saving if the previous successful save was too recent
            # This frame is discarded rather than held until the interval expires.
            current_save_time = time.monotonic()
            if current_save_time - last_save_time < SAVE_INTERVAL:
                print('Picture Saved: NO')
                print('Reason: Save interval.')
                print()
                if capture_stop_event.is_set():
                    print('Current picture analysis finished.')
                    print('Camera will now stop.')
                    break
                continue

            # Save the same frame that passed voting and sharpness checks.
            # A normal stop allows this save to finish; a thermal stop can cancel it.
            save_path, recognition_to_save = save_frame(detection_frame, votes, sharpness, recognition_time)

            # Update the save timer only after a successful save.
            if save_path is not None:
                last_save_time = time.monotonic()
                print('Picture Saved: YES')
                print(f'Saved: {save_path}')
                print(f'Recognition -> Save: {recognition_to_save:.3f} s')
            else:
                print('Picture Saved: NO')
                if thermal_stop_event.is_set():
                    print('Reason: Thermal stop.')
                else:
                    print('Reason: Save error.')
            print()

            # Following a normal stop at this boundary before continuing the loop.
            if capture_stop_event.is_set():
                print('Current picture analysis and saving finished.')
                print('Camera will now stop.')
                break

    # Handle Ctrl+C and proceed to camera cleanup.
    except KeyboardInterrupt:
        capture_stop_event.set()

    # Report an unexpected error before releasing camera resources.
    except Exception as error:
        print()
        print('==========================================')
        print('PROGRAM ERROR')
        print('==========================================')
        print(error)

    # Release the camera and preview when the loop ends or an error occurs.
    finally:
        capture_stop_event.set()
        close_camera(camera)
        print()
        print('==========================================')
        print('CAMERA STOPPED')
        print('==========================================')
        print('No new pictures are being acquired.')
        print('Current picture processing completed.')
        print(f'Total Pictures Saved: {total_saved}')
        print(f'Pictures Folder: {SAVE_FOLDER}')

# ------------------------------------------------------------
# START
# ------------------------------------------------------------

if __name__ == '__main__':
    main()
