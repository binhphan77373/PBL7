import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import timm
import segmentation_models_pytorch as smp

from torchvision.transforms import ToTensor
from PIL import Image, ImageDraw
# from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
#                          vis_pose_result)
from mmdet.apis import init_detector

# Depth Estimation Modules
def depth_preprocess_image(image):
    # Resize the input image to dimensions divisible by 32
    h, w, _ = image.shape
    new_h = h - (h % 32)
    new_w = w - (w % 32)
    resized_image = cv2.resize(image, (new_w, new_h))

    input_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return input_transform(resized_image).unsqueeze(0)
def postprocess_depth(depth_tensor, original_size):
    depth_map = depth_tensor.squeeze().cpu().numpy()
    depth_map = cv2.resize(depth_map, original_size)
    depth_map = depth_map / depth_map.max()
    return depth_map
def extract_depth_map(model, image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = depth_preprocess_image(img_rgb)
    with torch.no_grad():
        depth_tensor = model(input_tensor)
        depth_map = postprocess_depth(depth_tensor, (img_rgb.shape[1], img_rgb.shape[0]))
    return depth_map
# Segmentation Modules
def segment_preprocess_image(image):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((512, 512)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Load an example image
    img_rgb = np.array(image)

    # Convert NumPy array to PIL Image object
    image = Image.fromarray(img_rgb)
    image = transform(image)
    return image
def extract_segmentation_map(model, image):
    image = segment_preprocess_image(image)
    output = model(image.unsqueeze(0))
    segmentation_mask = output['out'][0].argmax(0).detach().cpu().numpy()
    return segmentation_mask
# Object Detection Modules
coco_classes = {0: u'__background__',
 1: u'person',
 2: u'bicycle',
 3: u'car',
 4: u'motorcycle',
 5: u'airplane',
 6: u'bus',
 7: u'train',
 8: u'truck',
 9: u'boat',
 10: u'traffic light',
 11: u'fire hydrant',
 12: u'stop sign',
 13: u'parking meter',
 14: u'bench',
 15: u'bird',
 16: u'cat',
 17: u'dog',
 18: u'horse',
 19: u'sheep',
 20: u'cow',
 21: u'elephant',
 22: u'bear',
 23: u'zebra',
 24: u'giraffe',
 25: u'backpack',
 26: u'umbrella',
 27: u'handbag',
 28: u'tie',
 29: u'suitcase',
 30: u'frisbee',
 31: u'skis',
 32: u'snowboard',
 33: u'sports ball',
 34: u'kite',
 35: u'baseball bat',
 36: u'baseball glove',
 37: u'skateboard',
 38: u'surfboard',
 39: u'tennis racket',
 40: u'bottle',
 41: u'wine glass',
 42: u'cup',
 43: u'fork',
 44: u'knife',
 45: u'spoon',
 46: u'bowl',
 47: u'banana',
 48: u'apple',
 49: u'sandwich',
 50: u'orange',
 51: u'broccoli',
 52: u'carrot',
 53: u'hot dog',
 54: u'pizza',
 55: u'donut',
 56: u'cake',
 57: u'chair',
 58: u'couch',
 59: u'potted plant',
 60: u'bed',
 61: u'dining table',
 62: u'toilet',
 63: u'tv',
 64: u'laptop',
 65: u'mouse',
 66: u'remote',
 67: u'keyboard',
 68: u'cell phone',
 69: u'microwave',
 70: u'oven',
 71: u'toaster',
 72: u'sink',
 73: u'refrigerator',
 74: u'book',
 75: u'clock',
 76: u'vase',
 77: u'scissors',
 78: u'teddy bear',
 79: u'hair drier',
 80: u'toothbrush'}

def detection_preprocess_image(image):
    img_tensor = ToTensor()(image).unsqueeze(0)
    img_tensor = img_tensor.cuda()
    return img_tensor

def extract_detection_boxes(model, image, confidence_threshold):
    img_tensor = detection_preprocess_image(image)
    with torch.no_grad():
        output = model(img_tensor)[0]

    # Extract bounding boxes, labels, and confidences for detected objects
    boxes = output['boxes'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    labels = output['labels'].cpu().numpy()

    # Filter out low-confidence detections
    valid_indices = scores > confidence_threshold
    boxes = boxes[valid_indices]
    labels = labels[valid_indices]
    save_labels = []
    save_boxes = []
    for box, label in zip(boxes, labels):
        if label == 1 or label == 6 or label == 8:
            save_labels.append(label)
            save_boxes.append(box)
    vehicle_bboxes = []
    vehicle_labels = []
    vehicle_uncertainties = []
    for i, check_label in enumerate(save_labels):
        if check_label == 8:
            vehicle_bboxes.append(save_boxes[i])
            vehicle_labels.append(save_labels[i])

    return vehicle_bboxes, vehicle_labels, scores[valid_indices]
# MMDetection modules
def convert_negative_to_zero(list):
    for i in range(len(list)):
        if list[i] < 0:
            list[i] = 0
    return list
def convert_to_bboxes_mmdet(results, threshold, prefix = 'worker'):
    boxes_list = []
    scores_list = []
    labels_list = []
    for predicted_class in range(len(results)):
        for preds in results[predicted_class]:
            if preds[-1] >= threshold:
                # print(preds)
                box = preds[:4]
                box = convert_negative_to_zero(box)
                boxes_list.append(box)
                score = preds[-1].astype(float)
                scores_list.append(score)
                labels_list.append('{}_{}'.format(prefix,predicted_class+1))
    return boxes_list, labels_list, scores_list

def mmdet3x_convert_to_bboxes_mmdet(results, threshold, prefix = 'worker'):
    boxes_list = []
    scores_list = []
    labels_list = []
    confidence_score = results.pred_instances.scores.tolist()
    for i, conf in enumerate(confidence_score):
        if conf >= threshold:
            # print(conf)
            extracted_box = results.pred_instances.bboxes[i].cpu().tolist()
            extracted_label = results.pred_instances.labels[i].cpu().tolist()
            boxes_list.append([int(extracted_box[0]),
                               int(extracted_box[1]),
                                int(extracted_box[2]),
                                int(extracted_box[3])])
            scores_list.append(conf)
            labels_list.append('{}_{}'.format(prefix,extracted_label+1))
    return boxes_list, labels_list, scores_list

def mmdet3x_visualize_mmdet(image, boxes, labels):
    image = image.copy()
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, str(label), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    # plt.imshow(image)
    # plt.show()
    return image

def visualize_mmdet(image, boxes, labels):
    image = image.copy()
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, str(label), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    plt.imshow(image)
    plt.show()



# Multi-Object Tracking Modules
def undistort_frame(frame, mtx, dist):
    undistorted_frame = cv2.undistort(frame, mtx, dist)
    return undistorted_frame
def extract_keypoints(keypoints):
    left_shoulder = keypoints[5, :2]
    right_shoulder = keypoints[6, :2]
    left_hip = keypoints[11, :2]
    right_hip = keypoints[12, :2]
    return left_shoulder, right_shoulder, left_hip, right_hip
def estimate_ankle_positions_mirror(left_shoulder_coord, right_shoulder_coord, left_hip_coord, right_hip_coord, mirroring_fator = 3):
    # Calculate the midpoint of the shoulders and hips
    shoulder_midpoint = (left_shoulder_coord + right_shoulder_coord) / 2
    hip_midpoint = (left_hip_coord + right_hip_coord) / 2

    # Calculate the vector from the shoulder midpoint to the hip midpoint
    shoulder_to_hip_vector = hip_midpoint - shoulder_midpoint

    # Double the length of the vector to approximate the upper body's mirror image
    mirrored_vector = shoulder_to_hip_vector * mirroring_fator

    # Add the resulting vector to the hip midpoint to obtain the estimated ankle midpoint
    estimated_ankle_midpoint = hip_midpoint + mirrored_vector

    # Calculate the vector from the left hip to the left shoulder
    left_hip_to_shoulder_vector = left_shoulder_coord - left_hip_coord

    # Add the left hip-to-shoulder vector to the estimated ankle midpoint to obtain the left ankle position
    left_ankle_coord = estimated_ankle_midpoint + left_hip_to_shoulder_vector

    # Calculate the vector from the right hip to the right shoulder
    right_hip_to_shoulder_vector = right_shoulder_coord - right_hip_coord

    # Add the right hip-to-shoulder vector to the estimated ankle midpoint to obtain the right ankle position
    right_ankle_coord = estimated_ankle_midpoint + right_hip_to_shoulder_vector

    return left_ankle_coord, right_ankle_coord
def calculate_center_point(left_ankle_coord, right_ankle_coord):
    center_point = [(left_ankle_coord[0][0] + right_ankle_coord[0][0])/2, (left_ankle_coord[0][1] + right_ankle_coord[0][1])/2]
    return center_point
def estimate_person_location(od_model, pose_model, image, od_confidence = 0.95, keypoint_confidence = 0.5):
    od_boxes, od_labels, od_uncertainties = extract_detection_boxes(od_model, image, od_confidence)
    center_points = []
    if od_boxes:
        print("Person detected")
        org_image = image.copy()
        for od_box in od_boxes:
            x1, y1, x2, y2 = [int(coord) for coord in od_box]
            cropped_frame = org_image[y1:y2, x1:x2]
            cropped_results = inference_top_down_pose_model(pose_model, cropped_frame)
            # extract ankles
            keypoints = [r['keypoints'] for r in cropped_results[0]]
            # Visualize in org original image
            right_ankle_keypoints = [kp[15, :2] + np.array([x1, y1]) for kp in keypoints if kp[15, 2] > keypoint_confidence]
            left_ankle_keypoints = [kp[16, :2] + np.array([x1, y1]) for kp in keypoints if kp[16, 2] > keypoint_confidence]
            if len(right_ankle_keypoints) != 0:
                # print("Detected ankles")
                try:
                    center_point = calculate_center_point(left_ankle_keypoints, right_ankle_keypoints)
                    center_point = calculate_center_point(left_ankle_keypoints, right_ankle_keypoints)
                    # print("Center point: ", center_point)
                    cv2.circle(org_image, (int(center_point[0]), int(center_point[1])), 5, (0, 0, 255), -1)
                except:
                    center_point = []
            else:
                # print("Estimated ankles")
                left_shoulder, right_shoulder, left_hip, right_hip = extract_keypoints(keypoints[0])
                left_ankle, right_ankle_ = estimate_ankle_positions_mirror(left_shoulder, right_shoulder, left_hip, right_hip)
                # print(head_top, left_shoulder, right_shoulder)
                left_ankle_keypoints = [np.array([left_ankle[0], left_ankle[1]]) + np.array([x1, y1])]
                right_ankle_keypoints = [np.array([right_ankle_[0], right_ankle_[1]]) +  np.array([x1, y1])]
                center_point = calculate_center_point(left_ankle_keypoints, right_ankle_keypoints)
                cv2.circle(org_image, (int(center_point[0]), int(center_point[1])), 5, (0, 0, 255), -1)
            center_points.append(center_point)
    else:
        print("No person detected")
        center_points = []
        org_image = image.copy()
    return org_image, center_points, od_boxes

## Depth map and check distance
def extract_depth_from_boxes(boxes, depth_map):
    depths = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        object_depth_map = depth_map[y1:y2, x1:x2]
        median_depth = np.mean(object_depth_map)
        depths.append(median_depth)
    return depths

def extract_bounding_boxes_and_depth(detected_boxes, detected_labels, detected_uncertainties, depths):
    objects = []
    for i, box in enumerate(detected_boxes):
        x1, y1, x2, y2 = map(int, box)
        # Store the bounding box and depth information
        obj = {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'class': detected_labels[i],
            'uncertainty': detected_uncertainties[i],
            'depth': depths[i]
        }
        objects.append(obj)
    return objects
def compute_object_distances_with_uncertainties(objects):
    object_distances = {}
    num_objects = len(objects)
    for i in range(num_objects):
        for j in range(i+1, num_objects):
            obj_i = objects[i]
            obj_j = objects[j]
            # Check if one object is class 1 and the other is class 8
            if (obj_i['class'] == 'worker_1' and obj_j['class'] == '8') \
                    or (obj_i['class'] == '8' and obj_j['class'] == 'worker_1'):
                # Compute the distance between objects
                distance = np.sqrt((obj_i['x1'] - obj_j['x1']) ** 2 + (obj_i['y1'] - obj_j['y1']) ** 2 + (obj_i['depth'] - obj_j['depth']) ** 2)
                # Save the distance information
                object_distances[(i, j)] = {'distance': distance}
    return object_distances



