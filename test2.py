import os
import io
import numpy as np
from typing import List, Tuple
from PIL import Image, ImageDraw
from fastapi import FastAPI, Body, HTTPException, Security
from fastapi.responses import FileResponse, Response, StreamingResponse
from mmdet.apis import inference_detector, init_detector, async_inference_detector

# from utils import signJWT, decodeJWT
from app.auth.auth_handler import signJWT, decodeJWT
from app.model import UserSchema, UserLoginSchema, PointsSchema
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from utils import *
from fastapi.middleware.cors import CORSMiddleware
from math import sqrt, pow

from torchvision.transforms import ToTensor
from mmdet.apis.inference import init_detector
import time
from utils import *
from segment_anything import sam_model_registry, SamPredictor
import segmentation_models_pytorch as smp
import cv2
import torch
import torchvision.transforms as transforms
import math


app = FastAPI()
security = HTTPBearer()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Variables
# Global Variables
person_mmdet_config_file = "/home/asus/Downloads/PBL/M1_violate_warning_area_detection/source/yolov8_x_syncbn_fast_8xb16-500e_coco.py"
person_mmdet_checkpoint = '/home/asus/Downloads/PBL/M1_violate_warning_area_detection/source/yolov8_x_syncbn_fast_8xb16-500e_coco_20230218_023338-5674673c.pth'
person_mmdet_model = init_detector(person_mmdet_config_file, person_mmdet_checkpoint, device='cuda:0')
polygon_points = []

users = []

# merge


def merge_images(images: List[np.ndarray]) -> np.ndarray:
    complete_img = np.zeros_like(images[0])
    part_h = images[0].shape[0] // 2
    part_w = images[0].shape[1] // 3

    def get_image_index(
        i: int, j: int, images: List[np.ndarray], part_centers: List[Tuple[int, int]]
    ) -> int:
        distances = [
            sqrt(pow(i - center[0], 2) + pow(j - center[1], 2))
            for center in part_centers
        ]
        min_distance_index = np.argmin(distances)
        if np.array_equal(images[min_distance_index][i, j], np.array([0, 0, 0])):
            distances[min_distance_index] = 1000
            min_distance_index = np.argmin(distances)
        return min_distance_index

    part_centers = [
        (row * part_h + part_h // 2, col * part_w + part_w // 2)
        for row in range(2)
        for col in range(3)
    ]

    for i in range(images[0].shape[0]):
        for j in range(images[0].shape[1]):
            index = get_image_index(i, j, images, part_centers)
            complete_img[i, j] = images[index][i, j]

    return complete_img


def read_points(file_path):
    points = []
    with open(file_path, "r") as file:
        for line in file:
            x, y = map(float, line.strip().split())
            points.append([x, y])
    return np.array(points, dtype=np.float32)


@app.get("/merge-images", tags=["Merge Images"])
def merge_images_endpoint():
    images = []
    for i in range(6):
        image_path = "./Merge_camera/img/{}.jpg".format(i + 1)
        image = cv2.imread(image_path)
        if image is None:
            return {"error": f"Image {i + 1} could not be loaded."}

        src_points = read_points("./Merge_camera/Points/origin/{}.txt".format(i + 1))
        dst_points = read_points(
            "./Merge_camera/Points/destination/{}.txt".format(i + 1)
        )
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped_image = cv2.warpPerspective(image, M, (1280, 1024))
        output_path = "./Merge_camera/IPM/{}.jpg".format(i + 1)
        cv2.imwrite(output_path, warped_image)
        images.append(warped_image)

    merged_image = merge_images(images)

    _, img_encoded = cv2.imencode(".jpg", merged_image)
    img_bytes = img_encoded.tobytes()

    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")


def check_user(data: UserLoginSchema):
    for user in users:
        if user.email == data.email and user.password == data.password:
            return True
    return False


def get_frame_from_camera(camera_name: str):
    video_path = f"video/{camera_name}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Camera not found or video file not available"
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None, "Failed to read frame from video"
    return frame, None


def save_points_to_file(camera_name: str, points: List[Tuple[float, float]]):
    if not points:
        raise ValueError("No points provided")
    file_path = f"points/{camera_name}.txt"
    with open(file_path, "w") as file:
        for point in points:
            file.write(f"{point[0]},{point[1]}\n")


def read_points_from_txt(file_path: str) -> List[Tuple[int, int]]:
    points = []
    with open(file_path, "r") as file:
        for line in file:
            try:
                x_str, y_str = line.strip().split()
                point = (
                    int(float(x_str)),
                    int(float(y_str)),
                )  # Convert from string to float, then to int
                points.append(point)
            except ValueError:
                print(f"Error parsing line: {line}")  # Output the problematic line
    return points


@app.get("/Video/ListCamera", tags=["Video"])
def get_cameras():
    path = "video"
    cameras = [file.split("/")[-1].split(".")[0] for file in os.listdir(path)]
    return {"cameras": cameras}


# @app.get("/Video/{camera_name}", tags=["Video"])
# def get_camera_videos(camera_name: str):
#     file_path = 'video/' + camera_name + '.mp4'
#     def iterfile():  #
#         with open(file_path, mode="rb") as file_like:
#             yield from file_like
#     return StreamingResponse(iterfile(), media_type="video/mp4")


@app.get("/video/{camera_name}", tags=["Video"])
def stream_camera_video(camera_name: str):
    file_path = f"video/{camera_name}.mp4"
    try:
        return StreamingResponse(open_video_file(file_path), media_type="video/mp4")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Video file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def open_video_file(file_path: str):
    """Helper function to open a video file for streaming."""
    return open(file_path, mode="rb")


@app.get("/Danger_Area/Create_Point/{camera_name}", tags=["Danger Area"])
def get_camera_frame(camera_name: str):
    frame, error = get_frame_from_camera(camera_name + ".mp4")
    if error:
        return {"error": error}
    success, png_image = cv2.imencode(".png", frame)
    if not success:
        return {"error": "Failed to encode frame to PNG format"}
    image_bytes = png_image.tobytes()
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")


@app.post("/Danger_Area/Save_Points/{camera_name}", tags=["Danger Area"])
def save_points(camera_name: str, points: PointsSchema = Body(...)):
    try:
        save_points_to_file(camera_name + ".mp4".split(".")[-2], points.points)
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": "Internal Server Error"}
    return {"message": "Points saved successfully"}


@app.get("/Danger_Area/Get_Points/{camera_name}", tags=["Danger Area"])
def get_points(camera_name: str):
    file_path = f"./points/{camera_name}.txt"  # Path to the text file containing points
    try:
        points = read_points_from_txt(file_path)
    except FileNotFoundError:
        return {"error": f"No points found for camera {camera_name}.txt"}
    except Exception as e:
        return {"error": str(e)}

    return {"points": points}


@app.post("/user/signup", tags=["user"])
def create_user(user: UserSchema = Body(...)):
    users.append(user)
    return signJWT(user.email)


@app.post("/user/login", tags=["user"])
def user_login(user: UserLoginSchema = Body(...)):
    if check_user(user):
        return signJWT(user.email)
    return {"error": "Wrong login details!"}


def is_point_inside_polygon(
    x: float, y: float, points: List[Tuple[float, float]]
) -> bool:
    """Determine if the point (x, y) is inside the given polygon of `points`"""
    n = len(points)
    inside = False
    p1x, p1y = points[0]
    for i in range(1, n + 1):
        p2x, p2y = points[i % n]
        if min(p1y, p2y) < y <= max(p1y, p2y):
            if x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
        p1x, p1y = p2x, p2y
    return inside


# INFERENCE and CHECK
def filter_class(raw_worker_bboxes, raw_worker_labels, raw_worker_scores, class_name):
    worker_bboxes = []
    worker_labels = []
    worker_scores = []
    for i, worker_bbox in enumerate(raw_worker_bboxes):
        if raw_worker_labels[i] == class_name:
            worker_bboxes.append(worker_bbox)
            worker_labels.append("worker")
            worker_scores.append(raw_worker_scores[i])
    return worker_bboxes, worker_labels, worker_scores


# visualize the center bottom of the bounding box
def find_center_bottom(worker_bboxes):
    worker_centers = []
    for i, worker_bbox in enumerate(worker_bboxes):
        x1, y1, x2, y2 = worker_bbox
        x_center = int((x1 + x2) / 2)
        y_center = int(y2)
        worker_centers.append([x_center, y_center])
    return worker_centers


# draw a polygon from given danger_points
def draw_polygon(frame, danger_points, color="red"):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.polygon(danger_points, outline=color, width=5)
    return np.array(img)


def check_inside_polygon(worker_centers, danger_points):
    inside = []
    for i, worker_center in enumerate(worker_centers):
        x, y = worker_center
        inside.append(is_point_inside_polygon(x, y, danger_points))
    return inside


@app.get("/predict_danger/{camera_name}", tags=["Prediction"])
def predict_danger_areas(
    camera_name: str,
    # polygon_points_1: str,
    # polygon_points_2: str,
    # camera_name: str, polygon_points_1: str
):
    video_path = f"video/{camera_name}.mp4"
    # file_path_1 = f"points/{polygon_points_1}.txt"
    # file_path_2 = f"points/{polygon_points_2}.txt"
    file_path_1 = f"points/test1.txt"
    file_path_2 = f"points/test2.txt"
    # file_path_2 = f"points/test3.txt"

    try:
        points_1 = read_points_from_txt(file_path_1)
        points_2 = read_points_from_txt(file_path_2)
        if not points_1 or not points_2:
            # if not points_1:
            raise ValueError("One or both lists of points are empty")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to read polygon points: {str(e)}"
        )

    worker_threshold = 0.3

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(
            status_code=404, detail="Camera not found or video file not available"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    def generate_frames():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # inference
            # Show danger area
            danger_frame = draw_polygon(frame, points_1, color="red")
            danger_frame = draw_polygon(danger_frame, points_2, color="blue")

            # Show prediction
            worker_results = inference_detector(person_mmdet_model, frame)
            raw_worker_bboxes, raw_worker_labels, raw_worker_scores = (
                mmdet3x_convert_to_bboxes_mmdet(worker_results, worker_threshold)
            )
            worker_bboxes, worker_labels, worker_scores = filter_class(
                raw_worker_bboxes, raw_worker_labels, raw_worker_scores, "worker_1"
            )
            worker_centers = find_center_bottom(worker_bboxes)
            violated_worker_level_1 = 0
            violated_worker_level_2 = 0
            for i, worker_center in enumerate(worker_centers):
                x_center, y_center = worker_center
                # check worker in danger area level 1
                if is_point_inside_polygon(x_center, y_center, points_1):
                    violated_worker_level_1 += 1
                    danger_frame = cv2.circle(
                        danger_frame, (x_center, y_center), 5, (0, 0, 255), -1
                    )
                    danger_frame = cv2.putText(
                        danger_frame,
                        "Danger",
                        (x_center, y_center),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                # check worker in danger area level 2
                elif is_point_inside_polygon(x_center, y_center, points_2):
                    violated_worker_level_2 += 1
                else:
                    danger_frame = cv2.circle(
                        danger_frame, (x_center, y_center), 5, (0, 255, 0), -1
                    )
            # draw violated worker level 2 if number of person exceed 2
            if violated_worker_level_2 >= 2:
                for i, worker_center in enumerate(worker_centers):
                    x_center, y_center = worker_center
                    if is_point_inside_polygon(x_center, y_center, points_2):
                        danger_frame = cv2.circle(
                            danger_frame, (x_center, y_center), 5, (0, 0, 255), -1
                        )
                        danger_frame = cv2.putText(
                            danger_frame,
                            "Danger",
                            (x_center, y_center),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

            # Show FPS
            cv2.putText(
                danger_frame,
                "FPS: {}".format(fps),
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            frame_bytes = cv2.imencode(".jpg", danger_frame)[1].tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    return StreamingResponse(
        generate_frames(), media_type="multipart/x-mixed-replace;boundary=frame"
    )


def filter_class(
    raw_worker_bboxes, raw_worker_labels, raw_worker_scores, class_name, prefix="worker"
):
    worker_bboxes = []
    worker_labels = []
    worker_scores = []
    for i, worker_bbox in enumerate(raw_worker_bboxes):
        if raw_worker_labels[i] == class_name:
            worker_bboxes.append(worker_bbox)
            worker_labels.append(prefix)
            worker_scores.append(raw_worker_scores[i])
    return worker_bboxes, worker_labels, worker_scores


def extract_depth_from_boxes(boxes, depth_map):
    object_depths = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        object_depth_map = depth_map[y1:y2, x1:x2]
        median_depth = np.mean(object_depth_map)
        object_depths.append(median_depth)
    return object_depths


def extract_bounding_boxes_and_depth(detected_boxes, detected_labels, depths):
    objects = []
    for i, box in enumerate(detected_boxes):
        x1, y1, x2, y2 = map(int, box)
        # Store the bounding box and depth information
        obj = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "class": detected_labels[i],
            "depth": depths[i],
        }
        objects.append(obj)
    return objects


# compute distance between worker and vehicle
image_width = 1920  # Replace with the actual image width
image_height = 1080  # Replace with the actual image height


def normalize_coordinates(box, width, height):
    x1, y1, x2, y2, depth = box
    x1 /= width
    y1 /= height
    x2 /= width
    y2 /= height
    return [x1, y1, x2, y2, depth]


def find_center_coordinates_with_depth(box):
    x1, y1, x2, y2, depth = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y, depth


def euclidean_distance_3d(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


# compute distance between worker and vehicle
def computer_object_distance(object_infor):
    object_distances = {}
    num_objects = len(object_infor)
    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            obj_i = object_infor[i]
            obj_j = object_infor[j]
            # Check if one object is class 1 and the other is class 8
            if (obj_i["class"] == "worker" and obj_j["class"] == "vehicle") or (
                obj_i["class"] == "vehicle" and obj_j["class"] == "worker"
            ):

                object_i_bboxes_norm = normalize_coordinates(
                    [
                        obj_i["x1"],
                        obj_i["y1"],
                        obj_i["x2"],
                        obj_i["y2"],
                        obj_i["depth"],
                    ],
                    image_width,
                    image_height,
                )
                object_j_bboxes_norm = normalize_coordinates(
                    [
                        obj_j["x1"],
                        obj_j["y1"],
                        obj_j["x2"],
                        obj_j["y2"],
                        obj_j["depth"],
                    ],
                    image_width,
                    image_height,
                )

                object_i_center = find_center_coordinates_with_depth(
                    object_i_bboxes_norm
                )
                object_j_center = find_center_coordinates_with_depth(
                    object_j_bboxes_norm
                )

                distance = euclidean_distance_3d(object_i_center, object_j_center)
                # Compute the distance between objects
                # distance = np.sqrt((obj_i['x1'] - obj_j['x1']) ** 2 + (obj_i['y1'] - obj_j['y1']) ** 2 + (obj_i['depth'] - obj_j['depth']) ** 2)
                # Save the distance information
                object_distances[(i, j)] = {"distance": distance}
    return object_distances


# Load Models
print("Loading models...")
person_mmdet_config_file1 = "/home/asus/Downloads/PBL/M1_violate_warning_area_detection/source/yolov8_x_syncbn_fast_8xb16-500e_coco.py"
person_mmdet_checkpoint1 = '/home/asus/Downloads/PBL/M1_violate_warning_area_detection/source/yolov8_x_syncbn_fast_8xb16-500e_coco_20230218_023338-5674673c.pth'

person_mmdet_model1 = init_detector(person_mmdet_config_file1, person_mmdet_checkpoint1, device='cuda:0')

vehicle_mmdet_config_file = "/home/asus/Downloads/PBL/M1_violate_warning_area_detection/source/yolov8_x_syncbn_fast_8xb16-500e_coco.py"
vehicle_mmdet_checkpoint = '/home/asus/Downloads/PBL/M1_violate_warning_area_detection/source/yolov8_x_syncbn_fast_8xb16-500e_coco_20230218_023338-5674673c.pth'

vehicle_mmdet_model = init_detector(vehicle_mmdet_config_file, vehicle_mmdet_checkpoint, device='cuda:0')

depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
depth_model.eval()

path_video = "video/Demo.mp4"
# Visualize frame
worker_threshold = 0.5
vehicle_threshold = 0.5
safe_distance = 0.25


@app.get("/calculate_distance_danger_area/{camera_name}", tags=["Distance"])
def calculate_distance_danger_area(
    camera_name: str,
):
    cap = cv2.VideoCapture(path_video)
    # get first frame
    ret, frame = cap.read()
    # get depth map
    print("Inference depth map")
    depth_map = extract_depth_map(depth_model, frame)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Visualize results")

    def generate_frames():
        while True:
            ret, frame = cap.read()
            if ret:
                worker_results = inference_detector(person_mmdet_model1, frame)
                raw_worker_bboxes, raw_worker_labels, raw_worker_scores = (
                    mmdet3x_convert_to_bboxes_mmdet(worker_results, worker_threshold)
                )
                worker_bboxes, worker_labels, worker_scores = filter_class(
                    raw_worker_bboxes,
                    raw_worker_labels,
                    raw_worker_scores,
                    "class_1",
                    prefix="worker",
                )
                vehicle_results = inference_detector(vehicle_mmdet_model, frame)
                raw_vehicle_bboxes, raw_vehicle_labels, raw_vehicle_scores = (
                    mmdet3x_convert_to_bboxes_mmdet(vehicle_results, vehicle_threshold)
                )
                vehicle_bboxes, vehicle_labels, vehicle_scores = filter_class(
                    raw_vehicle_bboxes,
                    raw_vehicle_labels,
                    raw_vehicle_scores,
                    "class_8",
                    prefix="vehicle",
                )
                # combined and visualize
                combined_bboxes = worker_bboxes + vehicle_bboxes
                combined_labels = worker_labels + vehicle_labels
                # Depth map
                # evey one second compute depth map

                # cv2.putText(frame, "Compute depth map...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                time.sleep(1)
                print("compute depth map")
                depth_map = extract_depth_map(depth_model, frame)

                object_depths = extract_depth_from_boxes(combined_bboxes, depth_map)
                object_infor = extract_bounding_boxes_and_depth(
                    combined_bboxes, combined_labels, object_depths
                )
                object_distances = computer_object_distance(object_infor)
                center_points = [
                    ((box["x1"] + box["x2"]) // 2, (box["y1"] + box["y2"]) // 2)
                    for box in object_infor
                ]
                # visualize
                for (i, j), distance_info in object_distances.items():
                    pt1, pt2 = center_points[i], center_points[j]
                    distance = distance_info["distance"]

                    # Draw a line between the center points
                    if distance < safe_distance:
                        line_color = (255, 0, 0)
                        cv2.putText(
                            frame,
                            "Danger! Worker too near with construction vehicle = {}".format(
                                np.around(distance, 2)
                            ),
                            (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                        )
                    else:
                        line_color = (0, 255, 255)

                    cv2.line(frame, pt1, pt2, line_color, 2)
                    # Calculate the midpoint of the line and display the distance
                    midpoint = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                    distance_text = f"{distance:.2f}"
                    cv2.putText(
                        frame,
                        distance_text,
                        midpoint,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2,
                    )

                cv2.putText(
                    frame,
                    "FPS: {}".format(fps),
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
            frame_bytes = cv2.imencode(".jpg", frame)[1].tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    return StreamingResponse(
        generate_frames(), media_type="multipart/x-mixed-replace;boundary=frame"
    )