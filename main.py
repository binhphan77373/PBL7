import os
import cv2
import io
import numpy as np
from typing import List, Tuple
from PIL import Image, ImageDraw
from fastapi import FastAPI, Body, HTTPException, Security
from fastapi.responses import FileResponse, Response, StreamingResponse
from mmdet.apis import inference_detector, init_detector
#from utils import signJWT, decodeJWT
from app.auth.auth_handler import signJWT, decodeJWT
from app.model import UserSchema, UserLoginSchema, PointsSchema
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from utils import *

app = FastAPI()
security = HTTPBearer()

# Global Variables
person_mmdet_config_file = "/home/asus/Downloads/PBL/M1_violate_warning_area_detection/source/yolov8_x_syncbn_fast_8xb16-500e_coco.py"
person_mmdet_checkpoint = '/home/asus/Downloads/PBL/M1_violate_warning_area_detection/source/yolov8_x_syncbn_fast_8xb16-500e_coco_20230218_023338-5674673c.pth'
person_mmdet_model = init_detector(person_mmdet_config_file, person_mmdet_checkpoint, device='cuda:0')
#person_mmdet_model = init_detector(person_mmdet_config_file, person_mmdet_checkpoint, device='cpu')
polygon_points = []

users = []

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
                point = (int(float(x_str)), int(float(y_str)))  # Convert from string to float, then to int
                points.append(point)
            except ValueError:
                print(f"Error parsing line: {line}")  # Output the problematic line
    return points


@app.get("/Video/ListCamera", tags=["Video"])
def get_cameras():
    path = 'video'
    cameras = [file.split('/')[-1].split('.')[0] for file in os.listdir(path)]
    return {'cameras': cameras}

# @app.get("/Video/{camera_name}", tags=["Video"])
# def get_camera_videos(camera_name: str):
#     file_path = 'video/' + camera_name + '.mp4'
#     def iterfile():  # 
#         with open(file_path, mode="rb") as file_like:  
#             yield from file_like  
#     return StreamingResponse(iterfile(), media_type="video/mp4")

@app.get("/video/{camera_name}", tags=["Video"])
def stream_camera_video(camera_name: str):
    file_path = f'video/{camera_name}.mp4'
    try:
        return StreamingResponse(open_video_file(file_path), media_type="video/mp4")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Video file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def open_video_file(file_path: str):
    """ Helper function to open a video file for streaming. """
    return open(file_path, mode="rb")


@app.get("/Danger_Area/Create_Point/{camera_name}", tags=["Danger Area"])
def get_camera_frame(camera_name: str):
    frame, error = get_frame_from_camera(camera_name + '.mp4')
    if error:
        return {"error": error}
    success, png_image = cv2.imencode('.png', frame)
    if not success:
        return {"error": "Failed to encode frame to PNG format"}
    image_bytes = png_image.tobytes()
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")

@app.post("/Danger_Area/Save_Points/{camera_name}", tags=["Danger Area"])
def save_points(camera_name: str, points: PointsSchema = Body(...)):
    try:
        save_points_to_file(camera_name + '.mp4'.split('.')[-2], points.points)
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
    return {
        "error": "Wrong login details!"
    }


def is_point_inside_polygon(x: float, y: float, points: List[Tuple[float, float]]) -> bool:
    """ Determine if the point (x, y) is inside the given polygon of `points` """
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
            worker_labels.append('worker')
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
def draw_polygon(frame, danger_points, color = 'red'):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.polygon(danger_points, outline =color, width = 5)
    return np.array(img)

def check_inside_polygon(worker_centers, danger_points):
    inside = []
    for i, worker_center in enumerate(worker_centers):
        x, y = worker_center
        inside.append(is_point_inside_polygon(x, y, danger_points))
    return inside

@app.get("/predict_danger/{camera_name}", tags=["Prediction"])
def predict_danger_areas(camera_name: str, polygon_points_1: str, polygon_points_2: str):
    video_path = f"video/{camera_name}.mp4"
    file_path_1 = f"points/{polygon_points_1}.txt"
    file_path_2 = f"points/{polygon_points_2}.txt"

    try:
        points_1 = read_points_from_txt(file_path_1)
        points_2 = read_points_from_txt(file_path_2)
        if not points_1 or not points_2:
            raise ValueError("One or both lists of points are empty")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read polygon points: {str(e)}")

    worker_threshold = 0.3

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=404, detail="Camera not found or video file not available")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    def generate_frames():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # inference
            # Show danger area
            danger_frame = draw_polygon(frame, points_1, color='red')
            danger_frame = draw_polygon(danger_frame, points_2, color='blue')

            # Show prediction
            worker_results = inference_detector(person_mmdet_model, frame)
            raw_worker_bboxes, raw_worker_labels, raw_worker_scores = mmdet3x_convert_to_bboxes_mmdet(worker_results, worker_threshold)
            worker_bboxes, worker_labels, worker_scores = filter_class(raw_worker_bboxes, raw_worker_labels,
                                                                    raw_worker_scores, 'worker_1')
            worker_centers = find_center_bottom(worker_bboxes)
            violated_worker_level_1 = 0
            violated_worker_level_2 = 0
            for i, worker_center in enumerate(worker_centers):
                x_center, y_center = worker_center
                # check worker in danger area level 1
                if is_point_inside_polygon(x_center, y_center, points_1):
                    violated_worker_level_1 +=1
                    danger_frame = cv2.circle(danger_frame, (x_center, y_center), 5, (0, 0, 255), -1)
                    danger_frame = cv2.putText(danger_frame, 'Danger', (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                # check worker in danger area level 2
                elif is_point_inside_polygon(x_center, y_center, points_2):
                    violated_worker_level_2 +=1
                else:
                    danger_frame = cv2.circle(danger_frame, (x_center, y_center), 5, (0, 255, 0), -1)
            # draw violated worker level 2 if number of person exceed 2
            if violated_worker_level_2 >= 2:
                for i, worker_center in enumerate(worker_centers):
                    x_center, y_center = worker_center
                    if is_point_inside_polygon(x_center, y_center, points_2):
                        danger_frame = cv2.circle(danger_frame, (x_center, y_center), 5, (0, 0, 255), -1)
                        danger_frame = cv2.putText(danger_frame, 'Danger', (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Show FPS
            cv2.putText(danger_frame, 'FPS: {}'.format(fps), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            frame_bytes = cv2.imencode('.jpg', danger_frame)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace;boundary=frame")
