
from fastapi import FastAPI, Body, Depends, File, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
import os
# from decouple import config
from app.model import UserSchema, UserLoginSchema, PointsSchema
from app.auth.auth_handler import signJWT
import cv2
import io
from typing import List, Tuple
users = []

app = FastAPI()


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
            
def read_points_from_txt(file_path: str) -> List[Tuple[float, float]]:
    points = []
    with open(file_path, "r") as file:
        for line in file:
            x_str, y_str = line.strip().split(',')
            point = (float(x_str), float(y_str))
            points.append(point)
    return points

@app.get("/Video/ListCamera", tags=["Video"])
def get_cameras():
    path = 'video'
    cameras = [file.split('/')[-1].split('.')[0] for file in os.listdir(path)]
    return {'cameras': cameras}

@app.get("/Video/{camera_name}", tags=["Video"])
def get_camera_videos(camera_name: str):
    file_path = 'video/' + camera_name + '.mp4'
    def iterfile():  # 
        with open(file_path, mode="rb") as file_like:  
            yield from file_like  
    return StreamingResponse(iterfile(), media_type="video/mp4")

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
