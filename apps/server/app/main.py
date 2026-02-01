from fastapi import FastAPI, UploadFile, File, Form, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
from . import vision

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze-video")
async def analyze_video_endpoint(file: UploadFile = File(...)):
    return {
        "landmark": {
            "name": "Sample Landmark",
            "facts": [
                "This is a sample landmark for testing",
                "Built in the year 2024",
                "Located in a test location"
            ]
        },
        "face_region": {
            "x": 100,
            "y": 100,
            "width": 200,
            "height": 200
        }
    }


@app.post("/analyze-ar")
async def analyze_ar_endpoint(
    frame: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    yaw: float = Form(...),
    pitch: float = Form(...),
    roll: float = Form(...)
):
    return {
        "landmark": {
            "name": "AR Sample Landmark",
            "facts": [
                "Detected via AR mode",
                "GPS coordinates processed",
                "Device orientation tracked"
            ]
        },
        "face_region": {
            "x": 150,
            "y": 150,
            "width": 100,
            "height": 100
        }
    }


@app.post("/analyze-ar-frame")
async def analyze_ar_frame_endpoint(
    frame: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    accuracy: float = Form(...),
    alpha: float = Form(...),
    beta: float = Form(...),
    gamma: float = Form(...),
    use_api: bool = Form(False)
):
    """Analyze AR frame and return visible landmarks based on GPS and orientation"""
    # Use vision module for landmark detection
    result = vision.analyze_ar_frame(
        None, latitude, longitude, accuracy, alpha, beta, gamma, use_api=use_api)
    return result


@app.websocket("/ar-stream")
async def ar_websocket_endpoint(websocket: WebSocket):
    print("WebSocket connection attempt received")
    try:
        await websocket.accept()
        print("WebSocket connection established!")
    except Exception as e:
        print(f"WebSocket accept failed: {e}")
        return

    try:
        while True:
            try:
                data = await websocket.receive_json()
            except Exception as e:
                print(f"Failed to receive JSON: {e}")
                break

            latitude = data.get('latitude', 0)
            longitude = data.get('longitude', 0)
            accuracy = data.get('accuracy', 0)
            alpha = data.get('alpha', 0)
            beta = data.get('beta', 0)
            gamma = data.get('gamma', 0)

            print(
                f"WebSocket received: lat={latitude}, lon={longitude}, alpha={alpha}, beta={beta}, gamma={gamma}")

            # Find visible landmarks based on GPS and orientation
            use_api = data.get('use_api', False)  # Optional API integration
            visible_landmarks = vision.find_visible_landmarks(
                latitude, longitude, alpha, use_api=use_api)

            if visible_landmarks:
                print(visible_landmarks)
                # Return the closest visible landmark
                landmark = visible_landmarks[0]
                result = {
                    "landmark": {
                        "name": landmark['name'],
                        "facts": landmark['facts']
                    },
                    "face_region": {
                        "x": 200,
                        "y": 150,
                        "width": 120,
                        "height": 120
                    },
                    "location": {
                        "latitude": latitude,
                        "longitude": longitude,
                        "accuracy": accuracy
                    },
                    "orientation": {
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma
                    },
                    "landmark_info": {
                        "distance": landmark['distance'],
                        "bearing": landmark['bearing'],
                        "angle_diff": landmark['angle_diff']
                    }
                }
            else:
                # No landmarks visible
                result = {
                    "landmark": {
                        "name": "No landmarks detected",
                        "facts": [
                            "Try moving to a different location",
                            "Point your camera toward nearby attractions",
                            f"Current position: {latitude:.4f}, {longitude:.4f}"
                        ]
                    },
                    "face_region": {
                        "x": 200,
                        "y": 150,
                        "width": 120,
                        "height": 120
                    },
                    "location": {
                        "latitude": latitude,
                        "longitude": longitude,
                        "accuracy": accuracy
                    },
                    "orientation": {
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma
                    }
                }

            await websocket.send_json(result)

    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await websocket.close()
        print("WebSocket connection closed")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
