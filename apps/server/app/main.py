from fastapi import FastAPI, UploadFile, File, Form, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json

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
    gamma: float = Form(...)
):
    """Dummy endpoint for AR frame analysis - returns sample landmark data"""
    return {
        "landmark": {
            "name": "Live AR Landmark",
            "facts": [
                "Real-time landmark detection",
                f"GPS: {latitude:.4f}, {longitude:.4f}",
                f"Orientation: α={alpha:.1f}°, β={beta:.1f}°, γ={gamma:.1f}°"
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


@app.websocket("/ar-stream")
async def ar_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()

            latitude = data.get('latitude', 0)
            longitude = data.get('longitude', 0)
            accuracy = data.get('accuracy', 0)
            alpha = data.get('alpha', 0)
            beta = data.get('beta', 0)
            gamma = data.get('gamma', 0)

            result = {
                "landmark": {
                    "name": "Live AR Landmark (WebSocket)",
                    "facts": [
                        "Real-time WebSocket streaming enabled",
                        f"GPS: {latitude:.4f}, {longitude:.4f}",
                        f"Orientation: α={alpha:.1f}°, β={beta:.1f}°, γ={gamma:.1f}°",
                        "Frame received via WebSocket"
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
    finally:
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
