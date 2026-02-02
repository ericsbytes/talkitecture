from .tts.tts import TTSService, read_generated_text
from .landmarks import LANDMARKS
from . import vision
from fastapi import FastAPI, HTTPException, Form, WebSocket
from fastapi.staticfiles import StaticFiles
from typing import Optional
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import re
import json
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables FIRST before any other imports
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Force reload the API key after dotenv loads
os.environ['ELEVENLABS_API_KEY'] = os.getenv('ELEVENLABS_API_KEY', '')

# NOW import modules that depend on env variables

# Re-initialize TTS service with the loaded API key
api_key = os.getenv("ELEVENLABS_API_KEY")
if api_key:
    api_key = api_key.strip()  # Remove any whitespace
    print(f"✓ API Key loaded: {api_key[:10]}... (length: {len(api_key)})")
else:
    print("WARNING: ELEVENLABS_API_KEY not found in environment!")

tts_service = TTSService(api_key=api_key if api_key else "")


def get_landmark_info(landmark_name: str) -> Optional[dict]:
    """
    Get landmark info by name from the already-loaded LANDMARKS
    Returns dict with 'script', 'persona', 'latitude', 'longitude', etc., or None if not found
    """
    for landmark in LANDMARKS:
        if landmark.get('name') == landmark_name:
            return landmark
    return None


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Audio files directory
audio_dir = Path(__file__).parent.parent / "audio"


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve audio files with proper CORS headers and content type"""
    # Security: only allow alphanumeric, underscore, hyphen in filename
    if not all(c.isalnum() or c in '-_.' for c in filename):
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = audio_dir / filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found")

    try:
        with open(file_path, "rb") as f:
            audio_data = f.read()

        return Response(
            content=audio_data,
            media_type="audio/mpeg",
            headers={
                "Content-Type": "audio/mpeg",
                "Content-Length": str(len(audio_data)),
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Cache-Control": "public, max-age=3600",
                "Accept-Ranges": "bytes",
            }
        )
    except Exception as e:
        print(f"Error serving audio file {filename}: {e}")
        raise HTTPException(status_code=500, detail="Error serving audio file")


@app.get("/landmark/voice")
async def generate_landmark_voice(landmark_name: str, voice_id: Optional[str] = None):
    """
    Generate voice narration for a landmark using script from landmarks.json
    If no voice_id provided, creates a custom voice using the persona
    Returns MP3 audio file
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        # Debug: Check if API key is available
        api_key = os.getenv("ELEVENLABS_API_KEY")
        logger.info(
            f"API Key in endpoint: {bool(api_key)}, starts with: {api_key[:10] if api_key else 'None'}")

        # Get landmark info from landmarks JSON database
        landmark_info = get_landmark_info(landmark_name)

        if not landmark_info:
            raise HTTPException(
                status_code=404,
                detail=f"Landmark '{landmark_name}' not found in database"
            )

        # Use script for narration text
        narration_text = landmark_info.get("script")
        if not narration_text:
            # Provide a default narration if no script exists
            narration_text = f"You are viewing {landmark_name}. This is a notable landmark in the area."

        # For now, always use the default voice to avoid custom voice creation issues
        # Custom voice creation requires a paid ElevenLabs plan
        final_voice_id = voice_id if voice_id else "21m00Tcm4TlvDq8ikWAM"

        logger.info(
            f"Calling TTS with text length: {len(narration_text)}, voice_id: {final_voice_id}")

        # Generate audio using TTS service
        audio_data = tts_service.text_to_speech(
            text=narration_text, voice_id=final_voice_id)

        if audio_data is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate audio narration"
            )

        return Response(
            content=audio_data,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"attachment; filename={landmark_name.replace(' ', '_')}_narration.mp3"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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
            screen_width = int(data.get('screen_width', 640))
            screen_height = int(data.get('screen_height', 480))

            print(
                f"WebSocket received: lat={latitude}, lon={longitude}, alpha={alpha}, beta={beta}, gamma={gamma}, screen={screen_width}x{screen_height}")

            # Find visible landmarks based on GPS and orientation
            use_api = data.get('use_api', False)
            visible_landmarks = vision.find_visible_landmarks(
                latitude, longitude, alpha, use_api=use_api)

            if visible_landmarks:
                print(visible_landmarks)
                landmark = visible_landmarks[0]

                # Calculate 3D projection of building plane onto screen
                import math

                # Horizontal angle to landmark
                angle_diff = landmark['angle_diff']
                distance = landmark['distance']  # Distance in meters

                # Screen parameters (from client)
                fov_horizontal = 60  # degrees
                fov_vertical = 45    # degrees

                # Calculate horizontal screen position from angle
                # Map FOV to screen width
                pixels_per_degree = screen_width / fov_horizontal
                horizontal_offset = angle_diff * pixels_per_degree
                center_x = screen_width / 2 + horizontal_offset

                # Calculate vertical position from beta (tilt)
                # beta: positive = looking down, negative = looking up
                # Assume landmark is at eye level when beta = 0
                pixels_per_degree_v = screen_height / fov_vertical
                vertical_offset = beta * pixels_per_degree_v
                center_y = screen_height / 2 + vertical_offset

                # Calculate face size based on distance (closer = bigger)
                # Assume building face is ~10m tall, visible at 100m as 200px
                reference_distance = 100  # meters
                reference_size = 200  # pixels
                face_size = int(reference_size *
                                (reference_distance / max(distance, 10)))
                face_size = max(100, min(400, face_size))  # Clamp size

                # Calculate 3D perspective quad for building plane
                # Simulate a vertical wall facing the camera
                # The wall appears as a trapezoid based on viewing angle

                # Perspective distortion based on horizontal angle
                # The more off-center, the more skewed the plane appears
                angle_rad = math.radians(abs(angle_diff))
                horizontal_skew = math.sin(angle_rad) * 0.3  # 30% max skew

                # Perspective distortion based on vertical tilt
                # Looking up/down creates vertical perspective
                beta_rad = math.radians(beta)
                vertical_skew = math.sin(beta_rad) * 0.2  # 20% max skew

                # Define quad corners with perspective (clockwise from top-left)
                half_w = face_size / 2
                half_h = face_size / 2

                # Apply perspective transforms
                # If looking right: left side is closer (bigger)
                # If looking up: bottom is closer (bigger)
                left_scale = 1.0 - (horizontal_skew if angle_diff > 0 else 0)
                right_scale = 1.0 - (horizontal_skew if angle_diff < 0 else 0)
                top_scale = 1.0 - (vertical_skew if beta > 0 else 0)
                bottom_scale = 1.0 - (vertical_skew if beta < 0 else 0)

                quad = [
                    [center_x - half_w * left_scale * top_scale,
                        center_y - half_h * top_scale],      # top-left
                    [center_x + half_w * right_scale * top_scale,
                        center_y - half_h * top_scale],     # top-right
                    [center_x + half_w * right_scale * bottom_scale,
                        center_y + half_h * bottom_scale],  # bottom-right
                    [center_x - half_w * left_scale * bottom_scale,
                        center_y + half_h * bottom_scale]  # bottom-left
                ]

                result = {
                    "landmark": {
                        "name": landmark['name'],
                        "script": landmark.get('script', ''),
                        "persona": landmark.get('persona', ''),
                        "audio_url": landmark.get('audio_url', '')
                    },
                    "face_region": quad,  # Send as quad for 3D perspective
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

                print(f"Sending face_region quad: {quad}")
                print(
                    f"Center: ({center_x}, {center_y}), Size: {face_size}, Angle: {angle_diff}°, Beta: {beta}°")
            else:
                result = {
                    "landmark": {
                        "name": "No landmarks detected",
                        "script": "Try moving to a different location or point your camera toward nearby attractions.",
                        "persona": ""
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
