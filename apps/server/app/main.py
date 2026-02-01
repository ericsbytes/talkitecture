from .tts.tts import TTSService, read_generated_text
from .landmarks import LANDMARKS
from . import vision
from fastapi import FastAPI, HTTPException, Form, WebSocket
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
                        "script": landmark.get('script', ''),
                        "persona": landmark.get('persona', '')
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
