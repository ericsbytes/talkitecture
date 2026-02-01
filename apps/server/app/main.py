from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from typing import List, Optional
from pydantic import BaseModel
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import re
import json
from vision import analyze_video
import vision

# Import TTS service
from tts.tts import tts_service, read_generated_text

# Load buildings database
BUILDINGS_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "building_info", "buildings_database.json")

def get_building_info(building_name: str) -> Optional[dict]:
    """
    Read building info from buildings_database.json
    Returns dict with 'script', 'persona', etc., or None if not found
    """
    try:
        with open(BUILDINGS_DB_PATH, 'r', encoding='utf-8') as f:
            db = json.load(f)
        return db.get(building_name)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None
    except Exception as e:
        print(f"Error reading buildings database: {e}")
        return None

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
    """Analyze video and return landmark analysis"""

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


@app.post("/landmark/voice")
async def generate_landmark_voice(landmark_name: str, voice_id: Optional[str] = None):
    """
    Generate voice narration for a landmark using script from buildings_database.json
    If no voice_id provided, creates a custom voice using the persona
    Returns MP3 audio file
    """
    try:
        # Get building info from JSON database
        building_info = get_building_info(landmark_name)
        
        if not building_info:
            raise HTTPException(
                status_code=404,
                detail=f"Building '{landmark_name}' not found in database"
            )
        
        # Use script for narration text
        narration_text = building_info.get("script")
        if not narration_text:
            raise HTTPException(
                status_code=400,
                detail=f"No script found for '{landmark_name}'"
            )
        
        # If no voice_id provided, create a custom voice using persona
        final_voice_id = voice_id
        if not final_voice_id:
            persona = building_info.get("persona")
            if persona:
                # Parse persona into adjectives
                adjectives = [p.strip() for p in re.split(r"[,;]", persona) if p.strip()]
                # Create custom voice with building name and adjectives
                voice_result = tts_service.create_custom_voice(
                    name=f"{landmark_name}_voice",
                    description=persona,
                    prompt_adjectives=adjectives
                )
                if voice_result and "voice_id" in voice_result:
                    final_voice_id = voice_result["voice_id"]
        
        # Use default voice if custom voice creation failed or no persona
        if not final_voice_id:
            final_voice_id = "21m00Tcm4TlvDq8ikWAM"
        
        # Generate audio using TTS service
        audio_data = tts_service.text_to_speech(text=narration_text, voice_id=final_voice_id)
        
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
        raise HTTPException(status_code=500, detail=str(e))


class VoiceCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    adjectives: Optional[List[str]] = None
    landmark_name: Optional[str] = None


@app.post("/voice/create")
async def create_custom_voice(request: VoiceCreateRequest):
    """
    Create a custom voice on ElevenLabs using a prompt built from adjectives.
    If landmark_name is provided, pulls persona from buildings_database.json
    Returns the created voice metadata on success.
    """
    # If a landmark_name is provided and no adjectives were supplied, pull persona from building database
    adjectives = request.adjectives
    description = request.description
    
    if not adjectives and request.landmark_name:
        building_info = get_building_info(request.landmark_name)
        if building_info:
            persona = building_info.get("persona")
            if persona and isinstance(persona, str):
                adjectives = [p.strip() for p in re.split(r"[,;]", persona) if p.strip()]
                # Use persona as description if not explicitly provided
                if not description:
                    description = persona

    result = tts_service.create_custom_voice(
        name=request.name, 
        description=description or "", 
        prompt_adjectives=adjectives
    )

    if result is None:
        raise HTTPException(status_code=500, detail="Failed to create custom voice")

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
