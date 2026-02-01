from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import re
from vision import analyze_video
from facts import get_landmark_facts

# Import TTS service
from tts.tts import tts_service, read_generated_text

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
    Generate voice narration for a landmark
    Returns MP3 audio file
    """
    try:
        # Get facts from facts database and derive narration text
        facts = get_landmark_facts(landmark_name)
        # Prefer an explicit 'script' or 'narration' field, fall back to joining 'facts' or using the name
        narration_text = None
        if isinstance(facts, dict):
            narration_text = (
                facts.get("script") or facts.get("narration") or
                (" ".join(facts.get("facts", [])) if facts.get("facts") else None) or
                facts.get("name")
            )

        if not narration_text:
            raise HTTPException(
                status_code=404,
                detail=f"Landmark '{landmark_name}' not found"
            )
        
        # Generate audio using TTS service (use custom voice if provided)
        audio_data = tts_service.text_to_speech(text=narration_text, voice_id=voice_id or "21m00Tcm4TlvDq8ikWAM")
        
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
    Returns the created voice metadata on success.
    """
    # If a landmark_name is provided and no adjectives were supplied, try to pull persona from facts
    adjectives = request.adjectives
    if not adjectives and request.landmark_name:
        facts = get_landmark_facts(request.landmark_name)
        if isinstance(facts, dict):
            persona = facts.get("persona")
            # persona may be a comma-separated string of adjectives
            if persona and isinstance(persona, str):
                # split on commas and semicolons
                adjectives = [p.strip() for p in re.split(r"[,;]", persona) if p.strip()]

    result = tts_service.create_custom_voice(name=request.name, description=request.description or "", prompt_adjectives=adjectives)

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
