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
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze-video")
async def analyze_video_endpoint(file: UploadFile = File(...)):
    """Analyze video and return landmark analysis"""
    # Save the uploaded file temporarily
    with open(f"temp_{file.filename}", "wb") as buffer:
        buffer.write(await file.read())

    # Analyze the video
    result = analyze_video(f"temp_{file.filename}")

    # Clean up temp file
    os.remove(f"temp_{file.filename}")

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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
