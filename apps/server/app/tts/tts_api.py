"""
FastAPI endpoints for TTS functionality
"""

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import logging

from tts import tts_service, read_generated_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


class LandmarkTTSRequest(BaseModel):
    """Request model for reading text file and generating TTS"""
    file_path: str
    voice_id: Optional[str] = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice
    

class TextTTSRequest(BaseModel):
    """Request model for direct text TTS"""
    text: str
    voice_id: Optional[str] = "21m00Tcm4TlvDq8ikWAM"
    

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AR Landmark TTS API",
        "endpoints": {
            "voices": "/voices",
            "text_from_file_audio": "/audio/from-file",
            "text_from_file_audio_stream": "/audio/from-file/stream",
            "text_to_speech": "/tts",
            "text_to_speech_stream": "/tts/stream"
        }
    }


@app.get("/voices")
async def get_voices():
    """Get available ElevenLabs voices"""
    voices = tts_service.get_voices()
    
    if voices is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve voices from ElevenLabs"
        )
    
    return voices


@app.post("/audio/from-file")
async def generate_audio_from_file(request: LandmarkTTSRequest):
    """
    Read text from a file and generate audio narration
    Returns audio file directly
    """
    try:
        # Read text from file
        text = read_generated_text(request.file_path)
        
        if text is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to read text from file"
            )
        
        logger.info(f"Generating audio for text from file: {text[:50]}...")
        
        # Generate audio
        audio_data = tts_service.text_to_speech(
            text=text,
            voice_id=request.voice_id
        )
        
        if audio_data is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate audio"
            )
        
        return Response(
            content=audio_data,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=narration.mp3"
            }
        )
    
    except Exception as e:
        logger.error(f"Error generating audio from file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/audio/from-file/stream")
async def stream_audio_from_file(request: LandmarkTTSRequest):
    """
    Read text from a file and stream audio narration
    Returns streaming audio response
    """
    try:
        # Read text from file
        text = read_generated_text(request.file_path)
        
        if text is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to read text from file"
            )
        
        logger.info(f"Streaming audio for text from file: {text[:50]}...")
        
        # Stream audio
        return StreamingResponse(
            tts_service.stream_text_to_speech(
                text=text,
                voice_id=request.voice_id
            ),
            media_type="audio/mpeg"
        )
    
    except Exception as e:
        logger.error(f"Error streaming audio from file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts")
async def text_to_speech(request: TextTTSRequest):
    """
    Convert any text to speech
    Returns audio file directly
    """
    try:
        logger.info(f"Generating TTS for: {request.text[:50]}...")
        
        audio_data = tts_service.text_to_speech(
            text=request.text,
            voice_id=request.voice_id
        )
        
        if audio_data is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate audio"
            )
        
        return Response(
            content=audio_data,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=tts_output.mp3"
            }
        )
    
    except Exception as e:
        logger.error(f"Error in text to speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/stream")
async def stream_text_to_speech(request: TextTTSRequest):
    """
    Stream text to speech
    Returns streaming audio response
    """
    try:
        logger.info(f"Streaming TTS for: {request.text[:50]}...")
        
        return StreamingResponse(
            tts_service.stream_text_to_speech(
                text=request.text,
                voice_id=request.voice_id
            ),
            media_type="audio/mpeg"
        )
    
    except Exception as e:
        logger.error(f"Error streaming TTS: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)