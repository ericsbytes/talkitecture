"""
ElevenLabs TTS Service
Generates voice narration
"""

import requests
import os
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

class TTSService:
    """ElevenLabs Text-to-Speech Service"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.elevenlabs.io/v1"
        self.headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
    def get_voices(self) -> Optional[Dict]:
        """Retrieve available voices from ElevenLabs"""
        try:
            response = requests.get(
                f"{self.base_url}/voices",
                headers={"xi-api-key": self.api_key}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch voices: {e}")
            return None
    
    def text_to_speech(
        self,
        text: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # Default: Rachel voice
        model_id: str = "eleven_monolingual_v1",
        voice_settings: Optional[Dict] = None
    ) -> Optional[bytes]:
        """
        Convert text to speech using ElevenLabs API
        
        Args:
            text: The text to convert to speech
            voice_id: ElevenLabs voice ID (default: Rachel)
            model_id: Model to use for generation
            voice_settings: Optional voice configuration
            
        Returns:
            Audio data as bytes or None if failed
        """
        if voice_settings is None:
            voice_settings = {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True
            }
        
        data = {
            "text": text,
            "model_id": model_id,
            "voice_settings": voice_settings
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/text-to-speech/{voice_id}",
                json=data,
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            logger.error(f"TTS generation failed: {e}")
            return None
    
    def stream_text_to_speech(
        self,
        text: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model_id: str = "eleven_monolingual_v1",
        voice_settings: Optional[Dict] = None
    ):
        """
        Stream text to speech for real-time playback
        
        Yields audio chunks for streaming
        """
        if voice_settings is None:
            voice_settings = {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True
            }
        
        data = {
            "text": text,
            "model_id": model_id,
            "voice_settings": voice_settings
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/text-to-speech/{voice_id}/stream",
                json=data,
                headers=self.headers,
                stream=True,
                timeout=30
            )
            response.raise_for_status()
            
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    yield chunk
        except requests.exceptions.RequestException as e:
            logger.error(f"TTS streaming failed: {e}")
            raise

    def create_custom_voice(self, name: str, description: str = "", prompt_adjectives: Optional[list] = None) -> Optional[Dict]:
        """
        Create a custom voice on ElevenLabs using a prompt generated from adjectives.

        Args:
            name: Desired name for the custom voice
            description: Optional description
            prompt_adjectives: List of adjectives or short phrases describing the desired voice

        Returns:
            Response JSON from ElevenLabs if successful, otherwise None
        """
        # Build a voice prompt from adjectives + description
        prompt_parts = []
        if prompt_adjectives:
            # ensure all adjectives are str
            prompt_parts.append(", ".join([str(a).strip() for a in prompt_adjectives if a]))
        if description:
            prompt_parts.append(description.strip())

        voice_prompt = "; ".join(prompt_parts) if prompt_parts else ""

        payload = {
            "name": name,
            "description": description,
            # use a field 'voice_prompt' to pass natural-language instructions
            "voice_prompt": voice_prompt
        }

        try:
            response = requests.post(
                f"{self.base_url}/voices",
                headers={"Content-Type": "application/json", "xi-api-key": self.api_key},
                json=payload,
                timeout=30
            )

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to create custom voice via prompt: {e}")
            return None


# Initialize TTS service with your API key
tts_service = TTSService(api_key="sk_d9a4154b78f417f531cc952c8dc18c2329d9161ca3eee91f")


def read_generated_text(file_path: str) -> Optional[str]:
    """
    Read pre-generated narration text from a file
    
    Args:
        file_path: Path to the file containing generated text
        
    Returns:
        The text content from the file, or None if failed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        return text
    except FileNotFoundError:
        logger.error(f"Text file not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Failed to read text file: {e}")
        return None