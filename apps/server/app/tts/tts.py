"""
ElevenLabs TTS Service
Generates voice narration
"""

import requests
import os
from typing import Optional, Dict
import logging
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file in the server directory
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)


class TTSService:
    """ElevenLabs Text-to-Speech Service"""

    def __init__(self, api_key: str):
        self.api_key = api_key.strip() if api_key else ""
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
        model_id: str = "eleven_turbo_v2",  # Free tier compatible model
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
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            return None

    def stream_text_to_speech(
        self,
        text: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model_id: str = "eleven_turbo_v2",  # Free tier compatible model
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
        Create a custom voice on ElevenLabs using voice design (text-to-voice).
        This is a two-step process:
        1. Generate voice previews from description
        2. Create voice from selected preview
        """
        # Build voice description from adjectives + description
        prompt_parts = []
        if prompt_adjectives:
            prompt_parts.append(
                ", ".join([str(a).strip() for a in prompt_adjectives if a]))
        if description:
            prompt_parts.append(description.strip())

        voice_description = "; ".join(
            prompt_parts) if prompt_parts else "A clear, natural voice"

        # Step 1: Generate voice previews
        design_payload = {
            "text": "Hello, this is a preview of the generated voice.",  # Sample text for preview
            "voice_description": voice_description,
            "model_id": "eleven_multilingual_ttv_v2"  # Text-to-voice model
        }

        try:
            # Generate previews
            response = requests.post(
                f"{self.base_url}/text-to-voice/design",
                headers={"Content-Type": "application/json",
                         "xi-api-key": self.api_key},
                json=design_payload,
                timeout=30
            )
            response.raise_for_status()
            previews = response.json()

            if not previews.get("previews"):
                logger.error("No voice previews generated")
                return None

            # Step 2: Create voice from first preview
            generated_voice_id = previews["previews"][0]["generated_voice_id"]

            create_payload = {
                "name": name,
                "description": description,
                "generated_voice_id": generated_voice_id
            }

            create_response = requests.post(
                f"{self.base_url}/text-to-voice/create",
                headers={"Content-Type": "application/json",
                         "xi-api-key": self.api_key},
                json=create_payload,
                timeout=30
            )
            create_response.raise_for_status()

            return create_response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to create custom voice: {e}")
            return None


# Initialize TTS service with API key from environment variable
api_key = os.getenv("ELEVENLABS_API_KEY")
logger.info(
    f"Loading API key... Found: {bool(api_key)}, Length: {len(api_key) if api_key else 0}")
if not api_key:
    raise ValueError(
        "ELEVENLABS_API_KEY environment variable not set. Please add it to .env file")

logger.info(
    f"Initializing TTS service with API key starting with: {api_key[:10] if api_key else 'None'}")
tts_service = TTSService(api_key=api_key)


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
