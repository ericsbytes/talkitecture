"""
TTS Module
Handles text-to-speech conversion using ElevenLabs API
"""

from tts.tts import TTSService, tts_service, read_generated_text
from tts.tts_api import app

__all__ = [
    "TTSService",
    "tts_service",
    "read_generated_text",
    "app"
]
