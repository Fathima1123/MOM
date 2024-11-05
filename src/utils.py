import os
import logging
import httpx
from typing import Dict, Any, Optional
from datetime import datetime
import tempfile
from pathlib import Path
import json
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    PrerecordedOptions
)
from pydub import AudioSegment
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioUtils:
    """Utility functions for audio processing."""
    
    @staticmethod
    def convert_to_wav(audio_file, target_sample_rate: int = 16000) -> Path:
        """
        Converts uploaded audio to WAV format with specified sample rate.
        
        Args:
            audio_file: Uploaded audio file
            target_sample_rate: Desired sample rate
            
        Returns:
            Path to converted WAV file
        """
        try:
            # Create temp file
            temp_dir = Path(tempfile.gettempdir())
            temp_path = temp_dir / f"converted_{datetime.now().timestamp()}.wav"
            
            # Read audio file
            if isinstance(audio_file, (str, Path)):
                audio = AudioSegment.from_file(str(audio_file))
            else:
                # Handle StreamlitUploadedFile
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(audio_file.getvalue())
                    audio = AudioSegment.from_file(tmp.name)
                    os.unlink(tmp.name)
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Set sample rate
            if audio.frame_rate != target_sample_rate:
                audio = audio.set_frame_rate(target_sample_rate)
            
            # Export
            audio.export(temp_path, format="wav")
            return temp_path
            
        except Exception as e:
            logger.error(f"Error converting audio: {str(e)}")
            raise

class DeepgramUtils:
    """Utility functions for Deepgram integration."""
    
    @staticmethod
    def get_client() -> DeepgramClient:
        """
        Creates and returns a Deepgram client instance.
        
        Returns:
            DeepgramClient instance
        """
        api_key = os.getenv("DG_API_KEY")
        if not api_key:
            raise ValueError("Deepgram API key not found in environment variables")
            
        return DeepgramClient(
            api_key,
            DeepgramClientOptions(verbose=logging.DEBUG)
        )

    @staticmethod
    def get_transcription_options(
        language: str = "en",
        model: str = "nova-2"
    ) -> PrerecordedOptions:
        """
        Creates Deepgram transcription options.
        
        Args:
            language: Language code
            model: Deepgram model name
            
        Returns:
            PrerecordedOptions instance
        """
        return PrerecordedOptions(
            model=model,
            smart_format=True,
            utterances=True,
            punctuate=True,
            diarize=True,
            language=language
        )

def transcribe_uploaded_file(
    file,
    speech_language: str = "en",
    timeout: int = 300
) -> Dict[str, Any]:
    """
    Transcribes an uploaded audio file using Deepgram.
    
    Args:
        file: Uploaded audio file
        speech_language: Language of the audio
        timeout: Maximum time to wait for transcription
        
    Returns:
        Transcription response dictionary
    """
    try:
        # Convert audio to proper format
        wav_path = AudioUtils.convert_to_wav(file)
        
        # Read converted file
        with open(wav_path, 'rb') as audio_file:
            buffer_data = audio_file.read()
        
        # Clean up temp file
        os.unlink(wav_path)
        
        # Setup Deepgram client and options
        client = DeepgramUtils.get_client()
        options = DeepgramUtils.get_transcription_options(speech_language)
        
        # Perform transcription
        response = client.listen.prerecorded.v("1").transcribe_file(
            {"buffer": buffer_data},
            options,
            timeout=httpx.Timeout(timeout, connect=10.0)
        )
        
        return response.to_dict()
        
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise

def create_transcript(response: Dict[str, Any]) -> str:
    """
    Creates a formatted transcript from Deepgram response.
    
    Args:
        response: Deepgram transcription response
        
    Returns:
        Formatted transcript string
    """
    try:
        lines = []
        words = response["results"]["channels"][0]["alternatives"][0]["words"]
        
        curr_speaker = 0
        curr_line = ''
        
        for word_struct in words:
            word_speaker = word_struct["speaker"]
            word = word_struct["punctuated_word"]
            
            if word_speaker == curr_speaker:
                curr_line += ' ' + word
            else:
                tag = f"SPEAKER {curr_speaker}:"
                full_line = tag + curr_line + '\n'
                curr_speaker = word_speaker
                lines.append(full_line)
                curr_line = ' ' + word
                
        # Add final line
        lines.append(f"SPEAKER {curr_speaker}:" + curr_line)
        
        return '\n'.join(lines)
        
    except Exception as e:
        logger.error(f"Error creating transcript: {str(e)}")
        raise

def create_cache_key(*args: Any) -> str:
    """
    Creates a cache key for Streamlit caching.
    
    Args:
        *args: Values to include in cache key
        
    Returns:
        Cache key string
    """
    return str(hash(json.dumps(args, sort_keys=True)))

@st.cache_data(ttl=3600)
def cached_transcribe_file(
    file_content: bytes,
    speech_language: str = "en"
) -> Dict[str, Any]:
    """
    Cached version of file transcription.
    
    Args:
        file_content: Audio file content
        speech_language: Language of the audio
        
    Returns:
        Transcription response dictionary
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file_content)
        result = transcribe_uploaded_file(tmp.name, speech_language)
        os.unlink(tmp.name)
        return result

def format_time(seconds: float) -> str:
    """
    Formats time in seconds to HH:MM:SS format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    return str(datetime.timedelta(seconds=round(seconds)))

def sanitize_filename(filename: str) -> str:
    """
    Sanitizes a filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    return "".join(c for c in filename if c.isalnum() or c in "._- ")