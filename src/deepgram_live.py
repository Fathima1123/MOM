import asyncio
import os
from typing import Callable, Optional
from dotenv import load_dotenv
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

load_dotenv()

class TranscriptCollector:
    """Collects and manages transcript parts for complete sentences."""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part: str):
        self.transcript_parts.append(part)

    def get_full_transcript(self) -> str:
        return ' '.join(self.transcript_parts)

class DeepgramLiveTranscriber:
    """Handles live audio transcription using Deepgram."""
    
    def __init__(self):
        self.api_key = os.getenv("DG_API_KEY")
        if not self.api_key:
            raise ValueError("Deepgram API key not found in environment variables")
            
        self.transcript_collector = TranscriptCollector()
        self.callback = None
        self.transcription_complete = None

    async def setup_connection(self):
        """Sets up the Deepgram connection with specified options."""
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram = DeepgramClient(self.api_key, config)
        return deepgram.listen.asynclive.v("1")

    async def on_message(self, connection, result, **kwargs):
        """Handles incoming transcription messages."""
        try:
            sentence = result.channel.alternatives[0].transcript
            
            if not sentence.strip():
                return
                
            if not result.speech_final:
                self.transcript_collector.add_part(sentence)
            else:
                self.transcript_collector.add_part(sentence)
                full_sentence = self.transcript_collector.get_full_transcript()
                
                if full_sentence.strip():
                    if self.callback:
                        self.callback(full_sentence)
                    
                self.transcript_collector.reset()
                
                if self.transcription_complete:
                    self.transcription_complete.set()

        except Exception as e:
            print(f"Error processing message: {str(e)}")

    async def on_error(self, connection, error, **kwargs):
        """Handles connection errors."""
        print(f"Deepgram error: {error}")

    async def on_close(self, connection, code, reason, **kwargs):
        """Handles connection closure."""
        print(f"Connection closed with code {code}: {reason}")

    async def on_metadata(self, connection, metadata, **kwargs):
        """Handles metadata events."""
        print(f"Received metadata: {metadata}")

    def get_live_options(self) -> LiveOptions:
        """Configures and returns Deepgram live transcription options."""
        return LiveOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=300,
            smart_format=True,
            interim_results=True,
            utterance_end_ms=1000,
            vad_events=True
        )

    async def process_audio(self, audio_source, callback: Callable[[str], None]):
        """
        Processes audio from the given source and calls the callback with transcribed text.
        
        Args:
            audio_source: Source of audio data (must implement required interfaces)
            callback: Function to call with transcribed text
        """
        try:
            self.callback = callback
            self.transcription_complete = asyncio.Event()
            
            connection = await self.setup_connection()
            
            # Set up event handlers
            connection.on(LiveTranscriptionEvents.Transcript, self.on_message)
            connection.on(LiveTranscriptionEvents.Error, self.on_error)
            connection.on(LiveTranscriptionEvents.Close, self.on_close)
            connection.on(LiveTranscriptionEvents.Metadata, self.on_metadata)

            # Start the connection with specified options
            await connection.start(self.get_live_options())

            # Start processing audio from the source
            audio_source.start(connection.send)

            # Wait for completion or error
            try:
                await self.transcription_complete.wait()
            except asyncio.CancelledError:
                print("Transcription cancelled")
            finally:
                # Clean up
                audio_source.stop()
                await connection.finish()
                
        except Exception as e:
            print(f"Error in process_audio: {str(e)}")
            raise

    async def stop(self):
        """Stops the transcription process."""
        if self.transcription_complete:
            self.transcription_complete.set()