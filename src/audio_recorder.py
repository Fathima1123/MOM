import asyncio
import queue
import threading
from typing import Optional, Callable
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import time

class AudioRecorder:
    """Handles browser-based audio recording using WebRTC."""
    
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.recording = False
        self.send_function = None
        self.sample_rate = 16000
        self._setup_webrtc()
        
    def _setup_webrtc(self):
        """Sets up WebRTC configuration for audio recording."""
        self.webrtc_ctx = webrtc_streamer(
            key="audio-recorder",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=256,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={
                "audio": {
                    "echoCancellation": True,
                    "noiseSuppression": True,
                    "autoGainControl": True,
                },
                "video": False,
            }
        )

        if self.webrtc_ctx.audio_receiver:
            self.recording = True
            self._process_audio_thread = threading.Thread(
                target=self._process_audio_frames,
                args=(self.webrtc_ctx.audio_receiver,),
                daemon=True,
            )
            self._process_audio_thread.start()

    def _process_audio_frames(self, audio_receiver):
        """
        Processes incoming audio frames from WebRTC.
        
        Args:
            audio_receiver: WebRTC audio receiver object
        """
        while self.recording:
            try:
                audio_frames = audio_receiver.get_frames(timeout=1)
                for audio_frame in audio_frames:
                    self._handle_audio_frame(audio_frame)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing audio frames: {str(e)}")
                break

    def _handle_audio_frame(self, audio_frame: av.AudioFrame):
        """
        Processes individual audio frames and converts them to the required format.
        
        Args:
            audio_frame: Audio frame from WebRTC
        """
        try:
            # Convert audio frame to numpy array
            audio_data = audio_frame.to_ndarray()
            
            # Resample if necessary
            if audio_frame.sample_rate != self.sample_rate:
                audio_data = self._resample_audio(
                    audio_data,
                    audio_frame.sample_rate,
                    self.sample_rate
                )
            
            # Convert to the format expected by Deepgram
            audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            
            # Add to queue for processing
            self.audio_queue.put(audio_bytes)
            
            # Send to Deepgram if send function is set
            if self.send_function:
                self.send_function(audio_bytes)
                
        except Exception as e:
            print(f"Error handling audio frame: {str(e)}")

    def _resample_audio(self, audio_data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """
        Resamples audio data to the target sample rate.
        
        Args:
            audio_data: Input audio data
            src_rate: Source sample rate
            dst_rate: Target sample rate
            
        Returns:
            Resampled audio data
        """
        from scipy import signal
        
        # Calculate resampling ratio
        ratio = dst_rate / src_rate
        
        # Calculate new length
        new_length = int(len(audio_data) * ratio)
        
        # Resample using scipy
        resampled = signal.resample(audio_data, new_length)
        
        return resampled

    def start(self, send_function: Callable[[bytes], None]):
        """
        Starts the audio recording process.
        
        Args:
            send_function: Function to send audio data to Deepgram
        """
        self.send_function = send_function
        self.recording = True

    def stop(self):
        """Stops the audio recording process."""
        self.recording = False
        if hasattr(self, '_process_audio_thread'):
            self._process_audio_thread.join(timeout=1)
        self.send_function = None
        
    def is_recording(self) -> bool:
        """
        Checks if recording is currently active.
        
        Returns:
            True if recording, False otherwise
        """
        return self.recording and self.webrtc_ctx.state.playing

    def get_audio_data(self) -> Optional[bytes]:
        """
        Gets the next chunk of audio data from the queue.
        
        Returns:
            Audio data as bytes or None if queue is empty
        """
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

    def clear_queue(self):
        """Clears the audio data queue."""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break