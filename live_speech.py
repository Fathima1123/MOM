import asyncio
import os
from dotenv import load_dotenv
from datetime import datetime
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

load_dotenv()

class TranscriptCollector:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.transcript_parts = []
        
    def add_part(self, part):
        if part.strip():  # Only add non-empty parts
            self.transcript_parts.append(part)
            
    def get_transcript(self):
        return ' '.join(self.transcript_parts)

async def main():
    # Initialize the transcript collector
    collector = TranscriptCollector()
    
    # Get Deepgram API Key
    DEEPGRAM_API_KEY = os.getenv("DG_API_KEY")
    if not DEEPGRAM_API_KEY:
        raise ValueError("Please set DG_API_KEY in your .env file")
        
    print("\nInitializing Deepgram connection...")
    
    try:
        # Create Deepgram client
        client = DeepgramClient(
            DEEPGRAM_API_KEY, 
            DeepgramClientOptions(options={"keepalive": "true"})
        )
        
        # Create a connection to Deepgram
        connection = client.listen.asynclive.v("1")
        
        # Define event handlers
        async def on_message(self, result, **kwargs):
            if result.is_final:
                sentence = result.channel.alternatives[0].transcript
                if sentence.strip():
                    print(f"\nTranscribed: {sentence}")
                    collector.add_part(sentence)
                    
        async def on_error(self, error, **kwargs):
            print(f"\nError: {error}")
            
        async def on_close(self, code, reason, **kwargs):
            print(f"\nConnection closed: {reason} ({code})")
            
        # Add event handlers
        connection.on(LiveTranscriptionEvents.Transcript, on_message)
        connection.on(LiveTranscriptionEvents.Error, on_error)
        connection.on(LiveTranscriptionEvents.Close, on_close)
        
        # Set up live transcription options
        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            smart_format=True
        )
        
        # Start the connection
        await connection.start(options)
        print("Connected to Deepgram. Start speaking...")
        
        # Initialize microphone
        microphone = Microphone(connection.send)
        
        # Start recording
        print("\nRecording... Press Ctrl+C to stop.")
        microphone.start()
        
        # Keep the connection alive
        while True:
            await asyncio.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nStopping recording...")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        
    finally:
        # Clean up
        if 'microphone' in locals():
            microphone.finish()
        if 'connection' in locals():
            await connection.finish()
            
        # Show final transcript
        final_transcript = collector.get_transcript()
        if final_transcript:
            print("\nFinal Transcript:")
            print("-" * 50)
            print(final_transcript)
            print("-" * 50)
            
            # Save transcript to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transcript_{timestamp}.txt"
            with open(filename, "w") as f:
                f.write(final_transcript)
            print(f"\nTranscript saved to: {filename}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"\nProgram error: {str(e)}")