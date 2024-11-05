import os
from datetime import datetime
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class MoMGenerator:
    """Handles generation of Minutes of Meeting and translation using OpenAI."""
    
    def __init__(self):
        self.api_key = os.getenv("OPEN_AI_TOKEN")
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
            
        self.client = OpenAI(api_key=self.api_key)
        
    def translate_text(self, text: str, target_language: str) -> str:
        """
        Translates text to target language using OpenAI.
        
        Args:
            text: Text to translate
            target_language: Target language for translation
            
        Returns:
            Translated text
        """
        if target_language.lower() == 'english':
            return text
            
        prompt = f"""
        Translate the following diarized output to {target_language}:
        
        {text}
        
        This is output text from a diarization model having multiple speakers. 
        Find the person names from the output text given here and replace the speaker ids 
        with corresponding person names. Generate complete words in {target_language}.
        Give the output in conversational manner.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000
            )
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Translation error: {str(e)}")
            raise

    def create_mom_prompt(self, transcript: str, language: str = 'english') -> str:
        """
        Creates the prompt for MoM generation.
        
        Args:
            transcript: Meeting transcript
            language: Target language for MoM
            
        Returns:
            Formatted prompt for OpenAI
        """
        current_date = datetime.now().strftime("%d-%m-%Y")
        
        return f"""
        You are a professional Minutes of Meeting generator. Using the conversation transcript below:
        
        1. Identify all participants and their roles
        2. Create a concise summary of the main discussion points
        3. List all decisions made during the meeting
        4. Create a detailed table containing:
           - Tasks assigned
           - Person responsible
           - Current status
           - Deadlines
        5. Note any follow-up actions required
        6. Include any important dates mentioned
        
        Today's date is {current_date}.
        Generate the Minutes of Meeting in {language} only.
        Format the output in a professional manner with clear sections and bullet points.

        Transcript:
        {transcript}
        """

    def generate_mom(self, transcript: str, language: str = 'english', max_retries: int = 3) -> str:
        """
        Generates Minutes of Meeting from transcript.
        
        Args:
            transcript: Meeting transcript
            language: Target language for MoM
            max_retries: Maximum number of retry attempts
            
        Returns:
            Generated Minutes of Meeting
        """
        prompt = self.create_mom_prompt(transcript, language)
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.7,
                    presence_penalty=0.6,
                    frequency_penalty=0.3
                )
                return response.choices[0].message.content
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to generate MoM after {max_retries} attempts: {str(e)}")
                    raise
                print(f"Attempt {attempt + 1} failed, retrying...")
                continue

# Create global instances for easier imports
_mom_generator = None

def get_mom_generator() -> MoMGenerator:
    """
    Gets or creates a global MoMGenerator instance.
    
    Returns:
        MoMGenerator instance
    """
    global _mom_generator
    if _mom_generator is None:
        _mom_generator = MoMGenerator()
    return _mom_generator

def translate_text(text: str, target_language: str) -> str:
    """
    Convenience function for text translation.
    
    Args:
        text: Text to translate
        target_language: Target language
        
    Returns:
        Translated text
    """
    return get_mom_generator().translate_text(text, target_language)

def generate_mom(transcript: str, language: str = 'english') -> str:
    """
    Convenience function for MoM generation.
    
    Args:
        transcript: Meeting transcript
        language: Target language
        
    Returns:
        Generated Minutes of Meeting
    """
    return get_mom_generator().generate_mom(transcript, language)