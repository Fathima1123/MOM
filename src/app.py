import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
import time
from datetime import datetime
from audio_recorder_streamlit import audio_recorder
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    PrerecordedOptions
)
from openai import OpenAI
import json

load_dotenv()


def authenticate_user():
    """Handle user authentication"""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.title("Log in to Application")
        username = st.text_input("Username:", key="user")
        password = st.text_input("Password:", type="password", key="passwd")
        
        if st.button("Login"):
            if username.strip() == "admin" and password.strip() == "admin":
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Invalid username or password")
        return False
    return True

def transcribe_audio(audio_bytes, client):
    """Transcribe audio using Deepgram"""
    try:
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            utterances=True,
            punctuate=True,
            diarize=True,
        )
        
        response = client.listen.prerecorded.v("1").transcribe_file(
            {"buffer": audio_bytes},
            options
        )
        return response.to_dict()
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None

def create_transcript(response):
    """Create formatted transcript from Deepgram response"""
    if not response:
        return ""
        
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
    
    lines.append(f"SPEAKER {curr_speaker}:" + curr_line)
    return '\n'.join(lines)

def translate_text(text, target_language, openai_client):
    """Translate text to target language using OpenAI"""
    if target_language.lower() != 'english':
        sub_prompt = f' Translate the following diarized output to {target_language}'
    else:
        sub_prompt = ''
    
    prompt = (f"{sub_prompt}\n{text}\n\n"
             "This is the output text having multiple speakers from a diarization model. "
             "Find the person names from the output text given here and replace the speaker ids "
             "like SPEAKER 0, SPEAKER 1 etc with corresponding person names. "
             f"Generate complete words in {target_language}.")
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text

def create_prompt(transcript, language='english'):
    """Create prompt for MoM generation in specified language"""
    current_date = datetime.now().strftime("%d-%m-%Y")
    return f"""
    You are a MoM generator from the following transcript. Take the below conversation 
    from a meeting and generate the minutes of the meeting and create a detailed table 
    containing the list of tasks assigned to each person, the status of each task, 
    and the deadlines. Write dates as well in the output table. 
    Today is {current_date}. Identify the speaker names from the meeting transcript.
    
    Generate the Minutes of Meeting in {language} only.
    Format the output with clear sections:
    - Meeting Date
    - Attendees
    - Meeting Agenda
    - Discussion Points
    - Task Assignments (in table format)
    - Next Steps
    - Meeting Conclusion

    Transcript:
    {transcript}
    """

def generate_mom(prompt, openai_client):
    """Generate Minutes of Meeting using OpenAI"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"MoM Generation error: {str(e)}")
        return None


def main():
    st.set_page_config(page_title="Minutes of Meeting Generator", page_icon="👄")
    
    if not authenticate_user():
        return
        
    st.title("Minutes of Meeting Generator")
    
    try:
        dg_client = DeepgramClient(os.getenv("DG_API_KEY"))
        openai_client = OpenAI(api_key=os.getenv("OPEN_AI_TOKEN"))
    except Exception as e:
        st.error(f"Error initializing clients: {str(e)}")
        return
    
    mode = st.radio("Select Mode:", ["Upload Audio", "Live Recording"])
    
    # Language selection
    language = st.selectbox("Select the language for MoM:", ["English", "Japanese"])
    
    if mode == "Upload Audio":
        uploaded_file = st.file_uploader("Upload Audio File", type=['wav', 'mp3'])
        if uploaded_file:
            st.audio(uploaded_file)
            
            if st.button("Generate MoM"):
                with st.status("Transcribing and generating MoM...", expanded=True) as status:
                    try:
                        # Transcribe audio
                        start_time = time.time()
                        st.write("Transcribing audio...")
                        
                        audio_bytes = uploaded_file.read()
                        response = transcribe_audio(audio_bytes, dg_client)
                        transcript = create_transcript(response)
                        
                        transcribe_time = time.time() - start_time
                        st.write(f"Time taken to transcribe: {transcribe_time:.2f} seconds")
                        
                        # Translate if needed
                        if language.lower() != 'english':
                            start_time = time.time()
                            st.write(f"Translating to {language}...")
                            transcript = translate_text(transcript, language, openai_client)
                            translate_time = time.time() - start_time
                            st.write(f"Time taken to translate: {translate_time:.2f} seconds")
                        
                        # Generate MoM
                        st.write("Generating MoM...")
                        start_time = time.time()
                        
                        prompt = create_prompt(transcript, language)
                        mom = generate_mom(prompt, openai_client)
                        
                        generate_time = time.time() - start_time
                        st.write(f"Time taken to generate MoM: {generate_time:.2f} seconds")
                        
                        status.update(label="Processing Complete!", state="complete", expanded=False)
                        
                        # Display results
                        if transcript:
                            st.subheader(f"Transcript in {language}")
                            st.info(transcript)
                        
                        if mom:
                            st.subheader("Minutes of Meeting")
                            st.info(mom)
                            st.download_button(
                                "Download Minutes of Meeting",
                                mom,
                                "minutes_of_meeting.txt",
                                "text/plain"
                            )
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    else:  # Live Recording
        st.write("Click the microphone button to start/stop recording")
        
        audio_bytes = audio_recorder(
            pause_threshold=2.0,
            sample_rate=16000,
            neutral_color="#1976D2",
            recording_color="#ff0000",
            text="Click to Record"
        )
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            
            if st.button("Generate MoM"):
                with st.status("Processing audio...", expanded=True) as status:
                    try:
                        # Transcribe audio
                        start_time = time.time()
                        st.write("Transcribing audio...")
                        
                        response = transcribe_audio(audio_bytes, dg_client)
                        transcript = create_transcript(response)
                        
                        transcribe_time = time.time() - start_time
                        st.write(f"Time taken to transcribe: {transcribe_time:.2f} seconds")
                        
                        # Translate if needed
                        if language.lower() != 'english':
                            start_time = time.time()
                            st.write(f"Translating to {language}...")
                            transcript = translate_text(transcript, language, openai_client)
                            translate_time = time.time() - start_time
                            st.write(f"Time taken to translate: {translate_time:.2f} seconds")
                        
                        # Generate MoM
                        st.write("Generating MoM...")
                        start_time = time.time()
                        
                        prompt = create_prompt(transcript, language)
                        mom = generate_mom(prompt, openai_client)
                        
                        generate_time = time.time() - start_time
                        st.write(f"Time taken to generate MoM: {generate_time:.2f} seconds")
                        
                        status.update(label="Processing Complete!", state="complete", expanded=False)
                        
                        # Display results
                        if transcript:
                            st.subheader(f"Transcript in {language}")
                            st.info(transcript)
                        
                        if mom:
                            st.subheader("Minutes of Meeting")
                            st.info(mom)
                            st.download_button(
                                "Download Minutes of Meeting",
                                mom,
                                "minutes_of_meeting.txt",
                                "text/plain"
                            )
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()