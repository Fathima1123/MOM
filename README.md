# Minutes of Meeting Generator

An automated tool that generates minutes of meeting from audio recordings using Streamlit, Deepgram, and OpenAI. This application is a **streamlined solution for generating, transcribing, and translating** minutes of meeting (MoM) directly from audio recordings. Built with **Streamlit**, it utilizes **Deepgram's audio transcription** and **OpenAI's language models** to produce detailed, structured MoM documents.


## Features

- Audio upload and live recording capabilities
- Automatic speech-to-text transcription
- Support for English and Japanese languages
- Generates structured Minutes of Meeting
- Task assignment tracking
- Downloadable output

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Fathima1123/MOM.git
cd MOM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your API keys:
```
DG_API_KEY=your_deepgram_api_key
OPEN_AI_TOKEN=your_openai_api_key
```

4. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Login using the default credentials:
   - Username: admin
   - Password: admin

2. Select your preferred mode:
   - Upload Audio: Upload pre-recorded audio files
   - Live Recording: Record audio directly in the application

3. Choose your preferred language for the Minutes of Meeting

4. Generate and download your Minutes of Meeting

## Dependencies

- streamlit
- python-dotenv
- deepgram-sdk
- openai
- audio-recorder-streamlit

## Security Note

Please ensure your `.env` file is included in `.gitignore` to protect your API keys.
