import os
import requests

from dotenv import load_dotenv

# Load env variables
load_dotenv()

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_MODEL_ID = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")

VOICE_ID = os.environ.get("ELEVENLABS_ANANYA_VOICE_ID", "Ghr5KCyOzBvJpcdBbJhE")

def test_elevenlabs():
    if not ELEVENLABS_API_KEY:
        print("Error: ELEVENLABS_API_KEY is not set.")
        return
    
    text = "Hello! I am Ananya, your AI interviewer. I am ready to conduct your interview today. Are you ready?"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    
    print(f"Using Voice ID: {VOICE_ID}")
    print("Generating audio...")
    
    response = requests.post(
        url,
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        json={
            "text": text,
            "model_id": ELEVENLABS_MODEL_ID,
            "voice_settings": {
                "stability": 0.50,
                "similarity_boost": 0.60,
                "style": 0.50,
                "use_speaker_boost": True,
                "speed": 0.85,
            },
        },
        timeout=30,
    )
    
    if response.status_code == 200:
        output_file = "test_audio.mp3"
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"Success! Audio saved to {output_file}")
        
        # Play the audio file automatically on Windows
        try:
            os.startfile(output_file)
            print("Playing audio...")
        except Exception as e:
            print(f"Could not play audio automatically: {e}")
            
    else:
        print(f"Failed with status code: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_elevenlabs()
