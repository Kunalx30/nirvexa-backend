import os

import requests


ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_MODEL_ID = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
ELEVENLABS_TIMEOUT = float(os.environ.get("ELEVENLABS_TIMEOUT", "30"))
ELEVENLABS_VOICE_SPEED = float(os.environ.get("ELEVENLABS_VOICE_SPEED", "1.0"))

VOICES = {
    "ananya": os.environ.get(
        "ELEVENLABS_ANANYA_VOICE_ID",
        "Ghr5KCyOzBvJpcdBbJhE",  # Indian English HR voice
    ),
    "female": os.environ.get(
        "ELEVENLABS_FEMALE_VOICE_ID",
        "Ghr5KCyOzBvJpcdBbJhE",  # Same as ananya for consistency
    ),
    "male": os.environ.get(
        "ELEVENLABS_MALE_VOICE_ID",
        "jBpfuIE2acCMzjSB6vNt",
    ),
}


def text_to_speech(text: str, voice: str = "ananya") -> bytes:
    """
    Convert text to MP3 audio bytes via ElevenLabs with optimized HR interview settings.
    
    Args:
        text: Text to convert to speech
        voice: Voice name ('ananya', 'female', 'male') - defaults to 'ananya' for Indian English HR
    
    Returns:
        MP3 audio bytes
    """
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY is not configured")

    voice_id = VOICES.get(voice.lower(), VOICES.get("ananya", VOICES["female"]))
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    # Optimized voice settings for professional HR interviews
    # Higher stability = more consistent, less emotional
    # Lower similarity_boost = more natural variation
    # Lower style = more formal/professional tone
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
                "stability": 0.50,           # Lower stability = more natural variation
                "similarity_boost": 0.60,   # Lower = more natural, less robotic
                "style": 0.50,              # Higher style = more expressive and warm
                "use_speaker_boost": True,
                "speed": 0.85,              # Slightly slower for better clarity
            },
        },
        timeout=ELEVENLABS_TIMEOUT,
    )

    if response.status_code != 200:
        raise RuntimeError(f"ElevenLabs error {response.status_code}: {response.text}")

    return response.content
