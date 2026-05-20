"""
app/services/tts_service.py
NyrVexa — TTS with ElevenLabs primary, edge-tts free fallback.
"""
import os
import asyncio
import logging
import tempfile
import requests

logger = logging.getLogger(__name__)

ELEVENLABS_API_KEY   = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_MODEL_ID  = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
ELEVENLABS_TIMEOUT   = float(os.environ.get("ELEVENLABS_TIMEOUT", "30"))

# Only free/non-library ElevenLabs voice IDs work on the free tier.
# "Rachel" (21m00Tcm4TlvDq8ikWAM) is a built-in free voice.
VOICES = {
    "ananya": os.environ.get("ELEVENLABS_ANANYA_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
    "female": os.environ.get("ELEVENLABS_FEMALE_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
    "male":   os.environ.get("ELEVENLABS_MALE_VOICE_ID",   "TxGEqnHWrfWFTfGW9XjX"),
}

# edge-tts voice fallbacks (completely free, no API key needed)
EDGE_VOICES = {
    "ananya": "en-IN-NeerjaNeural",
    "female": "en-IN-NeerjaNeural",
    "male":   "en-IN-AaravNeural",
}
EDGE_FALLBACK = "en-US-AriaNeural"


# ── edge-tts fallback ────────────────────────────────────────────────────────

async def _edge_tts_async(text: str, voice: str) -> bytes:
    """Generate MP3 bytes via edge-tts (Microsoft free TTS)."""
    try:
        import edge_tts
    except ImportError:
        raise RuntimeError("edge-tts not installed. Run: pip install edge-tts")

    voice_name = EDGE_VOICES.get(voice, EDGE_FALLBACK)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        communicate = edge_tts.Communicate(text, voice_name)
        await communicate.save(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    except Exception as e:
        logger.warning("[TTS] edge-tts voice %s failed: %s", voice_name, e)
        # Try generic fallback
        communicate = edge_tts.Communicate(text, EDGE_FALLBACK)
        await communicate.save(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _edge_tts_sync(text: str, voice: str) -> bytes:
    """Sync wrapper for edge-tts async."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _edge_tts_async(text, voice))
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(_edge_tts_async(text, voice))
    except Exception:
        return asyncio.run(_edge_tts_async(text, voice))


# ── ElevenLabs primary ───────────────────────────────────────────────────────

def _elevenlabs_tts(text: str, voice: str) -> bytes:
    """Try ElevenLabs. Raises RuntimeError on any failure."""
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY not set")

    voice_id = VOICES.get(voice.lower(), VOICES["female"])
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

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
        timeout=ELEVENLABS_TIMEOUT,
    )

    if response.status_code != 200:
        raise RuntimeError(f"ElevenLabs error {response.status_code}: {response.text}")

    return response.content


# ── Public API ───────────────────────────────────────────────────────────────

def text_to_speech(text: str, voice: str = "ananya") -> bytes:
    """
    Convert text to MP3 bytes.
    Tries ElevenLabs first; falls back to edge-tts (free) on any error.

    Args:
        text:  Text to speak (max 500 chars recommended)
        voice: 'ananya' | 'female' | 'male'

    Returns:
        MP3 audio bytes
    """
    # ── Try ElevenLabs first ──
    if ELEVENLABS_API_KEY:
        try:
            logger.info("[TTS] Trying ElevenLabs voice: %s", voice)
            return _elevenlabs_tts(text, voice)
        except RuntimeError as e:
            logger.warning("[TTS] ElevenLabs failed (%s), falling back to edge-tts", e)
    else:
        logger.info("[TTS] No ElevenLabs key, using edge-tts directly")

    # ── Fallback: edge-tts (free, no API key) ──
    try:
        logger.info("[TTS] Using edge-tts voice: %s", EDGE_VOICES.get(voice, EDGE_FALLBACK))
        return _edge_tts_sync(text, voice)
    except Exception as e:
        logger.error("[TTS] edge-tts also failed: %s", e)
        raise RuntimeError(f"All TTS providers failed. Last error: {e}")