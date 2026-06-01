"""
app/services/tts_service.py
NirVexa TTS service.
"""
import asyncio
import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

TTS_POOL_SIZE = int(os.environ.get("TTS_POOL_SIZE", "16"))
TTS_SPEED = float(os.environ.get("TTS_SPEED", "1.3"))
_tts_executor = ThreadPoolExecutor(max_workers=TTS_POOL_SIZE, thread_name_prefix="tts")


def _gtts_tts(text: str) -> bytes:
    from gtts import gTTS

    buf = io.BytesIO()
    gTTS(text=text, lang="en", tld="co.in", slow=False).write_to_fp(buf)
    buf.seek(0)

    try:
        from pydub import AudioSegment

        audio = AudioSegment.from_mp3(buf)
        fast_audio = audio._spawn(
            audio.raw_data,
            overrides={"frame_rate": int(audio.frame_rate * TTS_SPEED)},
        ).set_frame_rate(audio.frame_rate)

        out = io.BytesIO()
        fast_audio.export(out, format="mp3")
        out.seek(0)
        return out.read()
    except Exception as e:
        logger.warning("[TTS] gTTS speedup failed (%s), using original speed", e)
        buf.seek(0)
        return buf.read()


def text_to_speech(text: str, voice: str = "ananya") -> bytes:
    """
    Convert text to MP3 bytes using gTTS Indian English.

    ElevenLabs free-tier blocking and edge-tts 403s were causing production
    latency and noisy fallback chains. Keep the server path predictable and let
    the frontend browser speech fallback handle rare gTTS/network failures.
    """
    try:
        logger.info("[TTS] Using gTTS Indian English voice: %s", voice)
        return _gtts_tts(text)
    except Exception as e:
        logger.error("[TTS] gTTS failed: %s", e)
        raise RuntimeError(f"TTS generation failed: {e}")


async def synthesize_speech(text: str, voice: str = "ananya", **_kwargs):
    loop = asyncio.get_running_loop()
    audio = await loop.run_in_executor(_tts_executor, text_to_speech, text, voice)
    return audio, "audio/mpeg"


def tts_status() -> dict:
    return {
        "provider": "gTTS",
        "gtts_speed": TTS_SPEED,
        "accent": "Indian English",
    }
