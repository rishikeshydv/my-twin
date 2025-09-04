from elevenlabs.client import ElevenLabs
import sounddevice as sd
import numpy as np
import webrtcvad
import wave
import tempfile

SAMPLE_RATE = 16000     # ElevenLabs expects 16 kHz PCM
CHANNELS = 1
DTYPE = "int16"

client = ElevenLabs(
    api_key="sk_aa1627e30b90a2510a4ed5cb888a53177bf43336f1f55aef"
)

vad = webrtcvad.Vad(2)  # Aggressiveness mode (0-3)

def frame_generator(frame_duration_ms, audio, sample_rate):
    """Yield PCM frames of `frame_duration_ms` from audio bytes."""
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)  # 2 bytes per sample (int16)
    offset = 0
    while offset + n < len(audio):
        yield audio[offset:offset + n]
        offset += n

def record_and_detect():
    """Record continuously, detect voice activity, return utterances."""
    buffer = []
    silence_counter = 0
    speaking = False

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE) as stream:
        print("🎤 Listening... (Ctrl+C to stop)")
        while True:
            # Read 30 ms of audio (480 samples at 16kHz)
            audio, _ = stream.read(int(SAMPLE_RATE * 0.03))
            pcm_data = audio.tobytes()

            # Voice activity detection
            is_speech = vad.is_speech(pcm_data, SAMPLE_RATE)

            if is_speech:
                buffer.append(pcm_data)
                speaking = True
                silence_counter = 0
            else:
                if speaking:
                    silence_counter += 1
                    # if silence for ~1.5s, finalize utterance
                    if silence_counter > int(1500 / 30):
                        yield b"".join(buffer)
                        buffer.clear()
                        speaking = False
                        silence_counter = 0

def save_wav(pcm_bytes):
    """Save PCM bytes to a temp wav file."""
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp.name, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(np.dtype(DTYPE).itemsize)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    return temp.name

def transcribe(file_path):
    with open(file_path, "rb") as f:
        result = client.speech_to_text.convert(
            file=f,
            model_id="scribe_v1",
            language_code="eng",
        )
    return result
