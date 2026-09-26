import sounddevice as sd
import numpy as np
import torch
import threading

from silero_vad import load_silero_vad

SAMPLE_RATE = 16000
BLOCK_SIZE = 512

SPEECH_THRESHOLD = 0.5
SPEECH_LIMIT = 10.0

is_speaking = False
stop_speaking = 0

stop_recording = threading.Event()

vad_model = load_silero_vad()

audio_buffer = []
def callback(indata, frames, time, status):
    global stop_speaking
    global is_speaking

    audio_chunk = indata[:, 0]
    audio_tensor = torch.from_numpy(audio_chunk.copy()).float()

    speaking_probability = vad_model(
        audio_tensor,
        SAMPLE_RATE
    ).item()
    
    chunk_duration = frames/SAMPLE_RATE
    
    if speaking_probability > SPEECH_THRESHOLD:
        print('Speaking')
        is_speaking = True
        stop_speaking = 0

    else:
        print('Silence')

        if is_speaking:
            stop_speaking += chunk_duration

            if stop_speaking >= SPEECH_LIMIT:
                print('Finish speaking')
                is_speaking = False
                stop_speaking = 0

                stop_recording.set()
                raise sd.CallbackStop

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    blocksize=BLOCK_SIZE,
    channels=1,
    dtype="float32",
    callback=callback
):
    print('Recording start...')

    stop_recording.wait()

print('Recording stop...')
