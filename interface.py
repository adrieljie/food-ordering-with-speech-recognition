import sounddevice as sd
import numpy as np

audio_buffer = []
def callback(indata, frames, time, status):
    audio_buffer.append(indata.copy())

with sd.InputStream(
    samplerate=16000,
    blocksize=1600,
    channels=1,
    dtype="float32",
    callback=callback
):
    print('Listening...')
    input('Press Enter to stop recording')

audio = np.concatenate(audio_buffer)
print(audio.shape)

duration = audio.shape[0] / 16000
print(f"Audio duration: {duration}s")