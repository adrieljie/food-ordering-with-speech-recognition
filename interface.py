import sounddevice as sd
import numpy as np
import torch
import threading
import queue

from silero_vad import load_silero_vad

import whisper

SAMPLE_RATE = 16000
BLOCK_SIZE = 512

SPEECH_THRESHOLD = 0.5
SPEECH_LIMIT = 5.0

PARTIAL_INTERVAL = 1.0

stop_recording = threading.Event()

vad_model = load_silero_vad()

whisper_model = whisper.load_model('base')

audio_queue = queue.Queue()
whisper_queue = queue.Queue(maxsize=1)

def callback(indata, frames, time, status):

    audio_chunk = indata[:, 0].copy()
    audio_queue.put(audio_chunk)

def send_to_whisper(audio, transcription_type):
    if transcription_type == "partial":
        if whisper_queue.full():
            return

    else:
        while whisper_queue.empty():
            try:
                whisper_queue.get_nowait()
            except queue.Empty:
                break

    whisper_queue.put((transcription_type, audio))

def vad():
    is_speaking = False
    stop_speaking = 0
    audio_buffer = []
    next_partial = PARTIAL_INTERVAL

    while not stop_recording.is_set():
        try:
            audio_chunk = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        audio_tensor = torch.from_numpy(audio_chunk.copy()).float()
        speaking_probability = vad_model(
                audio_tensor,
                SAMPLE_RATE
            ).item()
            
        chunk_duration = len(audio_chunk)/SAMPLE_RATE
            
        if speaking_probability > SPEECH_THRESHOLD:
            if not is_speaking:
                print("Recording started...")
                is_speaking = True

                audio_buffer = []
                next_partial = PARTIAL_INTERVAL

            audio_buffer.append(audio_chunk)
            stop_speaking = 0
            audio_duration = (sum(len(chunk) for chunk in audio_buffer) / SAMPLE_RATE)

            if audio_duration >= next_partial:
                audio = np.concatenate(audio_buffer).astype(np.float32)
                send_to_whisper(audio, "partial")

                next_partial += PARTIAL_INTERVAL

    
        else:
    
            if is_speaking:
                audio_buffer.append(audio_chunk)
                stop_speaking += chunk_duration
    
                if stop_speaking >= SPEECH_LIMIT:
                    print('Finish speaking')
                    final_audio = np.concatenate(audio_buffer).astype(np.float32)
                    send_to_whisper(final_audio, "final")
                    is_speaking = False
                    stop_speaking = 0
    
                    audio_buffer = []
                    next_partial = PARTIAL_INTERVAL

def whisper_worker():

    while not stop_recording.is_set():

        try:

            transcription_type, audio = (
                whisper_queue.get(timeout=0.1)
            )

        except queue.Empty:
            continue


        result = whisper_model.transcribe(
            audio,
            language="id",
            fp16=False,
            verbose=False
        )


        text = result["text"].strip()


        if transcription_type == "partial":

            print(
                "PARTIAL:",
                text
            )

        else:

            print(
                "FINAL:",
                text
            )

vad_thread = threading.Thread(
    target=vad,
    daemon=True
)

whisper_thread = threading.Thread(
    target=whisper_worker,
    daemon=True
)

vad_thread.start()
whisper_thread.start()


with sd.InputStream(
    samplerate=SAMPLE_RATE,
    blocksize=BLOCK_SIZE,
    channels=1,
    dtype="float32",
    callback=callback
):

    print("Listening...")
    print("Press Enter to stop application.")

    input()


stop_recording.set()

print("Application stopped.")
