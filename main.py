import os
import sys
import queue
import sounddevice as sd
import numpy as np
import torch
import json
from vosk import Model, KaldiRecognizer
import argostranslate.package
import argostranslate.translate
import threading
import time

# Parameters
SAMPLERATE = 16000
CHUNK_SIZE = 512  # For VAD
VAD_THRESHOLD = 0.5
SILENCE_CHUNKS = 15 # Wait for 15 chunks (~0.5s) of silence to finalize
VOSK_MODEL_EN = "models/vosk-model-small-en-us-0.15"
VOSK_MODEL_RU = "models/vosk-model-small-ru-0.22"

class SpeechTranslator:
    def __init__(self, from_lang="en", to_lang="ru"):
        self.from_lang = from_lang
        self.to_lang = to_lang
        
        print(f"Loading Vosk model for {from_lang}...")
        model_path = VOSK_MODEL_EN if from_lang == "en" else VOSK_MODEL_RU
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Vosk model not found at {model_path}. Please run setup_models.py first.")
        self.vosk_model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.vosk_model, SAMPLERATE)
        
        print("Loading Silero VAD...")
        self.vad_model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                  model='silero_vad',
                                                  force_reload=False,
                                                  trust_repo=True)
        (self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks) = self.utils
        
        print(f"Loading Argos Translate ({from_lang} -> {to_lang})...")
        self.translator = argostranslate.translate
        
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.speech_detected = False
        self.silence_count = 0
        
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        self.audio_queue.put(indata.copy())

    def translate(self, text):
        if not text.strip():
            return ""
        return self.translator.translate(text, self.from_lang, self.to_lang)

    def process_loop(self):
        print("\n--- Listening (Press Ctrl+C to stop) ---\n")
        self.is_running = True
        
        while self.is_running:
            try:
                chunk = self.audio_queue.get(timeout=1.0)
                
                audio_int16 = (chunk * 32768).astype(np.int16)
                audio_float32 = chunk.flatten().astype(np.float32)
                
                with torch.no_grad():
                    confidence = self.vad_model(torch.from_numpy(audio_float32), SAMPLERATE).item()
                
                if confidence > VAD_THRESHOLD:
                    if not self.speech_detected:
                        self.speech_detected = True
                    self.silence_count = 0
                else:
                    if self.speech_detected:
                        self.silence_count += 1
                        if self.silence_count > SILENCE_CHUNKS:
                            self.speech_detected = False
                            self.silence_count = 0
                            
                            final_res = json.loads(self.recognizer.FinalResult())
                            text = final_res.get("text", "")
                            if text:
                                translation = self.translate(text)
                                print(f"\r{' ' * 80}\r", end="")
                                print(f"USER: {text}")
                                print(f"TRAN: {translation}")
                                print("-" * 30)
                
                if self.recognizer.AcceptWaveform(audio_int16.tobytes()):
                    res = json.loads(self.recognizer.Result())
                    text = res.get("text", "")
                    if text and not self.speech_detected:
                        translation = self.translate(text)
                        print(f"\r{' ' * 80}\r", end="")
                        print(f"USER: {text}")
                        print(f"TRAN: {translation}")
                        print("-" * 30)
                else:
                    partial = json.loads(self.recognizer.PartialResult())
                    partial_text = partial.get("partial", "")
                    if partial_text:
                        print(f"\r>> {partial_text}", end="", flush=True)

            except queue.Empty:
                continue
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\nError in loop: {e}")

    def start(self):
        try:
            with sd.InputStream(samplerate=SAMPLERATE, channels=1, callback=self.audio_callback, blocksize=CHUNK_SIZE):
                self.process_loop()
        except Exception as e:
            print(f"Failed to start audio stream: {e}")

if __name__ == "__main__":
    from_lang = "en"
    to_lang = "ru"
    if len(sys.argv) > 2:
        from_lang = sys.argv[1]
        to_lang = sys.argv[2]
        
    translator = SpeechTranslator(from_lang, to_lang)
    translator.start()
