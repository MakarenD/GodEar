import os
import torch
import sounddevice as sd
import numpy as np
import time
import threading
import queue

class TTSEngine:
    def __init__(self, language='ru', speaker='aidar', device='cpu'):
        self.language = language
        self.speaker = speaker
        self.device = torch.device(device)
        self.sample_rate = 48000
        self.model = None
        self._load_model()
        
        self.playback_queue = queue.Queue()
        self.is_playing = False
        self.stop_event = threading.Event()
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()

    def _load_model(self):
        """Load Silero TTS model."""
        repo = 'snakers4/silero-models'
        model_id = 'silero_tts'
        
        # Mapping for languages to model names
        # For Silero v4/v3
        model_map = {
            'ru': 'v4_ru',
            'en': 'v3_en',
            'de': 'v3_de',
            'fr': 'v3_fr',
            'es': 'v3_es'
        }
        
        model_name = model_map.get(self.language, 'v4_ru')
        print(f"Loading Silero TTS model: {model_name} (speaker: {self.speaker})...")
        
        try:
            self.model, _ = torch.hub.load(repo_or_dir=repo,
                                          model=model_id,
                                          language=self.language,
                                          speaker=model_name,
                                          trust_repo=True)
            self.model.to(self.device)
        except Exception as e:
            print(f"Error loading TTS model: {e}")

    def update_settings(self, language=None, speaker=None):
        """Update language or speaker and reload if necessary."""
        changed = False
        if language and language != self.language:
            self.language = language
            changed = True
        if speaker and speaker != self.speaker:
            self.speaker = speaker
            # We don't necessarily need to reload model for speaker change within same language
            # but for simplicity we reload for now if language changed.
            
        if changed:
            self._load_model()

    def speak_async(self, text, output_device_id=None):
        """Queue text for synthesis and playback in background."""
        if not self.model or not text.strip():
            return
        self.playback_queue.put((text, output_device_id))

    def _playback_worker(self):
        """Thread worker to process TTS queue and play audio."""
        while not self.stop_event.is_set():
            try:
                text, device_id = self.playback_queue.get(timeout=0.5)
                
                # Generate audio
                try:
                    # Silero TTS models expect text as input and return audio tensor
                    audio = self.model.apply_tts(text=text,
                                               speaker=self.speaker,
                                               sample_rate=self.sample_rate)
                    
                    audio_np = audio.numpy()
                    
                    # Play audio
                    sd.play(audio_np, self.sample_rate, device=device_id)
                    sd.wait() # Wait for playback to finish before starting next one
                    
                except Exception as e:
                    print(f"TTS Synthesis/Playback error: {e}")
                finally:
                    self.playback_queue.task_done()
                    
            except queue.Empty:
                continue

    @staticmethod
    def get_output_devices():
        """Return list of (index, name) for output devices."""
        devices = sd.query_devices()
        return [(i, d['name']) for i, d in enumerate(devices) if d['max_output_channels'] > 0]

    @staticmethod
    def get_speakers_for_lang(lang):
        """Return available speakers for a given language."""
        speakers = {
            'ru': ['aidar', 'baya', 'kseniya', 'xenia', 'eugene'],
            'en': ['en_0', 'en_1', 'en_2', 'en_3', 'en_4', 'en_5', 'en_6', 'en_7', 'en_8', 'en_9'],
            'de': ['de_0', 'de_1', 'de_2'],
            'fr': ['fr_0', 'fr_1', 'fr_2'],
            'es': ['es_0', 'es_1', 'es_2']
        }
        return speakers.get(lang, ['default'])
