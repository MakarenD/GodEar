import os
import sys
import queue
import platform
import sounddevice as sd
import numpy as np
import torch
import torchaudio.transforms as T
import json
import argparse
from vosk import Model, KaldiRecognizer
from deep_translator import GoogleTranslator
import threading
import time

# Parameters
SAMPLERATE = 16000
CHUNK_SIZE = 512  # For VAD
VAD_THRESHOLD = 0.5
SILENCE_CHUNKS = 15 # Wait for 15 chunks (~0.5s) of silence to finalize

# Language name/code -> standard code (for Vosk + deep-translator)
LANG_ALIASES = {
    "german": "de", "english": "en", "russian": "ru", "french": "fr", "spanish": "es",
    "italian": "it", "portuguese": "pt", "dutch": "nl", "turkish": "tr", "chinese": "zh",
    "vietnamese": "vn",
}

# Vosk model mapping: language code -> model folder name
VOSK_MODELS = {
    "en": "vosk-model-small-en-us-0.15",
    "ru": "vosk-model-small-ru-0.22",
    "de": "vosk-model-small-de-0.15",
    "fr": "vosk-model-small-fr-0.22",
    "es": "vosk-model-small-es-0.42",
    "it": "vosk-model-small-it-0.22",
    "pt": "vosk-model-small-pt-0.3",
    "nl": "vosk-model-small-nl-0.22",
    "tr": "vosk-model-small-tr-0.3",
    "cn": "vosk-model-small-cn-0.22",
    "zh": "vosk-model-small-cn-0.22",
    "vn": "vosk-model-small-vn-0.4",
}


def _normalize_lang(lang):
    """Normalize language input (e.g. 'german' -> 'de')."""
    if not lang:
        return lang
    key = lang.strip().lower()
    return LANG_ALIASES.get(key, key)


def _get_input_devices():
    """Return list of (index, name) for input devices."""
    devices = sd.query_devices()
    return [(i, d['name']) for i, d in enumerate(devices) if d['max_input_channels'] > 0]


def _get_loopback_devices():
    """Return list of (device_info,) for WASAPI loopback devices (Windows only)."""
    if platform.system() != "Windows":
        return []
    try:
        import pyaudiowpatch as pyaudio
        p = pyaudio.PyAudio()
        try:
            loopbacks = list(p.get_loopback_device_info_generator())
            return loopbacks
        finally:
            p.terminate()
    except Exception:
        return []


def _interactive_menu(from_lang, to_lang):
    """Show menu and return (from_lang, to_lang, device_id, loopback, loopback_device_index)."""
    print("\n=== Speech Translation ===")
    from_in = input(f"Source language [{from_lang}]: ").strip() or from_lang
    to_in = input(f"Target language [{to_lang}]: ").strip() or to_lang
    from_lang, to_lang = from_in, to_in
    
    print("\n=== Audio Source ===")
    options = []
    option_data = []  # (device_id, loopback, loopback_device_index)
    
    # 1. Microphone (default)
    options.append("Microphone (default)")
    option_data.append((None, False, None))
    
    # 2. Loopback devices (speakers / playback) - from pyaudiowpatch
    loopbacks = _get_loopback_devices()
    for lb in loopbacks:
        name = lb.get("name", "Unknown")
        options.append(f"Loopback: {name}")
        option_data.append((None, True, lb["index"]))
    
    # 3. Input devices from sounddevice
    devices = _get_input_devices()
    for idx, name in devices:
        options.append(f"Input: {name}")
        option_data.append((idx, False, None))
    
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    
    try:
        default_input = sd.default.device[0]
        default_name = sd.query_devices(default_input, 'input')['name'] if default_input is not None else "Default"
    except Exception:
        default_name = "Default"
    print(f"\n  [Enter = 1 ({default_name})]")
    
    choice = input("Select (1-{}): ".format(len(options))).strip() or "1"
    
    device_id, loopback, loopback_device_index = None, False, None
    try:
        idx = int(choice)
        if 1 <= idx <= len(option_data):
            device_id, loopback, loopback_device_index = option_data[idx - 1]
    except (ValueError, IndexError):
        pass
    return from_lang, to_lang, device_id, loopback, loopback_device_index


def _get_loopback_device(device_index=None):
    """Get WASAPI loopback device. If device_index given, use it; else use default playback."""
    import pyaudiowpatch as pyaudio
    p = pyaudio.PyAudio()
    try:
        if device_index is not None:
            loopback_device = p.get_device_info_by_index(device_index)
            if not loopback_device.get("isLoopbackDevice", False):
                raise RuntimeError(f"Device {device_index} is not a loopback device")
            return p, loopback_device
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        if not default_speakers.get("isLoopbackDevice", False):
            for loopback in p.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    break
            else:
                raise RuntimeError("Loopback device not found. Run: python -m pyaudiowpatch")
        return p, default_speakers
    except Exception:
        p.terminate()
        raise


class SpeechTranslator:
    def __init__(self, from_lang="en", to_lang="ru", device_id=None, loopback=False, loopback_device_index=None):
        self.from_lang = _normalize_lang(from_lang) or from_lang
        self.to_lang = _normalize_lang(to_lang) or to_lang
        self.device_id = device_id
        self.loopback = loopback
        self.loopback_device_index = loopback_device_index
        
        print(f"Loading Vosk model for {self.from_lang}...")
        model_name = VOSK_MODELS.get(self.from_lang) or VOSK_MODELS.get(self.from_lang[:2])
        if not model_name:
            raise ValueError(f"No Vosk model for '{self.from_lang}'. Supported: {', '.join(VOSK_MODELS.keys())}.")
        model_path = f"models/{model_name}"
        if not os.path.exists(model_path):
            from setup_models import download_vosk_model
            download_vosk_model(model_name)
        self.vosk_model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.vosk_model, SAMPLERATE)
        
        print("Loading Silero VAD...")
        self.vad_model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                  model='silero_vad',
                                                  force_reload=False,
                                                  trust_repo=True)
        (self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks) = self.utils
        
        print(f"Loading translator ({self.from_lang} -> {self.to_lang})...")
        self.translator = GoogleTranslator(source=self.from_lang, target=self.to_lang)
        
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.speech_detected = False
        self.silence_count = 0
        
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        self.audio_queue.put(indata.copy())

    def _loopback_callback_factory(self, resampler, out_buffer, channels):
        """Create callback that resamples loopback audio to 16kHz and puts in queue."""
        def callback(in_data, frame_count, time_info, status):
            if status:
                print(status, file=sys.stderr)
            chunk = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1:
                chunk = chunk.reshape(-1, channels).mean(axis=1)  # stereo -> mono
            tensor = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0)  # (1, 1, N)
            with torch.no_grad():
                resampled = resampler(tensor).squeeze().numpy().astype(np.float32)
            out_buffer.extend(resampled)
            while len(out_buffer) >= CHUNK_SIZE:
                block = np.array(out_buffer[:CHUNK_SIZE], dtype=np.float32).reshape(-1, 1)
                del out_buffer[:CHUNK_SIZE]
                self.audio_queue.put(block)
            return (in_data, 0)  # paContinue
        return callback

    def translate(self, text):
        if not text.strip():
            return ""
        return self.translator.translate(text)

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
            if self.loopback:
                if platform.system() != "Windows":
                    print("--loopback is only supported on Windows (uses WASAPI).", file=sys.stderr)
                    return
                try:
                    import pyaudiowpatch as pyaudio
                except ImportError:
                    print("Install PyAudioWPatch for loopback: pip install PyAudioWPatch", file=sys.stderr)
                    return
                p, loopback_device = _get_loopback_device(self.loopback_device_index)
                src_rate = int(loopback_device["defaultSampleRate"])
                print(f"Capturing from: {loopback_device['name']} (loopback, {src_rate} Hz -> 16 kHz)")
                resampler = T.Resample(orig_freq=src_rate, new_freq=SAMPLERATE)
                out_buffer = []
                channels = loopback_device["maxInputChannels"]
                callback = self._loopback_callback_factory(resampler, out_buffer, channels)
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=loopback_device["maxInputChannels"],
                    rate=src_rate,
                    frames_per_buffer=CHUNK_SIZE,
                    input=True,
                    input_device_index=loopback_device["index"],
                    stream_callback=callback,
                )
                stream.start_stream()
                try:
                    self.process_loop()
                finally:
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
            else:
                device_name = sd.query_devices(self.device_id, 'input')['name'] if self.device_id is not None else "Default"
                print(f"Starting audio stream on device: {device_name} (ID: {self.device_id})")
                with sd.InputStream(device=self.device_id, samplerate=SAMPLERATE, channels=1, callback=self.audio_callback, blocksize=CHUNK_SIZE):
                    self.process_loop()
        except Exception as e:
            print(f"Failed to start audio stream: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Speech Translation")
    parser.add_argument("from_lang", nargs="?", default="en", help="Source language (e.g. en, ru, de)")
    parser.add_argument("to_lang", nargs="?", default="ru", help="Target language (e.g. en, ru, de)")
    parser.add_argument("--device", type=int, default=None, help="Audio input device ID (use list_devices.py to find)")
    parser.add_argument("--loopback", action="store_true", help="Capture from default playback (Discord, etc.) - Windows only")
    parser.add_argument("--loopback-device", type=int, default=None, metavar="ID", help="Loopback device index (use list_devices.py to see)")
    parser.add_argument("--no-menu", action="store_true", help="Skip interactive menu, use defaults for audio source")
    
    args = parser.parse_args()
    
    from_lang, to_lang = args.from_lang, args.to_lang
    device_id, loopback = args.device, args.loopback
    loopback_device_index = args.loopback_device
    if not args.no_menu and device_id is None and not args.loopback and loopback_device_index is None and sys.stdin.isatty():
        from_lang, to_lang, device_id, loopback, loopback_device_index = _interactive_menu(from_lang, to_lang)
    else:
        if loopback_device_index is not None:
            loopback = True
        
    translator = SpeechTranslator(from_lang, to_lang, device_id, loopback, loopback_device_index)
    translator.start()
