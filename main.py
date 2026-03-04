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
import shutil

def get_base_path():
    """Get the absolute path to the directory containing the executable or script."""
    if getattr(sys, 'frozen', False):
        # Running as a bundled executable
        return os.path.dirname(sys.executable)
    # Running in normal Python environment
    return os.path.dirname(os.path.abspath(__file__))

# Parameters
SAMPLERATE = 16000
CHUNK_SIZE = 512  # For VAD
VAD_THRESHOLD = 0.5
SILENCE_CHUNKS = 7 # Wait for 7 chunks (~0.22s) of silence to finalize

# Engine constants
ENGINE_VOSK = "vosk"
ENGINE_WHISPER = "whisper"
ENGINE_WHISPER_LITE = "whisper-lite"

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
    """Return list of (device_info,) for loopback devices."""
    sys_plat = platform.system()
    if sys_plat == "Windows":
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
    elif sys_plat == "Linux":
        try:
            devices = sd.query_devices()
            loopbacks = []
            for i, d in enumerate(devices):
                if d['max_input_channels'] > 0 and 'monitor' in d['name'].lower():
                    loopbacks.append({"name": d['name'], "index": i})
            return loopbacks
        except Exception:
            return []
    elif sys_plat == "Darwin":
        # ScreenCaptureKit doesn't expose "devices" in the same way, 
        # but we can return a virtual entry to indicate support.
        return [{"name": "System Audio (ScreenCaptureKit)", "index": -1}]
    return []


def _interactive_menu(from_lang, to_lang):
    """Show menu and return (from_lang, to_lang, device_id, loopback, loopback_device_index, engine, show_overlay)."""
    print("\n=== Speech Translation ===")
    from_in = input(f"Source language [{from_lang}]: ").strip() or from_lang
    to_in = input(f"Target language [{to_lang}]: ").strip() or to_lang
    from_lang, to_lang = from_in, to_in

    print("\n=== Recognition Engine ===")
    print(f"  1. Vosk (Fast, Local)")
    print(f"  2. Whisper Medium (High accuracy, requires GPU/High CPU)")
    print(f"  3. Whisper Base (Lite, Faster than Medium)")
    engine_choice = input("Select (1-3) [1]: ").strip() or "1"
    engine = ENGINE_VOSK
    if engine_choice == "2": engine = ENGINE_WHISPER
    elif engine_choice == "3": engine = ENGINE_WHISPER_LITE
    
    show_overlay = input("\nShow transparent overlay? (y/n) [n]: ").strip().lower() == "y"

    print("\n=== Audio Source ===")
    options = []
    option_data = []  # (device_id, loopback, loopback_device_index)
    
    # 1. Microphone (default)
    options.append("Microphone (default)")
    option_data.append((None, False, None))
    
    # 2. Loopback devices (speakers / playback)
    loopbacks = _get_loopback_devices()
    for lb in loopbacks:
        name = lb.get("name", "Unknown")
        options.append(f"Loopback: {name}")
        option_data.append((None, True, lb["index"]))
    
    # 3. Input devices from sounddevice
    devices = _get_input_devices()
    for idx, name in devices:
        # Avoid duplicate listing of monitor devices on Linux if they are already in loopbacks
        if any(lb["index"] == idx for lb in loopbacks if "index" in lb):
            continue
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
    return from_lang, to_lang, device_id, loopback, loopback_device_index, engine, show_overlay


def _get_loopback_device(device_index=None):
    """Get WASAPI loopback device (Windows)."""
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


class MacLoopbackCapture:
    """Helper to capture system audio on macOS using ScreenCaptureKit."""
    def __init__(self, audio_queue):
        self.audio_queue = audio_queue
        self.stream = None
        self.delegate = None
        self.is_running = False

        try:
            import objc
            from ScreenCaptureKit import (
                SCStream, SCShareableContent, SCStreamConfiguration,
                SCContentFilter, SCStreamOutputTypeAudio
            )
            import CoreMedia
        except ImportError:
            print("To use loopback on macOS, install: pip install pyobjc-framework-ScreenCaptureKit pyobjc-framework-CoreMedia")
            raise

    def start(self):
        import objc
        from Foundation import NSObject, NSRunLoop, NSDate
        from ScreenCaptureKit import (
            SCStream, SCShareableContent, SCStreamConfiguration,
            SCContentFilter, SCStreamOutputTypeAudio
        )
        import CoreMedia
        import threading

        self.is_running = True
        audio_queue_ref = self.audio_queue

        # Metadata for delegate: type is the 4th argument (self, _cmd, stream, sampleBuffer, type)
        objc.registerMetaDataForSelector(
            b"CaptureDelegate",
            b"stream:didOutputSampleBuffer:ofType:",
            {"arguments": {4: {"type": objc._C_NSInteger}}},
        )

        class CaptureDelegate(NSObject):
            def initWithQueue_(self, audio_queue):
                self = objc.super(CaptureDelegate, self).init()
                if self:
                    self.audio_queue = audio_queue
                    self.buffer = []
                    self.first_data = True
                return self

            def stream_didOutputSampleBuffer_ofType_(self, stream, sampleBuffer, outputType):
                if outputType == SCStreamOutputTypeAudio:
                    if self.first_data:
                        print("Successfully receiving audio data from ScreenCaptureKit.")
                        self.first_data = False
                    try:
                        block_buffer = CoreMedia.CMSampleBufferGetDataBuffer(sampleBuffer)
                        if not block_buffer:
                            return
                        
                        length = CoreMedia.CMBlockBufferGetDataLength(block_buffer)
                        if length <= 0:
                            return

                        data = bytearray(length)
                        CoreMedia.CMBlockBufferCopyDataBytes(block_buffer, 0, length, data)
                        
                        chunk = np.frombuffer(data, dtype=np.float32)
                        if len(chunk) > 0:
                            self.buffer.extend(chunk.tolist())
                            
                            while len(self.buffer) >= CHUNK_SIZE:
                                block = np.array(self.buffer[:CHUNK_SIZE], dtype=np.float32).reshape(-1, 1)
                                self.audio_queue.put(block)
                                del self.buffer[:CHUNK_SIZE]
                    except Exception:
                        pass

        def run_capture():
            def handle_content(content, error):
                if error:
                    print(f"Error getting content: {error}")
                    return

                if not content.displays():
                    print("No displays found for ScreenCaptureKit.")
                    return

                display = content.displays()[0]
                filter = SCContentFilter.alloc().initWithDisplay_excludingApplications_exceptingWindows_(
                    display, [], []
                )
                
                config = SCStreamConfiguration.alloc().init()
                config.setCapturesAudio_(True)
                config.setExcludesCurrentProcessAudio_(True)
                config.setSampleRate_(SAMPLERATE)
                config.setChannelCount_(1)

                # Initialize delegate with queue
                self.delegate = CaptureDelegate.alloc().initWithQueue_(audio_queue_ref)
                self.stream = SCStream.alloc().initWithFilter_configuration_delegate_(
                    filter, config, None
                )

                def start_handler(error):
                    if error: 
                        print(f"Error starting SCStream: {error}")
                    else: 
                        print("macOS ScreenCaptureKit started. Capturing system audio...")

                # Use a specific queue or None for default
                self.stream.addStreamOutput_type_sampleHandlerQueue_error_(self.delegate, SCStreamOutputTypeAudio, None, None)
                self.stream.startCaptureWithCompletionHandler_(start_handler)

            SCShareableContent.getShareableContentWithCompletionHandler_(handle_content)

            while self.is_running:
                NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.1))

        self.thread = threading.Thread(target=run_capture, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.stream:
            self.stream.stopCaptureWithCompletionHandler_(None)


class SpeechTranslator:
    def __init__(self, from_lang="en", to_lang="ru", device_id=None, loopback=False, loopback_device_index=None, engine="vosk", overlay_window=None):
        self.from_lang = _normalize_lang(from_lang) or from_lang
        self.to_lang = _normalize_lang(to_lang) or to_lang
        self.device_id = device_id
        self.loopback = loopback
        self.loopback_device_index = loopback_device_index
        self.engine = engine
        self.overlay_window = overlay_window

        base_dir = get_base_path()
        models_dir = os.path.join(base_dir, "models")

        if self.engine == ENGINE_VOSK:
            print(f"Loading Vosk model for {self.from_lang}...")
            model_name = VOSK_MODELS.get(self.from_lang) or VOSK_MODELS.get(self.from_lang[:2])
            if not model_name:
                raise ValueError(f"No Vosk model for '{self.from_lang}'. Supported: {', '.join(VOSK_MODELS.keys())}.")
            
            model_path = os.path.join(models_dir, model_name)
            if not os.path.exists(model_path):
                print(f"Model not found at {model_path}. Downloading...")
                import setup_models
                setup_models.download_vosk_model(model_name, models_dir)
            
            self.vosk_model = Model(model_path)
            self.recognizer = KaldiRecognizer(self.vosk_model, SAMPLERATE)
        else:
            try:
                import whisper
            except ImportError:
                print("\nERROR: Whisper not installed. Run: pip install openai-whisper\n")
                sys.exit(1)
            
            model_size = "base" if self.engine == ENGINE_WHISPER_LITE else "medium"
            # Adjust cache dir for whisper to be portable too if possible
            # Whisper uses ~/.cache/whisper by default. 
            # For now we leave it, but we could set XDG_CACHE_HOME.
            print(f"Loading Whisper model ({model_size})...")
            self.whisper_model = whisper.load_model(model_size)
            self.whisper_buffer = []

        print("Loading Silero VAD...")
        # For Silero VAD, we can also try to cache it locally
        torch_cache_dir = os.path.join(base_dir, ".cache", "torch")
        os.makedirs(torch_cache_dir, exist_ok=True)
        os.environ['TORCH_HOME'] = torch_cache_dir
        
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
        self.last_text_length = 0

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

    def _clear_line(self):
        """Clear the current line(s) taking wrapping into account."""
        if self.last_text_length > 0:
            columns, _ = shutil.get_terminal_size()
            lines = (self.last_text_length + columns - 1) // columns
            
            # Clear current line
            sys.stdout.write("\r\033[K")
            # Move up and clear previous lines if wrapped
            for _ in range(lines - 1):
                sys.stdout.write("\033[F\033[K")
            
            sys.stdout.flush()
            self.last_text_length = 0

    def process_loop(self):
        print("\n--- Listening (Press Ctrl+C to stop) ---\n")
        self.is_running = True

        while self.is_running:
            try:
                chunk = self.audio_queue.get(timeout=1.0)
                audio_float32 = chunk.flatten().astype(np.float32)

                with torch.no_grad():
                    confidence = self.vad_model(torch.from_numpy(audio_float32), SAMPLERATE).item()

                if confidence > VAD_THRESHOLD:
                    if not self.speech_detected:
                        self.speech_detected = True
                    self.silence_count = 0
                    if self.engine != ENGINE_VOSK:
                        self.whisper_buffer.append(audio_float32)
                else:
                    if self.speech_detected:
                        self.silence_count += 1
                        if self.engine != ENGINE_VOSK:
                            self.whisper_buffer.append(audio_float32)

                        if self.silence_count > SILENCE_CHUNKS:
                            self.speech_detected = False
                            self.silence_count = 0
                            
                            text = ""
                            if self.engine == ENGINE_VOSK:
                                final_res = json.loads(self.recognizer.FinalResult())
                                text = final_res.get("text", "")
                            else:
                                # Process accumulated buffer with Whisper
                                if self.whisper_buffer:
                                    audio_data = np.concatenate(self.whisper_buffer)
                                    self.whisper_buffer = []
                                    result = self.whisper_model.transcribe(audio_data, language=self.from_lang)
                                    text = result.get("text", "").strip()

                            if text:
                                translation = self.translate(text)
                                self._clear_line()
                                print(f"USER: {text}")
                                print(f"TRAN: {translation}")
                                print("-" * 30)
                                if self.overlay_window:
                                    self.overlay_window.text_updated.emit(text, "user")
                                    self.overlay_window.text_updated.emit(translation, "tran")

                # Real-time partial results only for Vosk
                if self.engine == ENGINE_VOSK:
                    audio_int16 = (chunk * 32768).astype(np.int16)
                    if self.recognizer.AcceptWaveform(audio_int16.tobytes()):
                        res = json.loads(self.recognizer.Result())
                        text = res.get("text", "")
                        if text and not self.speech_detected:
                            translation = self.translate(text)
                            self._clear_line()
                            print(f"USER: {text}")
                            print(f"TRAN: {translation}")
                            print("-" * 30)
                            if self.overlay_window:
                                self.overlay_window.text_updated.emit(text, "user")
                                self.overlay_window.text_updated.emit(translation, "tran")
                    else:
                        partial = json.loads(self.recognizer.PartialResult())
                        partial_text = partial.get("partial", "")
                        if partial_text:
                            # Instead of print with \r, we need more control
                            display_text = f">> {partial_text}"
                            self._clear_line()
                            sys.stdout.write(display_text)
                            sys.stdout.flush()
                            self.last_text_length = len(display_text)
                            if self.overlay_window:
                                self.overlay_window.text_updated.emit(partial_text, "partial")

            except queue.Empty:
                continue
            except KeyboardInterrupt:
                self._clear_line()
                break
            except Exception as e:
                self._clear_line()
                print(f"\nError in loop: {e}")

    def start(self):
        try:
            if self.loopback:
                sys_plat = platform.system()
                if sys_plat == "Windows":
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
                elif sys_plat == "Linux":
                    monitor_idx = self.loopback_device_index
                    if monitor_idx is None:
                        for i, d in enumerate(sd.query_devices()):
                            if d['max_input_channels'] > 0 and 'monitor' in d['name'].lower():
                                monitor_idx = i
                                break
                    if monitor_idx is None:
                        print("No monitor device found. Ensure PulseAudio/PipeWire is used.", file=sys.stderr)
                        return
                    device_info = sd.query_devices(monitor_idx)
                    print(f"Capturing from: {device_info['name']} (monitor)")
                    with sd.InputStream(device=monitor_idx, samplerate=SAMPLERATE, channels=1, callback=self.audio_callback, blocksize=CHUNK_SIZE):
                        self.process_loop()
                elif sys_plat == "Darwin":
                    capture = MacLoopbackCapture(self.audio_queue)
                    capture.start()
                    try:
                        self.process_loop()
                    finally:
                        capture.stop()
                else:
                    print(f"--loopback is not supported on {sys_plat}", file=sys.stderr)
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
    parser.add_argument("--engine", choices=[ENGINE_VOSK, ENGINE_WHISPER, ENGINE_WHISPER_LITE], default=ENGINE_VOSK, help="Speech recognition engine")
    parser.add_argument("--overlay", action="store_true", help="Show transparent overlay window with translation")
    
    args = parser.parse_args()
    
    from_lang, to_lang = args.from_lang, args.to_lang
    device_id, loopback = args.device, args.loopback
    loopback_device_index = args.loopback_device
    engine = args.engine
    show_overlay = args.overlay

    if not args.no_menu and device_id is None and not args.loopback and loopback_device_index is None and sys.stdin.isatty():
        from_lang, to_lang, device_id, loopback, loopback_device_index, engine, show_overlay = _interactive_menu(from_lang, to_lang)
    else:
        if loopback_device_index is not None:
            loopback = True

    overlay_window = None
    if show_overlay:
        try:
            from overlay import run_overlay_app
            app, overlay_window = run_overlay_app()
        except ImportError:
            print("\nERROR: PyQt6 not installed. Overlay disabled. Run: pip install PyQt6\n")
            show_overlay = False

    translator = SpeechTranslator(from_lang, to_lang, device_id, loopback, loopback_device_index, engine, overlay_window=overlay_window)
    
    if show_overlay:
        # Run translator in a background thread
        t = threading.Thread(target=translator.start, daemon=True)
        t.start()
        print("Overlay started. Close the overlay window or press Ctrl+C in terminal to stop.")
        try:
            sys.exit(app.exec())
        except KeyboardInterrupt:
            pass
    else:
        translator.start()
