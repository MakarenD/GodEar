import sys
import sounddevice as sd

def list_devices():
    print("\n--- Available Audio Input Devices ---")
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"ID {i}: {device['name']} (Channels: {device['max_input_channels']})")
        
        default_device = sd.default.device[0]
        print(f"\nDefault Input Device ID: {default_device}")

        # On Windows, show WASAPI loopback devices (speakers / playback capture)
        if sys.platform == "win32":
            try:
                import pyaudiowpatch as pyaudio
                p = pyaudio.PyAudio()
                print("\n--- Loopback (Speakers / Discord / System Audio) ---")
                for lb in p.get_loopback_device_info_generator():
                    print(f"  {lb['name']} (index: {lb['index']})")
                print("  -> Use: python main.py <from> <to> --loopback (or select from menu)")
                p.terminate()
            except ImportError:
                print("\n--- Loopback: Install PyAudioWPatch to see speakers ---")
            except Exception:
                pass
    except Exception as e:
        print(f"\nCould not list devices: {e}")

if __name__ == "__main__":
    list_devices()
