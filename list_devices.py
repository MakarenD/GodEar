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
        
        # On Linux, show monitor devices
        elif sys.platform.startswith("linux"):
            print("\n--- Loopback (Monitor Devices) ---")
            found = False
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0 and 'monitor' in device['name'].lower():
                    print(f"  ID {i}: {device['name']}")
                    found = True
            if found:
                print("  -> Use: python main.py <from> <to> --loopback")
            else:
                print("  No monitor devices found. Ensure PulseAudio or PipeWire is running.")

        # On macOS, mention ScreenCaptureKit
        elif sys.platform == "darwin":
            print("\n--- Loopback (System Audio) ---")
            print("  macOS uses ScreenCaptureKit for system audio capture.")
            print("  -> Use: python main.py <from> <to> --loopback")
            print("  Note: Requires 'Screen Recording' permission for the terminal.")
    except Exception as e:
        print(f"\nCould not list devices: {e}")

if __name__ == "__main__":
    list_devices()
