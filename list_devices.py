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
    except Exception as e:
        print(f"\nCould not list devices: {e}")

if __name__ == "__main__":
    list_devices()
