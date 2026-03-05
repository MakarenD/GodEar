import os
import sys
import subprocess
import shutil
import platform
import importlib.util

try:
    import importlib.metadata as metadata
except ImportError:
    import importlib_metadata as metadata # fallback for older python

def build():
    # 0. Check Python version (Enforce 3.9)
    v = sys.version_info
    if not (v.major == 3 and v.minor == 9):
        print("\n" + "!"*50)
        print(f"ERROR: Incorrect Python version (Found {v.major}.{v.minor}.{v.micro})")
        print("This project is optimized for Python 3.9.6.")
        print("Please activate 'venv39' before building.")
        print("!"*50 + "\n")
        sys.exit(1)

    print(f"Starting build for {platform.system()} (Python {v.major}.{v.minor})...")
    
    # 1. Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    release_dir = os.path.join(base_dir, "releases")
    dist_dir = os.path.join(base_dir, "dist")
    build_temp_dir = os.path.join(base_dir, "build")
    app_name = "speech-to-text"
    
    # Clean up redundant 'venv' if it exists accidentally
    bad_venv = os.path.join(base_dir, "venv")
    if os.path.exists(bad_venv):
        print(f"Removing redundant 'venv' folder...")
        shutil.rmtree(bad_venv)
    
    # 2. Check if PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("\n" + "!"*30)
        print("ERROR: PyInstaller not found!")
        print("Please activate your virtual environment and install it:")
        print("  source venv/bin/activate  # on Mac/Linux")
        print("  pip install pyinstaller")
        print("!"*30 + "\n")
        sys.exit(1)

    # 3. Clean up old build folders
    folders_to_clean = [release_dir, dist_dir, build_temp_dir]
    for folder in folders_to_clean:
        if os.path.exists(folder):
            print(f"Cleaning up {folder}...")
            shutil.rmtree(folder)
    
    os.makedirs(release_dir, exist_ok=True)

    # 4. Prepare metadata flags (only for installed distributions)
    # We check the actual distribution name (what you see in 'pip list')
    optional_distributions = [
        "torch", "tqdm", "regex", "requests", "packaging", 
        "filelock", "numpy", "tokenizers", "openai-whisper", "vosk", "PyQt6", "pynput",
        "ctranslate2", "transformers", "huggingface-hub", "sentencepiece"
    ]
    
    metadata_flags = []
    for dist_name in optional_distributions:
        try:
            metadata.distribution(dist_name)
            metadata_flags.extend(["--copy-metadata", dist_name])
            print(f"  Found metadata for: {dist_name}")
        except metadata.PackageNotFoundError:
            continue

    # 5. Define PyInstaller command
    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--onedir",
        "--console",
        "--name", app_name,
        "--distpath", release_dir,
        "--hidden-import", "pyaudiowpatch",
        "--hidden-import", "pyobjc",
        "--hidden-import", "torch",
        "--hidden-import", "torchaudio",
        "--hidden-import", "PyQt6",
        "--hidden-import", "pynput.keyboard._darwin",
        "--hidden-import", "pynput.keyboard._win32",
        "--hidden-import", "pynput.keyboard._xorg",
        "--hidden-import", "pynput.mouse._darwin",
        "--hidden-import", "pynput.mouse._win32",
        "--hidden-import", "pynput.mouse._xorg",
        "--hidden-import", "ctranslate2",
        "--hidden-import", "transformers",
        "--hidden-import", "huggingface_hub",
        "--hidden-import", "sentencepiece",
        "--collect-all", "vosk",
        "--collect-all", "whisper",
        "--collect-all", "deep_translator",
        "--collect-all", "torch",
        "--collect-all", "torchaudio",
        "--collect-all", "PyQt6",
        "--collect-all", "pynput",
        "--collect-all", "ctranslate2",
        "--collect-all", "transformers",
        "--collect-all", "huggingface_hub",
        "--collect-all", "sentencepiece"
    ] + metadata_flags + ["main.py"]

    print("\nRunning PyInstaller (this will take 5-10 minutes because of torch/whisper)...")
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error: {e}")
        return

    # 6. Add extra files to the release folder
    app_folder = os.path.join(release_dir, app_name)
    extra_files = ["README.md", "requirements.txt"]
    
    print("\nAdding extra files to the app folder...")
    for f in extra_files:
        src = os.path.join(base_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, app_folder)
            print(f"  Copied {f}")

    # 7. Create models directory inside
    os.makedirs(os.path.join(app_folder, "models"), exist_ok=True)

    # 8. Create final ZIP archive in the releases/ folder
    os_name = platform.system().lower()
    if os_name == "darwin": os_name = "macos"
    output_zip_name = f"speech-to-text-{os_name}"
    output_zip_path = os.path.join(release_dir, output_zip_name)
    
    print(f"\nCreating ZIP archive in releases folder...")
    shutil.make_archive(output_zip_path, 'zip', app_folder)

    # 9. Final Clean up of temp files
    print("Cleaning up temporary build files...")
    if os.path.exists(dist_dir): shutil.rmtree(dist_dir)
    if os.path.exists(build_temp_dir): shutil.rmtree(build_temp_dir)
    spec_file = os.path.join(base_dir, f"{app_name}.spec")
    if os.path.exists(spec_file): os.remove(spec_file)

    print("\n" + "="*50)
    print(f"BUILD COMPLETE!")
    print(f"1. Ready-to-use folder: {app_folder}")
    print(f"2. Final ZIP for GitHub: {output_zip_path}.zip")
    print("="*50)

if __name__ == "__main__":
    build()
