
import argostranslate.package
import argostranslate.translate
import os
import requests
import zipfile
from tqdm import tqdm

def download_vosk_model(model_name="vosk-model-small-en-us-0.15"):
    url = f"https://alphacephei.com/vosk/models/{model_name}.zip"
    dest = f"models/{model_name}"
    if os.path.exists(dest):
        print(f"Vosk model {model_name} already exists.")
        return dest
    
    os.makedirs("models", exist_ok=True)
    zip_path = f"models/{model_name}.zip"
    
    print(f"Downloading Vosk model from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_path, "wb") as f, tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
        for data in response.iter_content(chunk_size=4096):
            f.write(data)
            pbar.update(len(data))
            
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("models")
    
    os.remove(zip_path)
    return dest

def install_argos_models(from_code="en", to_code="ru"):
    print(f"Installing Argos Translate models for {from_code} -> {to_code}...")
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    
    # English to Russian
    package_to_install = next(
        filter(
            lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
        )
    )
    argostranslate.package.install_from_path(package_to_install.download())
    
    # Russian to English (optional but good to have)
    package_to_install = next(
        filter(
            lambda x: x.from_code == to_code and x.to_code == from_code, available_packages
        )
    )
    argostranslate.package.install_from_path(package_to_install.download())

if __name__ == "__main__":
    download_vosk_model("vosk-model-small-en-us-0.15")
    download_vosk_model("vosk-model-small-ru-0.22")
    install_argos_models("en", "ru")
