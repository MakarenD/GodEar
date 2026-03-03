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

if __name__ == "__main__":
    download_vosk_model("vosk-model-small-en-us-0.15")
    download_vosk_model("vosk-model-small-ru-0.22")
    download_vosk_model("vosk-model-small-de-0.15")
    print("Translation uses deep-translator (Google Translate API) - no model download needed.")
