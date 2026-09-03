import urllib.request
import zipfile
import io
import os

def download_cmu_fonts():
    os.makedirs("pdf_processor/fonts", exist_ok=True)
    
    # Download cm-unicode zip from CTAN mirror
    url = "https://mirrors.ctan.org/fonts/cm-unicode.zip"
    print(f"Downloading {url}...")
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        zip_bytes = resp.read()
        
    print(f"Downloaded zip archive ({len(zip_bytes)} bytes)")
    
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if name.endswith(".ttf") and "truetype" in name:
                filename = os.path.basename(name)
                target_path = os.path.join("pdf_processor/fonts", filename)
                with open(target_path, "wb") as f:
                    f.write(zf.read(name))
                print(f"Extracted {filename} ({os.path.getsize(target_path)} bytes)")

if __name__ == "__main__":
    download_cmu_fonts()
