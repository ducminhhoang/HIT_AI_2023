import uvicorn
import sys
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Form #import class FastAPI() từ thư viện fastapi
from pydantic import BaseModel #thu vien giup tu tao doi tuong (neu can)
from fastapi.responses import FileResponse #

from fastapi.middleware.cors import CORSMiddleware #thu vien de cho phep duong nguon khac truy cap vao server
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

# import NST #thu vien xu NST tra ve anh generate
import subprocess
from PIL import Image
import base64
from io import BytesIO
import random

import os

app = FastAPI() # gọi constructor và gán vào biến app
security = HTTPBasic()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Cho phép tất cả các nguồn gốc
    allow_credentials=True,
    allow_methods=["*"], # Cho phép tất cả các phương thức
    allow_headers=["*"], # Cho phép tất cả các tiêu đề
)

IMAGEDIR = "imgs/" #tao thu muc images để lưu ảnh lấy được và ảnh generate
styles_dir = "styles/"
content_dir = "content_user/"
generate_dir = "generate/"


def removeDir(dir: str):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        os.remove(file_path)


@app.get("/getImage")
async def hello():
    images = []
    for filename in os.listdir(os.path.join("imgs", "styles")):
            with Image.open(os.path.join("imgs", "styles", filename)) as img:
                buffered = BytesIO()
                img = img.convert("RGB")
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                filename = " ".join(filename.split("_"))
                filename = filename[:-4]
                images.append({"name": filename, "data": img_str})
    return {"images": images}


@app.post("/upload")
async def getImage(fileContent: UploadFile = File(...), fileStyle: str = Form(None)):
    removeDir(os.path.join("imgs", "generate"))
    removeDir(os.path.join("imgs", "content_user"))
    fileContent.filename = "tmp.jpg"
    content = await fileContent.read()
    with open(os.path.join("imgs", "content_user", fileContent.filename), "wb") as f:
        f.write(content)

    #code xử lý generate ảnh
    filepath = ""
    if fileStyle is None:
        filepath = [f for f in os.listdir(os.path.join("model")) if f.endswith(".pth")]
        r = random.randint(0, len(filepath))
        filepath = filepath[r]
    else:
        filepath = "_".join(fileStyle.split(" ")) + ".pth"

    code = f"{sys.executable} test_main.py  --model_load_path model/{filepath} --test_content imgs/content_user/tmp.jpg --imsize 256 --output imgs/generate/tmp.jpg".split()
    process = subprocess.Popen(code, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(stdout.decode("utf-8"))

    path = os.path.join("imgs", "generate", "tmp.jpg")
    print(path)
    return FileResponse(path)


if __name__ == "__apii__":
    uvicorn.run("apii:app", host="0.0.0.0", port=1000, reload=True)
# uvicorn apii:app --host 0.0.0.0 --port 8000 --reload
# http://127.0.0.1:8000/

