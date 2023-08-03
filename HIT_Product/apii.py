from typing import Union
import uvicorn

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Form #import class FastAPI() từ thư viện fastapi
from pydantic import BaseModel #thu vien giup tu tao doi tuong (neu can)
from fastapi.responses import FileResponse #

from fastapi.middleware.cors import CORSMiddleware #thu vien de cho phep duong nguon khac truy cap vao server
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

# import NST #thu vien xu NST tra ve anh generate
from zipfile import ZipFile
import subprocess
from PIL import Image
import base64
from io import BytesIO

import os
import uuid #giup tao ten ngau nhien 16 ky tu

app = FastAPI() # gọi constructor và gán vào biến app
security = HTTPBasic()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Cho phép tất cả các nguồn gốc
    allow_credentials=True,
    allow_methods=["*"], # Cho phép tất cả các phương thức
    allow_headers=["*"], # Cho phép tất cả các tiêu đề
)

@app.get("/dev")
def read_root(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, "nstadmin")
    correct_password = secrets.compare_digest(credentials.password, "123456")
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=400,
            detail="Incorrect usename or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return {"message": ""}



IMAGEDIR = "imgs/" #tao thu muc images để lưu ảnh lấy được và ảnh generate
styles_dir = "styles/"
content_dir = "content_user/"
generate_dir = "generate/"
dictStyle_Model = {"style1.jpg" : "William_Turner_13.pth", "style2.jpg" : "Frida_Kahlo.pth"}

def saveFile(content, file: UploadFile = File(...)):
    with open(f"{IMAGEDIR}{content_dir}{file.filename}", "wb") as f:
        f.write(content)

@app.post("/trainning")
async def setTrain(style: UploadFile = File(...), max_iter: int = Form(1000)):

    return {'massage': ''}

@app.get("/getNameStyle")
async def getName():
    images = os.listdir("model")
    print({"images": images})
    return {"images": images}

@app.get("/getImage")
async def hello(str: str):
    images = []
    for filename in os.listdir("imgs\\"):
        if filename.endswith(".jpg"):
            with Image.open(f"imgs/{filename}") as img:
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                images.append({"name": filename, "data": img_str})
    return {"images": images}

# @app.post("/train")
# async def setPath(fileStyle : UploadFile = File(...)):


@app.post("/upload")
async def getImage(fileContent: UploadFile = File(...), fileStyle: str = Form(...)):
    # mỗi lần chạy lại sẽ xóa file ảnh cũ đi
    print(fileStyle)
    for filename in os.listdir(f"{IMAGEDIR}{generate_dir}"):
       file_path = os.path.join(IMAGEDIR, generate_dir, filename)

       os.remove(file_path)


    fileName_rand = uuid.uuid4()
    fileName = fileContent.filename
    fileContent.filename = f"{fileName_rand}.jpg"
    content = await fileContent.read()

    saveFile(content, fileContent)

    #code xử lý generate ảnh

    if (fileStyle in dictStyle_Model.keys()):
        model_path = dictStyle_Model[fileStyle]
        print(model_path)
        code = f"python test_main.py  --model_load_path model\{model_path} --test_content imgs\content_user\{fileContent.filename} --imsize 256 --output imgs\generate\{fileName_rand}.jpg".split()
        print(code)
        process = subprocess.Popen(code, stdout=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(stdout.decode("utf-8"))

    else:
        raise HTTPException(
            status_code=400,
            detail="Not Found style in Database",
            headers={"WWW-Authenticate": "Basic"},
        )
    path = f"imgs\\generate\\{fileName_rand}.jpg"
    print(path)
    return FileResponse(path)



if __name__ == "__apii__":
    uvicorn.run("apii:app", host="0.0.0.0", port=8000, reload=True)
# uvicorn apii:app --host 0.0.0.0 --port 8000 --reload
# http://127.0.0.1:8000/

