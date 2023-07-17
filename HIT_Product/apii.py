from typing import Union

from fastapi import FastAPI, File, UploadFile #import class FastAPI() từ thư viện fastapi
from pydantic import BaseModel #thu vien giup tu tao doi tuong (neu can)
from fastapi.responses import FileResponse #

from fastapi.middleware.cors import CORSMiddleware #thu vien de cho phep duong nguon khac truy cap vao server

# import NST #thu vien xu NST tra ve anh generate

import os
import uuid #giup tao ten ngau nhien 16 ky tu

app = FastAPI() # gọi constructor và gán vào biến app

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Cho phép tất cả các nguồn gốc
    allow_credentials=True,
    allow_methods=["*"], # Cho phép tất cả các phương thức
    allow_headers=["*"], # Cho phép tất cả các tiêu đề
)

IMAGEDIR = "images/" #tao thu muc images để lưu ảnh lấy được và ảnh generate


def saveFile(content, file: UploadFile = File(...)):
    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(content)

@app.post("/upload")
async def getImage(fileContent: UploadFile = File(...), fileStyle: UploadFile = File(...)):
    # mỗi lần chạy lại sẽ xóa file ảnh cũ đi
    for filename in os.listdir(f"{IMAGEDIR}"):
        file_path = os.path.join(IMAGEDIR, filename)

        os.remove(file_path)

    fileName_rand = uuid.uuid4()
    # print(fileContent.read())
    fileContent.filename = f"{fileName_rand}Content.jpg"
    fileStyle.filename = f"{fileName_rand}Style.jpg"
    content = await fileContent.read()
    style = await fileStyle.read()

    saveFile(content, fileContent)
    saveFile(style, fileStyle)

    #code xử lý generate ảnh

    #tạm thời lấy ảnh Content
    path = f"{IMAGEDIR}{fileContent.filename}"
    print(path)
    return FileResponse(path)

# uvicorn apii:app --host 0.0.0.0 --port 8000 --reload
# http://127.0.0.1:8000/

