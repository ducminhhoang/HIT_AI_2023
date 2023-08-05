import streamlit as st
from PIL import Image
import os
import subprocess
import uuid


IMAGEDIR = "imgs/" #tao thu muc images để lưu ảnh lấy được và ảnh generate
styles_dir = "styles/"
content_dir = "content_user/"
generate_dir = "generate/"

def saveFile(content, filename):
    with open(f"{IMAGEDIR}{content_dir}{filename}", "wb") as f:
        f.write(content)


def removeDir(dir: str):
    for filename in os.listdir(dir):
       file_path = os.path.join(dir, filename)

       os.remove(file_path)

st.set_page_config(page_title='Hello',page_icon="📷", layout="wide")



st.title("Style Transfer")

content_file = st.file_uploader("Upload Content Image")
if content_file is not None:
    # Hiển thị ảnh nội dung
    content_image = Image.open(content_file)
    st.image(content_image, caption="Content Image", use_column_width=True)
    removeDir(os.path.join("HIT_Product", IMAGEDIR, generate_dir))
    removeDir(os.path.join("HIT_Product", IMAGEDIR, "content_user"))


    # Cho phép người dùng chọn kiểu ảnh từ danh sách các kiểu ảnh có sẵn
    style_options = [f for f in os.listdir(os.path.join("HIT_Product", "model")) if f.endswith(".pth")]
    style_name = st.selectbox("Select Style", style_options)

    # Nếu người dùng đã chọn kiểu ảnh
    if style_name:
        # Hiển thị ảnh kiểu
        style_img = style_name[:style_name.find(".pth")] + ".jpg"
        style_image = Image.open(os.path.join("HIT_Product", "imgs", "styles", style_img))
        st.image(style_image, caption="Style Image", use_column_width=True)

        fileName_rand = uuid.uuid4()
        filename = f"{fileName_rand}.jpg"
        content_image.save(os.path.join("HIT_Product", "imgs", "content_user", filename))

        code = f"{sys.executable} test_main.py  --model_load_path HIT_Product/model/{style_name} --test_content HIT_Product/imgs/content_user/{filename} --imsize 256 --output HIT_Product/imgs/generate/{filename}".split()
        process = subprocess.run(code, stdout=subprocess.PIPE)
        # print(process.decode("utf-8"))
        st.text(process.stdout.decode("utf-8"))

        result_image = Image.open(os.path.join("HIT_Product", "imgs", "generate", filename))
        st.image(result_image, caption="Result Image", use_column_width=True)
