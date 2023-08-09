import streamlit as st
from PIL import Image
import os
import subprocess
import sys

IMAGEDIR = "imgs/"  # tao thu muc images để lưu ảnh lấy được và ảnh generate
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


st.set_page_config(page_title='Style Transfer',page_icon="📷", layout="wide")

st.title("Style Image Transfer")
st.text("""Give us your pic, we'll make it more interesting 📷""")
check = [False, False]
col1, col2= st.columns(2)
content_file = ''
with col1:
    st.title("Content")
    content_file = st.file_uploader("Upload ảnh của bạn", type=["jpg", "jpeg", "png"])
    if content_file is not None:
        check[0] = True
        # Hiển thị ảnh nội dung
        content_image = Image.open(content_file)
        st.image(content_image, caption="Content Image", use_column_width=True)


filepath = ''
with col2:
    st.title("Chọn phong cách yêu thích")
    # Cho phép người dùng chọn kiểu ảnh từ danh sách các kiểu ảnh có sẵn
    listpath = [f for f in os.listdir(os.path.join("model")) if f.endswith(".pth")]
    style_options = [(" ".join(f.split("_")))[:-4] for f in listpath]
    style_name = st.selectbox("Chọn style bạn muốn", style_options)
    chose = st.selectbox("Chọn kiểu hiển thị", ["Origin Style", "Sample"])
    if style_name:
        check[1] = True
        style_img = "_".join(style_name.split(" ")) + ".jpg"
        filepath = "_".join(style_name.split(" ")) + ".pth"
        if chose == 'Origin Style':
            # Hiển thị ảnh kiểu
            style_image = Image.open(os.path.join("imgs", "styles", style_img[:-7], style_img))
            st.image(style_image, caption="Style Image", use_column_width=True)
            listpath = listpath[style_options.index(style_name)]
        else:
            # Hiển thị mẫu
            content_sample = Image.open(os.path.join("imgs", "sample", "content.jpg"))
            generate_sample = Image.open(os.path.join("imgs", "sample", style_img))
            st.image(content_image, caption="Content Sample", use_column_width=True)
            st.image(generate_sample, caption="Sample", use_column_width=True)


if False not in check:
    st.title("Result")
    content_image.save(os.path.join("imgs", "content_user", "tmp.jpg"))
    if st.button("Chạy"):
        code = f"python test_main.py  --model_load_path model\\{filepath} --test_content imgs\\content_user\\tmp.jpg --imsize 256 --output imgs\\generate\\tmp.jpg".split()
        st.text(code)
        process = subprocess.run(code, stdout=subprocess.PIPE)
        st.text(process.stdout.decode("utf-8"))
        result_image = Image.open(os.path.join("imgs", "generate", "tmp.jpg"))
        st.image(result_image, caption="Result Image", use_column_width=True)

        with open(os.path.join("imgs", "generate", "tmp.jpg"), 'rb') as f:
            st.download_button(
                label="Tải ảnh",
                data=f,
                file_name='result.jpg',
                mime="image/jpg"
            )
        st.success("Hoàn thành!")
        removeDir(os.path.join(IMAGEDIR, generate_dir))
        removeDir(os.path.join(IMAGEDIR, "content_user"))
