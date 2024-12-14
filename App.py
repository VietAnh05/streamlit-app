import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Tắt log cảnh báo không cần thiết
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Đảm bảo chỉ chạy trên CPU
import streamlit as st
# Đảm bảo các thông số cần thiết của Streamlit được khởi tạo
st.set_page_config(
    page_title="Phát hiện bệnh lá cây",
    layout="wide",
    initial_sidebar_state="expanded",
)
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D

# Định nghĩa lớp DepthwiseConv2D tùy chỉnh để loại bỏ tham số 'groups'
def custom_depthwise_conv2d(*args, **kwargs):
    if 'groups' in kwargs:
        del kwargs['groups']  # Loại bỏ tham số 'groups'
    return DepthwiseConv2D(*args, **kwargs)

# Tải mô hình với custom_objects
model = tf.keras.models.load_model(
    "D:\Dự án AI\Model\keras_model.h5", 
    compile=False, 
    custom_objects={'DepthwiseConv2D': custom_depthwise_conv2d}
)

# Hàm lấy thuốc từ CSV dựa trên bệnh
def get_medicine_from_csv(disease):
    # Đọc tệp CSV vào DataFram
    try:
    # Thử đọc với mã hóa UTF-8 và dấu phân cách là ';'
        df = pd.read_csv("D:\Dự án AI\Bệnh ở cây cà phê.csv", encoding='utf-8', delimiter=';')
    except UnicodeDecodeError:
    # Nếu gặp lỗi mã hóa, thử với mã hóa 'latin-1' và dấu phân cách là ';'
        df = pd.read_csv("D:\Dự án AI\Bệnh ở cây cà phê.csv", encoding='latin-1', delimiter=';')

    # Tìm đề xuất thuốc
    row = df[df['Benh'].str.lower() == disease.lower()]
    if not row.empty:
        return row.iloc[0][['Thuoc dac tri', 'Phuong phap tu nhien']]  # Hoặc 'Thuoc dac tri' tùy theo yêu cầu
    return {"Thuoc dac tri": "Không có đề xuất thuốc phù hợp.", "Phuong phap tu nhien": "Không có phương pháp tự nhiên phù hợp."}

# Hàm xử lý ảnh và dự đoán
def predict_image(image):
    # Thay đổi kích thước ảnh về 224x224
    image = image.resize((224, 224))  # Thay đổi kích thước ảnh

    # Chuyển ảnh sang định dạng RGB (loại bỏ kênh alpha nếu có)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    img = img_to_array(image) / 255.0  # Chuẩn hóa ảnh
    img = np.expand_dims(img, axis=0)  # Thêm chiều mới cho mô hình
    predictions = model.predict(img)  # Dự đoán
    class_names = ["Benh gi la", "Benh dom mat cua", "Benh sau duc la", "Benh thoi qua ca phe", "Cay khoe manh"]
    predicted_class = class_names[np.argmax(predictions)]  # Lấy lớp dự đoán có xác suất cao nhất
    return predicted_class

# Streamlit UI
st.title("Phát hiện bệnh lá cây")

# Tùy chọn sử dụng Webcam hoặc tải lên ảnh
use_webcam = st.checkbox("Sử dụng Webcam", value=False)

if use_webcam:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

    # Lớp xử lý video
    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.disease = None
            self.medicine = None

        def recv(self, frame):
            # Chuyển đổi ảnh từ frame video
            img = frame.to_image()
            
            # Dự đoán bệnh
            self.disease = predict_image(img)
            
            # Lấy thuốc và phương pháp từ CSV
            self.medicine = get_medicine_from_csv(self.disease)

            return frame

    # Sử dụng webrtc_streamer
    ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    # Hiển thị kết quả dự đoán bệnh và thuốc
    if ctx and ctx.video_processor:
        processor = ctx.video_processor
        st.subheader("Kết quả dự đoán:")
        
        # Hiển thị kết quả
        if processor.disease:
            st.write(f"Bệnh phát hiện: {processor.disease}")
            st.write(f"Đề xuất thuốc: {processor.medicine.get('Thuoc dac tri', 'Không có')}")  # Thuốc đặc trị
            st.write(f"Phương pháp tự nhiên: {processor.medicine.get('Phuong phap tu nhien', 'Không có')}")  # Phương pháp tự nhiên
        else:
            st.write("Đang chờ dữ liệu từ webcam")

else:
    # Tải lên ảnh tĩnh
    uploaded_file = st.file_uploader("Tải lên ảnh lá cây", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

        # Dự đoán bệnh
        predicted_disease = predict_image(image)
        
        # Lấy đề xuất thuốc
        medicine = get_medicine_from_csv(predicted_disease)

        # Hiển thị kết quả
        st.subheader("Kết quả dự đoán:")
        st.write(f"Bệnh phát hiện: {predicted_disease}")
        st.write(f"Đề xuất thuốc: {medicine.get('Thuoc dac tri', 'Không có')}")  # Thuốc đặc trị
        st.write(f"Phương pháp tự nhiên: {medicine.get('Phuong phap tu nhien', 'Không có')}")  # Phương pháp tự nhiên
