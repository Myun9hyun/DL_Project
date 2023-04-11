import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb
import seaborn as sns
from streamlit_option_menu import option_menu
import base64
import torch
import torchvision

def download_file_from_google_drive(id, destination):
    URL = f"https://drive.google.com/u/0/uc?id={id}&export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# 모델 다운로드
file_id = '1kLo4A1qbyn1D2aMRwkpLPp1ehHe1eVz3'  
destination = 'MH/model/vgg_weights_1000.pth' # 변경된 부분
download_file_from_google_drive(file_id, destination)

모델 불러오기
model = torch.load(destination)
# 스트림릿 앱 구현
st.title("딥러닝 모델 구현")
# 이미지 업로드
uploaded_file = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='업로드한 이미지', use_column_width=True)
    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    input_tensor = transform(image).unsqueeze(0)
    # 모델 예측
    with torch.no_grad():
        model.eval()
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
    st.write("예측 결과:", prediction)
