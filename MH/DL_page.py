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


st.set_page_config(page_title="Deep Learning Project", page_icon=":minidisc:", layout="wide")


# 이후 Streamlit 앱의 나머지 부분을 정의합니다.


# st.header("DL Project")
image_NJS = "MH/image/deep.jpg"
# st.image(image_NJS, use_column_width=True)

image = Image.open("MH/image/develop_jeans.jpg")
width, height = image.size
# 이미지에 텍스트 추가
draw = ImageDraw.Draw(image)
text_kor = "독산 개발진스"
text_eng = "Deep Learning"
font_kor = ImageFont.truetype("MH/font/NanumSquareNeo-eHv.ttf", 50)
font_eng = ImageFont.truetype("MH/font/ARIAL.TTF", 50)
text_width, text_height = draw.textsize(text_kor, font=font_kor)

stroke_width = 2
stroke_fill = (0, 0, 0)
# x = (width - text_width) // 2
# y = (height - text_height) // 2
x = (width - text_width) // 2
x1 = (width - text_width) // 2 - 10
y = height - text_height - 20
z = height - text_height - 100

# 이미지에 텍스트 추가
draw = ImageDraw.Draw(image)
# draw.text((x, y), text_kor, font=font_kor, fill=(0, 0, 0),outline=outline_color, width=outline_width)
# draw.text((x, z), text_eng, font=font_eng, fill=(0, 0, 0), outline=outline_color, width=outline_width)
draw.text((x - stroke_width, y), text_kor, font=font_kor, fill=stroke_fill, stroke_width=stroke_width)
draw.text((x + stroke_width, y), text_kor, font=font_kor, fill=stroke_fill, stroke_width=stroke_width)
draw.text((x, y - stroke_width), text_kor, font=font_kor, fill=stroke_fill, stroke_width=stroke_width)
draw.text((x, y + stroke_width), text_kor, font=font_kor, fill=stroke_fill, stroke_width=stroke_width)
draw.text((x, y), text_kor, font=font_kor, fill=(255, 255, 255))
draw.text((x1 - stroke_width, z), text_eng, font=font_eng, fill=stroke_fill, stroke_width=stroke_width)
draw.text((x1 + stroke_width, z), text_eng, font=font_eng, fill=stroke_fill, stroke_width=stroke_width)
draw.text((x1, z - stroke_width), text_eng, font=font_eng, fill=stroke_fill, stroke_width=stroke_width)
draw.text((x1, z + stroke_width), text_eng, font=font_eng, fill=stroke_fill, stroke_width=stroke_width)
draw.text((x1, z), text_eng, font=font_eng, fill=(255, 255, 255))
# streamlit에 이미지 표시
st.image(image, use_column_width=True)





with st.sidebar:
    choice = option_menu("Menu", ["홈페이지"],
                         icons=['house'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "4!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#fafafa"},
        "nav-link-selected": {"background-color": "#08c7b4"},
    }
    )
    # st.write("Link")
    data = {
        'Name': ['💾Team Repo', '💪Team Notion', '💿Data Source'],
        'Link': ['[![GitHub](https://img.shields.io/badge/Github-3152A0?style=for-the-badge&logo=Github&logoColor=white)](https://github.com/tkd8973/DL_Project)',
         '[![Notion](https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=notion&logoColor=white)](https://www.notion.so/DL_PROJECT-82b3fdfbde2e4937b0f9463fce66d056)',
         '[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/code/soumya044/artistic-neural-style-transfer-using-pytorch)']
    }
    df = pd.DataFrame(data)
    # st.sidebar.dataframe(df)
    st.write(df.to_markdown(index=False))
    col1, col2 = st.columns(2)
    # with col1:
    #     st.write("💾Team repo")
    #     st.markdown('<a href="https://github.com/tkd8973/DL_Project"><img src="https://img.shields.io/badge/Github-3152A0?style=for-the-badge&logo=Github&logoColor=white"></a>', unsafe_allow_html=True)
    # with col2:
    #     st.write("💪Team Notion")
    #     st.markdown('<a href="https://www.notion.so/DL_PROJECT-82b3fdfbde2e4937b0f9463fce66d056"><img src="https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=notion&logoColor=white"></a>', unsafe_allow_html=True)
    

    # st.write("💾Team repo")
    # st.markdown('<a href="https://github.com/tkd8973/DL_Project"><img src="https://img.shields.io/badge/Github-3152A0?style=for-the-badge&logo=Github&logoColor=white"></a>', unsafe_allow_html=True)
    # st.write("💪Team Notion")
    # st.markdown('<a href="https://www.notion.so/DL_PROJECT-82b3fdfbde2e4937b0f9463fce66d056"><img src="https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=notion&logoColor=white"></a>', unsafe_allow_html=True)

if choice == "홈페이지":
    
    tab0, tab1, tab2, tab3 = st.tabs(["🏠 Main", "🔎Explain", "📉Graph", "🔗Link"])
    image_molu = "MH/image/molu.gif"
    image_molu_ai = "MH/image/molu_ai.jpg"

        # Streamlit에서 GIF 보여주기  

    with tab0:
        st.write()
        '''
        **⬆️위의 탭에 있는 메뉴를 클릭해 선택하신 항목을 볼 수 있습니다!⬆️**
        '''
        col1, col2 = st.columns(2)
        with col1:
            st.write("**몰?루**")
            st.image(image_molu, use_column_width=True)
            st.write()
            '''
            ### Team 💪

            | 이름 | 역할 분담 | 그외 역할 | GitHub Profile |
            | :---: | :---: | :---: | :--- |
            | 서상원 | 데이터 모델링 | 딥러닝 모델링 구현 |[![GitHub](https://badgen.net/badge/icon/github%20tkd8973?icon=github&label)](https://github.com/tkd8973)|
            | 조성훈 | 데이터 전처리 | 딥러닝 모델링 구현 |[![GitHub](https://badgen.net/badge/icon/github%20chohoon901?icon=github&label)](https://github.com/chohoon901)|
            | 김명현 | 데이터 시각화 | Streamlit 구현 |[![GitHub](https://badgen.net/badge/icon/github%20Myun9hyun?icon=github&label)](https://github.com/Myun9hyun)|
            | 강성욱 | 데이터 소스 조사 | 딥러닝 모델링 구현 |[![GitHub](https://badgen.net/badge/icon/github%20JoySoon?icon=github&label)](https://github.com/JoySoon)|
            '''
        with col2:
            st.write("**몰?루 ai실사**")
            st.image(image_molu_ai, use_column_width=True)

    with tab1:
        tab1.subheader("🔎Explain tab")
        tab1.write()
        image_insert = "MH/image/sight.jpg"
        image_style = "MH/image/starrynight.jpg"
        image_convert = "MH/image/converted_small.jpg"

        '''
        ### 자료 설명
        > * 이미지를 두개를 선택합니다.
        > * 첫번째 이미지는 변환이 되는 이미지이고
        > * 두번째 이미지는 첫번째 이미지를 두번째 이미지의 느낌과 그림체로 변환을 시켜줍니다.
        > * 딥러닝을 거친 후 이미지가 출력이 됩니다.
        '''
        st.write("#### 이미지 변환 예시")
        col1, col2, col3 = st.columns(3)  
        with col1:
            st.write("**변환하고자 하는 이미지**")
            st.image(image_insert, use_column_width=True)
        with col2:
            st.write("**적용되는 style 이미지**")
            st.image(image_style, use_column_width=True)
        with col3:
            st.write("**변환되어 출력된 이미지**")
            st.image(image_convert, use_column_width=True)
        

        # 모델 로드
            # model = torch.load(destination, map_location=torch.device('cpu'))
    with tab2:
        tab2.subheader("📉Graph tab")
        image_graph = "MH/image/vgg19_graph.png"
        st.write()
        '''
        #### 다음의 과정을 거쳐 학습하는 모델입니다.
        '''
        st.image(image_graph, use_column_width=True)

    with tab3:
        tab3.subheader("🔗Link tab")
        st.write()
        '''
        #### Colab 링크
        [![Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/drive/1Vwi45lqrq8Cys2z-2idackArSV2PIY4H?usp=sharing)
        
        '''
        
elif choice == "페이지2":
    st.subheader("페이지2")


    