import streamlit as st
import pandas as pd
import numpy as np
# import plotly_express as px
from PIL import Image
import streamlit.components.v1 as components
# import matplotlib.pyplot as plt
from tensorflow import keras
import joblib
import operator
import sys

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.mobilenet import decode_predictions

from PIL import Image
sys.modules['Image'] = Image 
# [theme]
base="light"
primaryColor="purple"

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

model = keras.models.load_model('fcvmodel.h5')
dbfood = pd.read_csv('dbfood.csv',sep=";")
food = dbfood['nama'].tolist()

def getPrediction(data,model):
    img = Image.open(data)
    newsize = (224, 224)
    image = img.resize(newsize)
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = yhat[0]
    prob = []
    for i in range(len(label)):
        # prob.append(i)
        prob.append(np.round(label[i]*100,2))
    data = {'nama': food, 'prob': prob}
    dfhasil = pd.DataFrame.from_dict(data)
    top3 = dfhasil.nlargest(3, 'prob')
    # top = dict(zip(food, prob))
    # top3 = dict(sorted(top.items(), key=operator.itemgetter(1), reverse=True)[:3])
    return top3

# st.set_page_config(layout='wide')

def main():
    st.subheader("Heal - Food Analyzer")
    with st.expander('Open Camera'):
        data1 = st.camera_input('')
    with st.expander('Upload A Photo'):
        data2 = st.file_uploader('')

    if data1 != None:
        data = data1
    elif data2 != None:
        data = data2
    else:
        data = None

    if data == None:
        st.write('Please Upload Photo of Food')
    else:
        img = Image.open(data)
        newsize = (280, 230)
        image = img.resize(newsize)
        st.image(image)

#     if st.button('Jalankan Prediksi'):
        hasil = getPrediction(data,model)
        # dfhasil = pd.DataFrame.from_dict(hasil)
        # keys = list(hasil.keys())
        top = hasil.nlargest(1, 'prob')
        # dbkal = dbfood[dbfood['nama'].isin(keys)]
        dfk = pd.merge(hasil,dbfood,how='left',on='nama')
        dfk['Protein'] = dfk['protein']*dfk['prob']/100
        dfk['Lemak'] = dfk['lemak']*dfk['prob']/100
        dfk['Karbohidrat'] = dfk['karbohidrat']*dfk['prob']/100
        dfk['Kkal'] = dfk['kkal']*dfk['prob']/100
        dfk['Score'] = dfk['skor']*dfk['prob']/100
        tingkat = dfk['Score'].tolist()
        total= tingkat[0]+tingkat[1]+tingkat[2]
        risiko = None
        if total >450:
            risiko = 'High Risk to Consume'
        elif total >250:
            risiko = 'Medium Risk to Consume'
        elif total >105:
            risiko = 'Low Risk to Consume'
        else:
            risiko = 'Safe to Consume'
        top1 = top['nama'].tolist()
        st.subheader(top1[0])
#         st.write(f"Confidence: {top['prop'].tolist()[0]}")
#         out = '''<h3>f'{str(top1[0]}'<h3>'''
#         st.markdown(f'{str(top1[0])}', unsafe_allow_html=True)
        st.write(f'Risk for who have Diabetes/Heart Disease: {risiko}')
        # st.write(dfk)
        a = dfk['Kkal'].sum()
        b = dfk['Lemak'].sum()
        c = dfk['Karbohidrat'].sum()
        d = dfk['Protein'].sum()
        st.write(f'Calorie: {np.round(a)} Kkal')
        st.write(f'Fat: {np.round(b)} gr')
        st.write(f'Carbohydrate: {np.round(c)} gr')
        st.write(f'Protein: {np.round(d)} gr')

        

if __name__=='__main__':
    main()
