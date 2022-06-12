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

import requests
from recipe_scrapers import scrape_html, scrape_me
import re
from pygsearch import gsearch

# CSS to inject contained in a string
hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)

# Display a static table

def searchLink(foodname):
    search = gsearch(f"{foodname} site=allrecipes.com -allrecipes.com/recipes  -allrecipes.com/gallery",1)
    result1 = str(search.results[0])
    text = result1.split("link=",1)[1]
    link = text.split(">",1)[0]
    return link[1:-2]

# rd = searchLink('rendang')
# print(rd)

def scrapeRecipe(url):
    # url = f"https://www.allrecipes.com/recipe/72567/panna-cotta/"
    html = requests.get(url).content
    scraper = scrape_html(html=html, org_url=url)
    # title = scraper.title()
    # total_time=scraper.total_time()
    # yields= scraper.yields()
    ingredients =scraper.ingredients()
    ingredient = []
    for i in ingredients:
        ingredient.append(re.sub (r'([^a-zA-Z ]+?)', '', i))

    instructions = scraper.instructions()
    # links = scraper.links()
    nutrients = scraper.nutrients()
    return {'ingredient':ingredient,'ingredients':ingredients,'instructions':instructions,'nutrients':nutrients}

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
            header {visibility: hidden;}
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
    data = {'Food': food, 'Prob': prob}
    # return data
    dfhasil = pd.DataFrame.from_dict(data)

    dfhasil['Probability'] = dfhasil.apply(lambda x: f"{x['Prob']}%", axis=1)
    top3 = dfhasil.nlargest(3, 'Prob')
    # top = dict(zip(food, prob))
    # top3 = dict(sorted(top.items(), key=operator.itemgetter(1), reverse=True)[:3])
    return top3

# st.set_page_config(layout='wide')

# def main():
st.subheader("Food Ai Vision")
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
    c1,c2 = st.columns((1,1))
    with c1:
        st.image(image)
    with c2:
#     if st.button('Jalankan Prediksi'):
        hasil = getPrediction(data,model)
        hasil = hasil[['Food','Probability']]
        hasil.set_index('Food', inplace=True)
        st.write('Prediction')
        st.dataframe(hasil)
    # st.write(f'prediction: {hasil}')
    predicted = hasil.index[0]
    link = searchLink(predicted)
    recipe = scrapeRecipe(link)
    
    st.write(predicted.upper())
    st.write(link)
    st.write('Ingredients')
    st.table(recipe['ingredients'])
    st.write('Instructions')
    st.write(recipe['instructions'])
    nutrisi=recipe['nutrients']
    st.write('Nutrients')
    st.write(f'Food Nutrients:{nutrisi}')

# if __name__=='__main__':
#     main()
