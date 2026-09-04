import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

@st.cache_resource
def download_nltk_data():

    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')


download_nltk_data()

ps = PorterStemmer()

def data_transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    list = []
    for i in text:
        if i.isalnum():
            list.append(i)
    
    text = list[:]
    list.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            list.append(i)
            
    text = list[:]
    list.clear()
    
    for i in text:
        list.append(ps.stem(i))
    
            
    return " ".join(list)

with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('Email/SMS Spam Classifier')

input_message = st.text_area('Enter you message')

if st.button('Predict'):

    transformed_message = data_transform(input_message)
    prediction = model.predict([transformed_message])[0]
    probabilities = model.predict_proba([transformed_message])[0]

    if prediction == 1:
        st.header('Spam')
        st.write(f'Spam Probability: {probabilities[1]:.2%}')

    else:
        st.header('Not Spam')
        st.write(f'Spam Probability: {probabilities[1]:.2%}')
