import streamlit as st
import pickle
import numpy as np
import pickle


st.set_page_config(page_title="Sentiment Analysis", layout='centered')

st.title('Word2Vector Sentiment Analysis App')
st.write("Trained on kindle review dataset with a test accuracy of 77%.")

with open('artifacts/classifier.pkl', 'rb+') as file:
    classifier = pickle.load(file)

with open('artifacts/wv_model.pkl', 'rb+') as file:
    wv_model = pickle.load(file)

def avg_word2vec(doc, model):
    valid_words = [wv_model.wv[word] for word in doc if word in wv_model.wv.index_to_key]
    return np.mean(valid_words, axis=0) if valid_words else np.zeros(wv_model.vector_size)

def vectorize(column):
    col_vec = [i.split() for i in column]
    for i in range(len(col_vec)):
        col_vec[i] = avg_word2vec(col_vec[i], wv_model)
    return np.array(col_vec)

text_input = st.text_area("Enter Review Text", height=150)

if st.button('Analyze'):
    prediction = classifier.predict(vectorize([text_input]))

    if prediction == 1:
        sentiment = 'Positive'
    else:
        sentiment = 'Negative'
    
    st.subheader(f'Sentiment: {sentiment}')


