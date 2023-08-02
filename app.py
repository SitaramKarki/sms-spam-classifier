import streamlit as st
import string
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


ps = PorterStemmer()


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('SMS Spam Classifier \n -an initiative by Situ Entreprises Ltd.')

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess

    def transform_text(text):
        text = text.lower()
        text = nltk.word_tokenize(text)
        y = []
        for i in text:
            if i.isalnum():
                y.append(i)
        text = y[:]
        #data cannot be copied as text = y always be careful
        y.clear()
        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)
        text = y[:]
        y.clear()
        for i in text:
            y.append(ps.stem(i))


        return " ".join(y)

    transformed_sms = transform_text(input_sms)
    # 2. vectorize

    vector_imput = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_imput)[0]
    # 4. display

    if result == 1:
        st.header("This is a Spam SMS")
    else:
        st.header("This is a not a Spam SMS")


