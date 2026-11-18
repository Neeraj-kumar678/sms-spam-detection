import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()                     # 1. Lowercase everything
    text = nltk.word_tokenize(text)         # 2. Tokenize into words

    y = []
    for i in text:                          # 3. Keep only alphanumeric tokens
        if i.isalnum():
            y.append(i)

    text = y[:]                             # 4. Copy filtered tokens
    y.clear()

    for i in text:                          # 5. Remove stopwords & punctuation
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]                             # 4. Copy filtered tokens
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)                                # 6. Return cleaned tokens

tfidf = pickle.load(open('vectorizer_1.pkl', 'rb'))
model = pickle.load(open('model_1.pkl', 'rb'))
st.title("📱 SMS Spam Detection")
input_sms = st.text_area("Enter the SMS:")
if st.button('Predict'):

    transform_sms=transform_text(input_sms)
    transform_sms = transform_text(input_sms)
    st.write("Transformed Text:", transform_sms)
    vector_input=tfidf.transform([transform_sms])
    st.write(vector_input.shape)
    st.write(vector_input.nnz)
    result=model.predict(vector_input)[0]
    proba = model.predict_proba(vector_input)

    st.write("Prediction:", result)
    st.write("Probabilities:", proba)
    st.write(result)
    if result == 1:
     st.header("🚨 Spam SMS")
    else:
     st.header("✅ Not Spam (Ham)")

