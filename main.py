import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import re
import nltk
from deep_translator import GoogleTranslator
import time

st.markdown(
    """
    <style>
    body {
        background-color: #f0f0f0;
        margin: 0;
        padding: 0;
    }
    .st-bw {
        background-color: #eeeeee;
    }
    .st-cq {
        background-color: #cccccc;
        border-radius: 10px;
        padding: 8px 12px;
        color: black;
    }
    .st-cx {
        background-color: white;
    }
    .sidebar .block-container {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

nltk.download('wordnet')
nltk.download('stopwords')

with open("./bookgenremodel.pkl", 'rb') as file:
    loaded_model = pickle.load(file)

def cleantext(text):
    text = re.sub("'\''", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = ' '.join(text.split())
    text = text.lower()
    return text

def removestopwords(text):
    stop_words = set(stopwords.words('english'))
    removedstopword = [word for word in text.split() if word not in stop_words]
    return ' '.join(removedstopword)

def lematizing(sentence):
    lemma = WordNetLemmatizer()
    stemSentence = ""
    for word in sentence.split():
        stem = lemma.lemmatize(word)
        stemSentence += stem + " "
    return stemSentence.strip()

def stemming(sentence):
    stemmer = PorterStemmer()
    stemmed_sentence = ""
    for word in sentence.split():
        stemmed_sentence += stemmer.stem(word) + " "
    return stemmed_sentence.strip()

def test(text, model, tfidf_vectorizer):
    text = cleantext(text)
    text = removestopwords(text)
    text = lematizing(text)
    text = stemming(text)

    text_vector = tfidf_vectorizer.transform([text])
    predicted = model.predict(text_vector)

    newmapper = {0: 'fantasy', 1: 'science', 2: 'crime', 3: 'history', 4: 'horror', 5: 'thriller'}
    return newmapper[predicted[0]]

def predict_genre(book_summary):
    if not book_summary:
        st.warning("Mohon Masukkan Ringkasan Buku.")
    else:
        progress_placeholder = st.empty()
        progress_placeholder.info("Sedang melakukan prediksi...")
        time.sleep(2)

        cleaned_summary = cleantext(book_summary)

        with open("./tfidfvector.pkl", 'rb') as file:
            vectorizer = pickle.load(file)

        vectorized_summary = vectorizer.transform([cleaned_summary])

        with open("./bookgenremodel.pkl", 'rb') as file:
            loaded_model = pickle.load(file)

        prediction = loaded_model.predict(vectorized_summary)
        newmapper = {0: 'fantasy', 1: 'science', 2: 'crime', 3: 'history', 4: 'horror', 5: 'thriller'}
        predicted_genre = newmapper[prediction[0]]

        progress_placeholder.empty()

        st.write("Hasil Prediksi Genre Buku")
        st.title(predicted_genre)
        st.success("Prediksi selesai!")

st.markdown("""
    <div style='display: flex; align-items: center; gap: 15px;'>
        <h1 style='margin: 0;'>Prediksi Genre Buku</h1>
    </div>
""", unsafe_allow_html=True)

book_summary = st.text_area("Masukkan Ringkasan Buku:")

def translate_to_english(text, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            translated = GoogleTranslator(source='auto', target='en').translate(text)
            return translated
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}. Mencoba ulang...")
            retries += 1
            time.sleep(1)
    st.error("Gagal menerjemahkan setelah beberapa kali percobaan.")
    return ""

if st.button("Terjemahkan ke Bahasa Inggris"):
    translated_summary = translate_to_english(book_summary)
    st.write("Ringkasan Terjemahan:")
    st.write(translated_summary)
    book_summary = translated_summary

if st.button("Prediksi Genre"):
    predict_genre(book_summary)