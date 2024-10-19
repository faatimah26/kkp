import pickle
import streamlit as st

# Load model dan vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vect_file:
    vectorizer = pickle.load(vect_file)

class Model:
    def __init__(self):
        self.model = model
        self.vectorizer = vectorizer

    def predict(self, text):
        vectorized_text = self.vectorizer.transform([text])
        prediction = self.model.predict(vectorized_text)
        return 'Bullying' if prediction[0] == 0 else 'Non-Bullying'

# Inisialisasi instance model
model_instance = Model()

# Membuat antarmuka pengguna dengan Streamlit
st.title("Deteksi Bullying")
input_text = st.text_area("Masukkan komentar Anda:")

if st.button("Prediksi"):
    if input_text:
        prediction = model_instance.predict(input_text)
        st.write(f"Hasil Prediksi: {prediction}")
    else:
        st.write("Silakan masukkan komentar untuk prediksi.")
