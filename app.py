import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

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

@app.route('/')
def index():
    return render_template('app.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['comment']
        prediction = model_instance.predict(input_text)
        return render_template('app.html', prediction=prediction)
    return render_template('app.html')

if __name__ == '__main__':
    app.run(debug=True)
