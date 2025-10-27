from flask import Flask, render_template, request
from joblib import load

# Load model
model = load('sentiment_model.joblib')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        prediction = model.predict([review])[0]
        sentiment = 'Positive 😊' if prediction == 1 else 'Negative 😞'
        return render_template('index.html', prediction_text=f'Sentiment: {sentiment}')

if __name__ == '__main__':
    app.run(debug=True)
