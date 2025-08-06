from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        news_text = request.form['news_text']
        vect = vectorizer.transform([news_text])
        pred = model.predict(vect)[0]
        prediction = 'Real News' if pred == 1 else 'Fake News'
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
