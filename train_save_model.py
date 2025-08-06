import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Tiny example dataset
data = {
    'text': [
        'Breaking news: President signs new law',
        'Celebrity caught in scandal',
        'Scientists discover cure for disease',
        'Fake news about election fraud',
        'This is a hoax spreading false rumors',
        'Conspiracy theories about the government'
    ],
    'label': [1, 1, 1, 0, 0, 0]  # 1 = Real, 0 = Fake
}

df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_train_vectors, y_train)

# Save model and vectorizer
joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model trained and saved!")
