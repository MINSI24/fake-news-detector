import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset (SAFE FIX)
data = pd.read_csv("dataset.csv", header=None)
data.columns = ["text", "label"]

data.dropna(inplace=True)

X = data["text"]
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Accuracy
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

# Predict function
def predict_news(news):
    vec = vectorizer.transform([news])
    prediction = model.predict(vec)
    prob = model.predict_proba(vec)

    confidence = max(prob[0]) * 100
    return prediction[0], confidence

# Accuracy function
def get_accuracy():
    return round(accuracy * 100, 2)
