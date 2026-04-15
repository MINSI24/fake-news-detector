from flask import Flask, render_template, request
from model import predict_news, get_accuracy

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    accuracy = get_accuracy()

    if request.method == "POST":
        news = request.form["news"]

        prediction, confidence = predict_news(news)
        prediction = prediction.lower()

        if prediction == "fake":
            result = f"❌ FAKE NEWS ({confidence:.2f}%)"
        else:
            result = f"✅ REAL NEWS ({confidence:.2f}%)"

    return render_template("index.html", result=result, accuracy=accuracy)

if __name__ == "__main__":
    app.run(debug=True)
    