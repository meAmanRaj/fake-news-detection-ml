from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# load trained model
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news = request.json["news"]
    vector = vectorizer.transform([news])

    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0]

    confidence = max(probability) * 100

    result = "Real News" if prediction == 1 else "Fake News"

    return jsonify({
        "prediction": result,
        "confidence": f"{confidence:.2f}%"
    })

if __name__ == "__main__":
    app.run(debug=True)
