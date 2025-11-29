from flask import Flask, render_template, request
import pickle
import joblib
import numpy as np
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not found. Spam and Tweet models will be disabled.")

import os

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

models = {}
vectorizers = {}

if TF_AVAILABLE:
    try:
        models["spam"] = load_model(os.path.join(MODEL_DIR, "Spam_message.h5"))
        models["tweet"] = load_model(os.path.join(MODEL_DIR, "Tweet.h5"))
        
        vectorizers["spam"] = pickle.load(open(os.path.join(MODEL_DIR, "spam_vectorizer.pkl"), "rb"))
        vectorizers["tweet"] = pickle.load(open(os.path.join(MODEL_DIR, "tweet_vectorizer.pkl"), "rb"))
    except Exception as e:
        print(f"Error loading TF models: {e}")
        TF_AVAILABLE = False
else:
    models["spam"] = None
    models["tweet"] = None

LR_model, LR_vectorizer = joblib.load(os.path.join(MODEL_DIR, "news.joblib"))

def preprocess_txt(text):
    return text.lower().strip()

rumor_labels = {0: "false", 1: "non-rumor", 2: "true", 3: "unverified"}

@app.route("/", methods=["GET", "POST"])
def home():
    label = None
    prediction = None
    selected_model = None
    user_input = ""

    if request.method == "POST":
        user_input = request.form.get("user_input")
        selected_model = request.form.get("model_selected")

        if user_input and selected_model:
            clean_input = preprocess_txt(user_input)


            if selected_model == "news":
                vect = LR_vectorizer.transform([clean_input])
                pred = LR_model.predict(vect)
                label = "Fake News" if pred[0] == 1 else "Real News"

            elif selected_model == "tweet":
                if models["tweet"]:
                    vect = vectorizers["tweet"].transform([clean_input])
                    pred = models["tweet"].predict(vect.toarray())
                    prediction = np.argmax(pred, axis=1)[0]
                    label = rumor_labels[prediction]
                else:
                    label = "Error: Model not loaded (TensorFlow missing)"

            elif selected_model == "spam":
                if models["spam"]:
                    vect = vectorizers["spam"].transform([clean_input])
                    pred = models["spam"].predict(vect.toarray())
                    prob = pred[0][0]
                    prediction = int(prob >= 0.5)
                    label = "Spam message" if prediction == 1 else "Likely not spam"
                else:
                    label = "Error: Model not loaded (TensorFlow missing)"

    return render_template(
        "index.html",
        prediction=label,
        selected_model=selected_model,
        user_input=user_input,
    )

if __name__ == "__main__":
    app.run(debug=True)
