from flask import Flask, render_template, request
import pickle
import joblib
import numpy as np

# Try to import TFLite runtime, fallback to full TensorFlow if not found (for local dev)
try:
    import tflite_runtime.interpreter as tflite
    print("Using tflite-runtime")
except ImportError:
    try:
        import tensorflow.lite as tflite
        # Check if Interpreter is available, if not try to import it from python.interpreter
        try:
            _ = tflite.Interpreter
        except AttributeError:
            from tensorflow.lite.python.interpreter import Interpreter
            # Monkey patch it back into tflite module for consistency
            tflite.Interpreter = Interpreter
        print("Using tensorflow.lite")
    except ImportError:
        tflite = None
        print("Warning: TensorFlow Lite not found. Models will be disabled.")

import os

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

models = {}
vectorizers = {}

# Helper function to run TFLite inference
def predict_tflite(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Resize input tensor to match input_data shape if necessary
    # (Not strictly needed if we always pass 1 sample, but good practice)
    interpreter.resize_tensor_input(input_details[0]['index'], input_data.shape)
    interpreter.allocate_tensors()

    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

if tflite:
    try:
        # Load Spam Model
        models["spam"] = tflite.Interpreter(model_path=os.path.join(MODEL_DIR, "spam_model.tflite"))
        models["spam"].allocate_tensors()
        
        # Load Tweet Model
        models["tweet"] = tflite.Interpreter(model_path=os.path.join(MODEL_DIR, "tweet_model.tflite"))
        models["tweet"].allocate_tensors()
        
        vectorizers["spam"] = pickle.load(open(os.path.join(MODEL_DIR, "spam_vectorizer.pkl"), "rb"))
        vectorizers["tweet"] = pickle.load(open(os.path.join(MODEL_DIR, "tweet_vectorizer.pkl"), "rb"))
    except Exception as e:
        print(f"Error loading TFLite models: {e}")
        import traceback
        traceback.print_exc()
        models["spam"] = None
        models["tweet"] = None
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
                    # TFLite inference
                    pred = predict_tflite(models["tweet"], vect.toarray())
                    prediction = np.argmax(pred, axis=1)[0]
                    label = rumor_labels[prediction]
                else:
                    label = "Error: Model not loaded (TFLite missing)"

            elif selected_model == "spam":
                if models["spam"]:
                    vect = vectorizers["spam"].transform([clean_input])
                    # TFLite inference
                    pred = predict_tflite(models["spam"], vect.toarray())
                    prob = pred[0][0]
                    prediction = int(prob >= 0.5)
                    label = "Spam message" if prediction == 1 else "Likely not spam"
                else:
                    label = "Error: Model not loaded (TFLite missing)"

    return render_template(
        "index.html",
        prediction=label,
        selected_model=selected_model,
        user_input=user_input,
    )

if __name__ == "__main__":
    app.run(debug=True)
