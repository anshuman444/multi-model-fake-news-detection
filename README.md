# Multi-Model Fake News & Spam Detection

A lightweight, machine-learning-powered web application that detects:
1.  **Fake News**: Classifies news articles as Real or Fake.
2.  **Spam Messages**: Identifies SMS/Email spam.
3.  **Rumors in Tweets**: Categorizes tweets as True, False, Non-rumor, or Unverified.

Built with **Flask** and optimized with **TensorFlow Lite** for fast, low-memory deployment.

## 🚀 Features
- **Multi-Model Support**: Switch between three different AI models instantly.
- **Optimized for Web**: Uses TFLite (`.tflite`) models to keep the app lightweight (~50MB) and fast.
- **Responsive UI**: Clean, modern interface.

## 🛠️ Tech Stack
- **Backend**: Python, Flask
- **ML Engine**: TensorFlow Lite (TFLite), Scikit-learn
- **Frontend**: HTML, CSS
- **Deployment**: Ready for Render.com (Gunicorn + Procfile included)

## 📦 Installation (Local)

1.  **Clone the repository**
    ```bash
    git clone https://github.com/anshuman444/multi-model-fake-news-detection.git
    cd multi-model-fake-news-detection
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the App**
    ```bash
    python app.py
    ```
    Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## ☁️ Deployment (Render.com)

This project is pre-configured for **Render**.

1.  Fork/Clone this repo to your GitHub.
2.  Log in to [Render Dashboard](https://dashboard.render.com/).
3.  Click **New +** > **Web Service**.
4.  Connect your repository.
5.  **Settings**:
    - **Runtime**: Python 3
    - **Build Command**: `pip install -r requirements.txt`
    - **Start Command**: `gunicorn app:app`
6.  Click **Deploy**.

## 📂 Project Structure
```
├── models/             # TFLite models and vectorizers
├── static/             # CSS files
├── templates/          # HTML templates
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── Procfile            # Deployment configuration
└── README.md           # Documentation
```

## 🧠 Model Details
- **News Model**: Logistic Regression (Scikit-learn)
- **Spam Model**: Neural Network (Converted to TFLite)
- **Tweet Model**: Neural Network (Converted to TFLite)
