# Truth-Scope

**Truth-Scope** is an advanced AI-powered content detection tool designed to identify spam, verify rumors, and detect fake news. Built with Flask and Machine Learning, it provides a simple yet powerful interface for analyzing text content.

## Features

- **Spam Detection**: Identifies whether a message is spam or legitimate.
- **Rumor Verification**: Analyzes tweets to determine if they are true, false, non-rumors, or unverified.
- **Fake News Detector**: Classifies news articles as "Real News" or "Fake News".

## Technology Stack

- **Frontend**: HTML5, CSS3 (Premium Glassmorphism Design)
- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow/Keras, Scikit-learn
- **Models**:
    - `Spam_message.h5` (Keras)
    - `Tweet.h5` (Keras)
    - `news.joblib` (Scikit-learn)

## Setup & Usage

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd Truth-Scope
    ```

2.  **Install dependencies**:
    ```bash
    pip install flask tensorflow scikit-learn numpy joblib
    ```

3.  **Run the application**:
    ```bash
    python app.py
    ```

4.  **Access the tool**:
    Open your browser and navigate to `http://127.0.0.1:5000/`.

## Project Structure

- `app.py`: Main Flask application logic.
- `templates/index.html`: The user interface.
- `static/style.css`: Styling for the application.
- `models/`: Directory containing pre-trained ML models and vectorizers.

---
&copy; 2024 Truth-Scope AI
