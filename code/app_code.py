# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import pickle

# # --- SETTINGS ---
# MODEL_PATH = "sentiment analysis.keras"
# MAX_LEN = 100

# def load_prediction_tools():
#     """Load the model and required preprocessing tools"""
#     try:
#         # Load the trained model
#         model = tf.keras.models.load_model(r"C:\MH_Sentiment_Project\code\sentiment analysis .keras")
#         print(f"Successfully loaded: {MODEL_PATH}")
#         return model
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return None

# # Initialize the predictor
# predictor_model = load_prediction_tools()

# def get_intensity(confidence):
#     """Calculate the intensity based on model confidence"""
#     if confidence > 0.90: return "CRITICAL / HIGH"
#     elif confidence > 0.75: return "MODERATE"
#     else: return "MILD / STABLE"

# def predict_single_post(text):
#     """Analyze a single string of text"""
#     if predictor_model is None: return
    
#     # Preprocess (using the function from Cell 3)
#     clean = preprocess_text(text)
#     seq = tokenizer.texts_to_sequences([clean])
#     padded = pad_sequences(seq, maxlen=MAX_LEN)
    
#     # Predict
#     probs = predictor_model.predict(padded, verbose=0)[0]
#     idx = np.argmax(probs)
#     emotion = label_encoder.classes_[idx]
#     conf = probs[idx]
    
#     return {
#         "text": text,
#         "emotion": emotion,
#         "intensity": get_intensity(conf),
#         "confidence": f"{conf*100:.1f}%"
#     }

# def analyze_batch_file(file_path, text_column):
#     """Load a CSV file and analyze a specific column of posts"""
#     print(f"Reading file: {file_path}...")
#     data = pd.read_csv(file_path)
    
#     results = []
#     print("Processing posts...")
    
#     for post in data[text_column].astype(str):
#         res = predict_single_post(post)
#         results.append(res)
    
#     # Create results dataframe
#     results_df = pd.DataFrame(results)
#     output_name = "mental_health_analysis_results.csv"
#     results_df.to_csv(output_name, index=False)
    
#     print(f"\nAnalysis Complete! Results saved to: {output_name}")
#     print(results_df.head())

# # --- EXAMPLES OF USE ---

# # Example 1: Single Prediction
# print("\n[SINGLE POST TEST]")
# result = predict_single_post("I feel depressed and anxious all the time.")
# print(f"Result: {result}")
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

# --- SETTINGS ---
MODEL_PATH = r"C:\MH_Sentiment_Project\code\sentiment_analysis.keras"
TOKENIZER_PATH = r"C:\MH_Sentiment_Project\code\tokenizer.pkl"
LABEL_ENCODER_PATH = r"C:\MH_Sentiment_Project\code\label_encoder.pkl"
MAX_LEN = 100

# --- TEXT PREPROCESSING ---
def preprocess_text(text):
    """Clean input text"""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()

# --- LOAD MODEL + TOOLS ---
def load_prediction_tools():
    """Load model, tokenizer, and label encoder"""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)

        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)

        with open(LABEL_ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)

        print("✅ Model, tokenizer, and label encoder loaded successfully!")
        return model, tokenizer, label_encoder

    except Exception as e:
        print(f"❌ Error loading components: {e}")
        return None, None, None

# Initialize
predictor_model, tokenizer, label_encoder = load_prediction_tools()

# --- INTENSITY LOGIC ---
def get_intensity(confidence):
    """Determine severity level"""
    if confidence > 0.90:
        return "CRITICAL / HIGH"
    elif confidence > 0.75:
        return "MODERATE"
    else:
        return "MILD / STABLE"

# --- SINGLE TEXT PREDICTION ---
def predict_single_post(text):
    """Analyze a single text input"""

    if predictor_model is None or tokenizer is None or label_encoder is None:
        return {"error": "Model or preprocessing tools not loaded"}

    # Preprocess
    clean = preprocess_text(text)

    # Tokenize
    seq = tokenizer.texts_to_sequences([clean])

    # Pad
    padded = pad_sequences(seq, maxlen=MAX_LEN)

    # Predict
    probs = predictor_model.predict(padded, verbose=0)[0]

    idx = np.argmax(probs)
    emotion = label_encoder.classes_[idx]
    conf = probs[idx]

    return {
        "text": text,
        "emotion": emotion,
        "intensity": get_intensity(conf),
        "confidence": f"{conf * 100:.2f}%"
    }

# --- BATCH FILE ANALYSIS ---
def analyze_batch_file(file_path, text_column):
    """Analyze CSV file column"""

    if predictor_model is None:
        print("❌ Model not loaded")
        return

    print(f"📂 Reading file: {file_path}")
    data = pd.read_csv(file_path)

    results = []
    print("⚙️ Processing posts...")

    for post in data[text_column].astype(str):
        res = predict_single_post(post)
        results.append(res)

    results_df = pd.DataFrame(results)

    output_name = "mental_health_analysis_results.csv"
    results_df.to_csv(output_name, index=False)

    print(f"\n✅ Analysis Complete! Saved to: {output_name}")
    print(results_df.head())

# --- TEST RUN ---
if __name__ == "__main__":
    print("\n🔍 [SINGLE POST TEST]")
    
    result = predict_single_post("I feel depressed and anxious all the time.")
    
    print("Result:")
    print(result)