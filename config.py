"""Project paths and constants (single source of truth)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TEXT_CSV = DATA_DIR / "text" / "Combined Data.csv"
IMG_TRAIN = DATA_DIR / "img" / "train"
IMG_TEST = DATA_DIR / "img" / "test"
MODELS_DIR = ROOT / "models"

NLP_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
NLP_MODEL_PATH = MODELS_DIR / "nlp_logistic.joblib"
NLP_THRESHOLD_JSON = MODELS_DIR / "nlp_threshold.json"
NLP_EVAL_JSON = MODELS_DIR / "nlp_eval.json"
NLP_CONFUSION_PNG = MODELS_DIR / "nlp_confusion.png"
CNN_MODEL_PATH = MODELS_DIR / "emotion_cnn.pt"
CNN_CLASS_NAMES_JSON = MODELS_DIR / "cnn_class_names.json"
CNN_TRAINING_META_JSON = MODELS_DIR / "cnn_training_meta.json"
CNN_EPOCH_HISTORY_JSON = MODELS_DIR / "cnn_epoch_history.json"
CNN_EVAL_JSON = MODELS_DIR / "cnn_eval.json"
CNN_CONFUSION_PNG = MODELS_DIR / "cnn_confusion.png"

IMG_SIZE = (48, 48)
CNN_CLASS_NAMES = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")

NORMAL_LABEL = "Normal"

# Map sentiment classes to friendly names for output
NLP_CLASS_LABELS = {0: "Positive (Normal)", 1: "Negative (Depression)"}

# Confidence thresholds used by the supervisor agent
LOW_CONFIDENCE_THRESHOLD = 0.55

# Below this length/word count, TF-IDF rarely matches training n-grams — predictions unreliable.
# Bumped from 20/4 after a real failure case where a 5-word crisis sentence
# (with typos) was scored 94% Positive — the safety lexicon now catches that
# regardless, but tightening these thresholds makes the warning fire earlier.
NLP_MIN_CHARS_FOR_RELIABLE = 35
NLP_MIN_WORDS_FOR_RELIABLE = 6

# Final-status label used by the SafetyAgent override.
CRISIS_LABEL = "Crisis — seek support"
