import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect
from googletrans import Translator
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def detect_language(text):
    """
    Detect the language of the text
    """
    try:
        return detect(str(text))
    except:
        return 'en'

def translate_text(text, translator):
    """
    Translate non-English text to English
    """
    try:
        if detect_language(text) != 'en':
            return translator.translate(text, dest='en').text
        return text
    except:
        return text

def preprocess_text(text):
    """
    Preprocess text data
    """
    # Convert to lowercase
    text = str(text).lower()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def process_reviews(df):
    """
    Process review text data
    """
    # Initialize translator
    translator = Translator()
    
    # Create copy of dataframe
    df_processed = df.copy()
    
    # Translate non-English reviews
    print("Translating non-English reviews...")
    df_processed['REVIEW_REVIEWTEXT_EN'] = df_processed['REVIEW_REVIEWTEXT'].apply(
        lambda x: translate_text(x, translator)
    )
    
    # Preprocess reviews
    print("Preprocessing reviews...")
    df_processed['REVIEW_REVIEWTEXT_PROCESSED'] = df_processed['REVIEW_REVIEWTEXT_EN'].apply(
        preprocess_text
    )
    
    return df_processed
