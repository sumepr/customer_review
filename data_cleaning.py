import pandas as pd
import numpy as np
from datetime import datetime

def clean_data(df):
    """
    Clean the dataset and handle missing values
    """
    # Create a copy to avoid modifying original data
    df_clean = df.copy()
    
    # Convert timestamp columns to datetime
    timestamp_cols = ['REVIEW_FIRSTPUBLISHTIME', 'REVIEW_LASTMODIFICATIONTIME', 'REVIEW_SUBMISSIONTIME']
    for col in timestamp_cols:
        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    # Clean URL column
    df_clean['REVIEW_PRODUCTREVIEWSDEEPLINKEDURL'] = df_clean['REVIEW_PRODUCTREVIEWSDEEPLINKEDURL'].str.strip()
    
    # Convert rating to numeric
    df_clean['REVIEW_RATING'] = pd.to_numeric(df_clean['REVIEW_RATING'], errors='coerce')
    df_clean['REVIEW_RATINGRANGE'] = pd.to_numeric(df_clean['REVIEW_RATINGRANGE'], errors='coerce')
    
    # Convert boolean columns
    bool_cols = ['REVIEW_RATINGSONLY', 'PRODUCT_ACTIVE_INDICATOR', 'PRODUCT_SAMPLE_INDICATOR']
    for col in bool_cols:
        df_clean[col] = df_clean[col].map({'Y': True, 'N': False})
    
    return df_clean

def impute_data(df):
    """
    Impute missing values in the dataset
    """
    df_imputed = df.copy()
    
    # Impute numeric columns with median
    numeric_cols = ['REVIEW_RATING', 'REVIEW_RATINGRANGE']
    for col in numeric_cols:
        df_imputed[col].fillna(df_imputed[col].median(), inplace=True)
    
    # Impute categorical columns with mode
    categorical_cols = ['PRODUCT_PRICING_GROUP_NAME', 'PRODUCT_GROUP_NAME', 'PRODUCT_ROLL_UP_NAME']
    for col in categorical_cols:
        df_imputed[col].fillna(df_imputed[col].mode()[0], inplace=True)
    
    # Impute text columns with empty string
    text_cols = ['REVIEW_REVIEWTEXT', 'REVIEW_TITLE']
    for col in text_cols:
        df_imputed[col].fillna('', inplace=True)
    
    # Forward fill timestamps
    timestamp_cols = ['REVIEW_FIRSTPUBLISHTIME', 'REVIEW_LASTMODIFICATIONTIME', 'REVIEW_SUBMISSIONTIME']
    for col in timestamp_cols:
        df_imputed[col].fillna(method='ffill', inplace=True)
    
    return df_imputed

def process_data(input_path):
    """
    Main function to clean and impute data
    """
    # Read the data
    df = pd.read_csv(input_path)
    
    # Clean the data
    df_cleaned = clean_data(df)
    
    # Impute missing values
    df_processed = impute_data(df_cleaned)
    
    return df_processed
