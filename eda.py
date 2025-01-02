import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def analyze_numeric_columns(df):
    """
    Analyze numeric columns in the dataset
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = {}
    
    for col in numeric_cols:
        stats[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'missing': df[col].isnull().sum()
        }
    
    return pd.DataFrame(stats).T

def analyze_categorical_columns(df):
    """
    Analyze categorical columns in the dataset
    """
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
    stats = {}
    
    for col in categorical_cols:
        stats[col] = {
            'unique_values': df[col].nunique(),
            'most_common': df[col].mode()[0],
            'missing': df[col].isnull().sum()
        }
    
    return pd.DataFrame(stats).T

def analyze_temporal_patterns(df):
    """
    Analyze temporal patterns in reviews
    """
    df['review_month'] = df['REVIEW_SUBMISSIONTIME'].dt.month
    df['review_day'] = df['REVIEW_SUBMISSIONTIME'].dt.day_name()
    
    temporal_patterns = {
        'monthly_reviews': df['review_month'].value_counts().sort_index(),
        'daily_reviews': df['review_day'].value_counts(),
        'avg_rating_by_month': df.groupby('review_month')['REVIEW_RATING'].mean()
    }
    
    return temporal_patterns

def plot_rating_distribution(df):
    """
    Plot the distribution of ratings
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='REVIEW_RATING')
    plt.title('Distribution of Review Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    return plt

def generate_eda_report(df):
    """
    Generate comprehensive EDA report
    """
    report = {
        'numeric_analysis': analyze_numeric_columns(df),
        'categorical_analysis': analyze_categorical_columns(df),
        'temporal_patterns': analyze_temporal_patterns(df),
        'rating_distribution': plot_rating_distribution(df)
    }
    
    return report
