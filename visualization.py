import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

def generate_wordcloud_by_rating(df):
    """
    Generate word clouds for each rating
    """
    wordclouds = {}
    
    for rating in range(1, 6):
        # Filter reviews by rating
        reviews = df[df['REVIEW_RATING'] == rating]['REVIEW_REVIEWTEXT_PROCESSED']
        
        # Combine all reviews for this rating
        text = ' '.join(reviews)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            max_words=100
        ).generate(text)
        
        # Store wordcloud
        wordclouds[rating] = wordcloud
    
    return wordclouds

def plot_wordclouds(wordclouds):
    """
    Plot word clouds for each rating
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.ravel()
    
    for rating, wordcloud in wordclouds.items():
        ax = axes[rating-1]
        ax.imshow(wordcloud)
        ax.set_title(f'Rating {rating} Reviews')
        ax.axis('off')
    
    # Remove the extra subplot
    fig.delaxes(axes[5])
    plt.tight_layout()
    return fig

def generate_top_words(df, n=10):
    """
    Generate top words for each rating
    """
    top_words = {}
    
    for rating in range(1, 6):
        # Filter reviews by rating
        reviews = df[df['REVIEW_RATING'] == rating]['REVIEW_REVIEWTEXT_PROCESSED']
        
        # Count word frequencies
        words = ' '.join(reviews).split()
        word_freq = Counter(words)
        
        # Get top words
        top_words[rating] = dict(word_freq.most_common(n))
    
    return top_words
