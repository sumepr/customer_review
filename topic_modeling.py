import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from collections import defaultdict

def perform_topic_modeling(df, n_topics=5):
    """
    Perform topic modeling on reviews
    """
    # Initialize vectorizer
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
    # Create document-term matrix
    dtm = vectorizer.fit_transform(df['REVIEW_REVIEWTEXT_PROCESSED'])
    
    # Perform NMF
    nmf = NMF(n_components=n_topics, random_state=42)
    topic_matrix = nmf.fit_transform(dtm)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Extract top words for each topic
    topics = {}
    for topic_idx, topic in enumerate(nmf.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-10:-1]]
        topics[f'Topic_{topic_idx+1}'] = top_words
    
    return topics, topic_matrix

def generate_feedback(df, topics, topic_matrix):
    """
    Generate feedback based on topic modeling results
    """
    feedback = defaultdict(list)
    
    # Analyze topics by rating
    for rating in range(1, 6):
        # Filter reviews by rating
        rating_mask = df['REVIEW_RATING'] == rating
        rating_topics = topic_matrix[rating_mask]
        
        # Get dominant topics
        dominant_topics = rating_topics.argmax(axis=1)
        topic_counts = pd.Series(dominant_topics).value_counts()
        
        # Generate feedback
        if rating <= 3:  # Focus on improvement areas for lower ratings
            for topic_idx in topic_counts.index:
                topic_words = topics[f'Topic_{topic_idx+1}']
                feedback['improvement_areas'].append({
                    'rating': rating,
                    'topic': topic_words,
                    'count': topic_counts[topic_idx]
                })
        else:  # Identify strengths from higher ratings
            for topic_idx in topic_counts.index:
                topic_words = topics[f'Topic_{topic_idx+1}']
                feedback['strengths'].append({
                    'rating': rating,
                    'topic': topic_words,
                    'count': topic_counts[topic_idx]
                })
    
    return dict(feedback)
