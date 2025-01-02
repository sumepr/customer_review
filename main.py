import pandas as pd
from data_cleaning import process_data
from eda import generate_eda_report
from text_processing import process_reviews
from visualization import generate_wordcloud_by_rating, plot_wordclouds, generate_top_words
from topic_modeling import perform_topic_modeling, generate_feedback

def main():
    """
    Main function to run the entire analysis pipeline
    """
    # Load and process data
    print("Loading and cleaning data...")
    df = process_data('your_data.csv')
    
    # Generate EDA report
    print("Generating EDA report...")
    eda_report = generate_eda_report(df)
    
    # Process review text
    print("Processing review text...")
    df_processed = process_reviews(df)
    
    # Generate word clouds
    print("Generating word clouds...")
    wordclouds = generate_wordcloud_by_rating(df_processed)
    wordcloud_plot = plot_wordclouds(wordclouds)
    
    # Generate top words
    print("Generating top words...")
    top_words = generate_top_words(df_processed)
    
    # Perform topic modeling
    print("Performing topic modeling...")
    topics, topic_matrix = perform_topic_modeling(df_processed)
    
    # Generate feedback
    print("Generating feedback...")
    feedback = generate_feedback(df_processed, topics, topic_matrix)
    
    # Save results
    wordcloud_plot.savefig('wordclouds.png')
    pd.DataFrame(top_words).to_csv('top_words.csv')
    pd.DataFrame(feedback).to_csv('feedback.csv')
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
