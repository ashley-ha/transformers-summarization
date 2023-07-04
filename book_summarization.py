# Author @Ashley Eastman


# Setting up virtual environment and installing Hugging Face's Transformers
'''
    Commands:
    python3 -m venv .env # run in your project directory
    source .env/bin/activate
    pip3 install transformers
    python3 -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))" # Test if installed correctly
    For more information: https://huggingface.co/docs/transformers/installation

    Commands for YELP API connection:
    pip3 install yelpapi

'''

# Importing libraries 
from transformers import pipeline
import pandas as pd

# Book reviews using Amazon Reviews Dataset via Kaggle
books = pd.read_csv('Books_rating.csv')
books = books.drop(columns = ['Price', 'review/time', 'review/summary', 'review/score', 'profileName', 'review/helpfulness', 'User_id'])
les_miserables = books[books['Title'] == 'Les Miserables']
les_miserables = les_miserables.astype('str')

# Limiting sequences to 1000 to avoid max sequence errors in our model
seq = ' '.join(les_miserables['review/text'].tolist())
seq = seq[:2000]

# # Load the summarization pipeline
summarization_pipeline = pipeline("summarization")

# # # Use the pipeline to summarize a piece of text - in our case, Les Miserables book reviews
lesmiz_summary = summarization_pipeline(seq, max_length=150)

print(lesmiz_summary)

# output: [{'summary_text': " Some areas I skipped because I did not see the relevance at that moment . Parts of it are hard to understand, being that the book was written in old french, but the story as a whole isn't confusing . Norman Denny's translation lives up to its promise. It's the greatest novel of all time in my opinion ."}]

