from transformers import pipeline
import pandas as pd

books = pd.read_csv('Books_rating.csv')
books = books.drop(columns = ['Price', 'review/time', 'review/summary', 'review/score', 'profileName', 'review/helpfulness', 'User_id'])
les_miserables = books[books['Title'] == 'Les Miserables']
les_miserables = les_miserables.astype('str')
seq = ' '.join(les_miserables['review/text'].tolist())
seq = seq[:1000]

# # Load the summarization pipeline
summarization_pipeline = pipeline("summarization")

# # # Use the pipeline to summarize a piece of text
lesmiz_summary = summarization_pipeline(seq, max_length=100)

print(lesmiz_summary)



