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

from transformers import pipeline
from yelpapi import YelpAPI

# Initialize the Yelp API client with your API key
business_ids = []
API_KEY = 'your API Key here'
# Yelp API client
client = YelpAPI('API_KEY')

# Search for five businesses
response = client.search_query(term='food', location='Oakland, CA', limit = 20)

# Get the review text for each business
for business in response['businesses']:
    business_ids.append(business['id'])


reviews = []
# Get the reviews for a specific business
response = client.reviews_query(id='IZ5ya4olYUc19-EIWRCyuQ')

# Append the review text for each review to the list
for review in response['reviews']:
    reviews.append(review['text'])
# Join the reviews into a single string
review_text = ' '.join(reviews)


# # Load the summarization pipeline
summarization_pipeline = pipeline("summarization")

# # # Use the pipeline to summarize a piece of text
yelp_summary = summarization_pipeline(review_text, max_length=60)

print(yelp_summary)

#output = [{'summary_text': ' Azit offers up a selection of Korean cuisine such as fried chicken, large casserole stews, pancakes and appetizers .
#  The Korean food was unexpectedly amazing,
# and I loved and would recommend everything we ordered. Azit is such a good place for a late night drunchie'}]