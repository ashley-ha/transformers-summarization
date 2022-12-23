# Setting up virtual environment and installing Hugging Face's Transformers
'''
    Commands:
    pip3 install torch - Install PyTorch
    python3 -m venv .env # run in your project directory
    source .env/bin/activate
    pip3 install transformers
    python3 -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))" # Test if installed correctly
    For more information: https://huggingface.co/docs/transformers/installation

'''

from transformers import pipeline

# Load the summarization pipeline
summarization_pipeline = pipeline("summarization")

# Use the pipeline to summarize a piece of text
summary = summarization_pipeline("This is a longer piece of text that we want to summarize. It can be any length and contain any content, as long as it is in English. The pipeline will generate a shorter summary of the text, retaining the most important information.", max_length=50)

print(summary)