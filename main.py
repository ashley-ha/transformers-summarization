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
medical_summary = summarization_pipeline("The symptoms of a stroke often appear without warning. Some of the main symptoms include: confusion, including difficulty speaking and understanding speech, a headache, possibly with altered consciousness or vomiting numbness or an inability to move parts of the face, arm, or leg, particularly on one side of the body vision problems in one or both eyes difficulty walking, including dizziness and a lack of coordination Stroke can lead to long-term health problems. Depending on the speed of the diagnosis and treatment, a person can experience temporary or permanent disabilities after a stroke. Some people may also experience: bladder or bowel control problems, depression, paralysis or weakness on one or both sides of the body, difficulty controlling or expressing their emotions Symptoms vary and may range in severity. Learning the acronym “FAST” is a good way to remember the symptoms of stroke. This can help a person seek prompt treatment. FAST stands for: Face drooping: If the person tries to smile, does one side of their face droop? Arm weakness: If the person tries to raise both their arms, does one arm drift downward? Speech difficulty: If the person tries to repeat a simple phrase, is their speech slurred or unusual? Time to act: If any of these symptoms are occurring, contact the emergency services immediately. The outcome depends on how quickly someone receives treatment. Prompt care also means that they would be less likely to experience permanent brain damage or death.", max_length=75)

print(medical_summary)