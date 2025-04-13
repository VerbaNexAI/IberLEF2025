import os
import glob
import json
import pandas as pd
import torch
import re
import emoji
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, LongformerTokenizer, LongformerModel
from pycaret.classification import load_model, predict_model
from codecarbon import EmissionsTracker
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Define the base directory
BASE_DIR = Path(__file__).resolve().parent.parent

class DatasetColumnTransformer:
    """
    A class for text preprocessing including lemmatization, removing special patterns,
    and replacing certain elements with placeholders.
    """
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+\.\S+')
        self.user_pattern = re.compile(r'@\w+')
        self.html_pattern = re.compile(r'<.*?>')
        self.markdown_pattern = re.compile(r'\[.*?\]\(.*?\)')
        self.punctuation_pattern = re.compile(r'([!?.,]){7,}')
        self.double_quotes_pattern = re.compile(r'[“”]')
        self.single_quotes_pattern = re.compile(r'[‘’]')

    def clean_text(self, text):
        """
        Cleans and preprocesses the input text by applying various transformations.
        """
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = self.url_pattern.sub("[URL]", text)
        text = self.email_pattern.sub("[EMAIL]", text)
        text = self.user_pattern.sub("[USER]", text)
        text = self.html_pattern.sub("", text)
        text = self.markdown_pattern.sub("", text)
        text = emoji.demojize(text, language='es')
        text = re.sub(r':(\w+):', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = self.double_quotes_pattern.sub('"', text)
        text = self.single_quotes_pattern.sub("'", text)
        text = self.punctuation_pattern.sub(r'\1\1\1', text)
        words = word_tokenize(text)
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return " ".join(words)

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BETO
model_name = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# Load the trained classification model
best_model_path = BASE_DIR / "models" / "final_models" / "lr_task2_final"
best_model = load_model(str(best_model_path))

# Directory containing data
directory = BASE_DIR / "data" / "trial" / "task2" / "subjects"

# Define category mapping
mapping = {
    'betting': 0,
    'trading': 1,
    'lootboxes': 2,
    'onlinegaming': 3
}

text_cleaner = DatasetColumnTransformer()

def get_longformer_embedding(text, max_length=512):
    """
    Generate embeddings using the Longformer model.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def process_incremental_messages(user_id, messages):
    """
    Process messages incrementally, updating the conversation history
    and making predictions for each new message.
    """
    history = ""
    print(f"\nProcessing user {user_id}...")
    for i in range(len(messages)):
        cleaned_message = text_cleaner.clean_text(messages[i])
        history = " [SEP] ".join([text_cleaner.clean_text(msg) for msg in messages[:i + 1]])
        embedding = get_longformer_embedding(history)
        df_features = pd.DataFrame([embedding], columns=[f"feature_{i + 1}" for i in range(len(embedding))])
        df_features["nick"] = user_id
        pred_df = predict_model(best_model, data=df_features)
        pred = pred_df["prediction_label"].iloc[0]
        prob = pred_df["prediction_score"].iloc[0]
        pred_text = {v: k for k, v in mapping.items()}.get(pred, "Unknown")
        print(f"Messages: {i + 1}, Prediction: {pred_text}, Probability: {prob:.4f}")

def main():
    """
    Main function to process all user message data and track carbon emissions.
    """
    tracker = EmissionsTracker()
    tracker.start()
    if not os.path.isdir(directory):
        print("The specified directory does not exist.")
        return
    files = glob.glob(os.path.join(directory, "*.json"))
    if not files:
        print("No JSON files found.")
        return
    for file in files:
        user_id = os.path.splitext(os.path.basename(file))[0]
        try:
            with open(file, encoding="utf-8") as f:
                data = json.load(f)
            messages = [item.get("message", "") for item in data if item.get("message")]
            if messages:
                process_incremental_messages(user_id, messages)
            else:
                print(f"No messages found in {file}")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    emissions = tracker.stop()
    df_emissions = pd.read_csv('emissions.csv')
    print("\nFormatted emission data:")
    for index, row in df_emissions.iterrows():
        print(f'"duration": {row["duration"]},')
        print(f'"emissions": {row["emissions"]},')
        print(f'"cpu_energy": {row["cpu_energy"]},')
        print(f'"gpu_energy": {row["gpu_energy"]},')
        print(f'"ram_energy": {row["ram_energy"]},')
        print(f'"energy_consumed": {row["energy_consumed"]},')
        print(f'"cpu_count": {row["cpu_count"]},')
        print(f'"gpu_count": {row["gpu_count"]},')
        print(f'"cpu_model": "{row["cpu_model"]}",')
        print(f'"gpu_model": "{row["gpu_model"]}",')
        print(f'"ram_total_size": {row["ram_total_size"]},')
        print(f'"country_iso_code": "{row["country_iso_code"]}"')

if __name__ == "__main__":
    main()


