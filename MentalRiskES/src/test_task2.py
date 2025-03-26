#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
predicciones_desde_json.py

This script:
  1. Iterates through all .json files in a specified directory.
  2. For each file (name = user ID), reads the content (list of messages),
     and concatenates messages using " [SEP] " as a separator.
  3. Generates embeddings for each user using Longformer.
  4. Constructs a DataFrame with embeddings and the user identifier.
  5. Loads a trained PyCaret model and makes predictions.
  6. Prints the predictions and the associated probability for each user.
"""

import os
import glob
import json
import pandas as pd
import torch
from transformers import LongformerTokenizer, LongformerModel, AutoTokenizer, AutoModel
from pycaret.classification import load_model, predict_model
from codecarbon import EmissionsTracker

# Configure device and model settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model_name = "PlanTL-GOB-ES/longformer-base-4096-bne-es"
model_name = "dccuchile/bert-base-spanish-wwm-cased"

# Load tokenizer and model
# tokenizer = LongformerTokenizer.from_pretrained(model_name)
# model = LongformerModel.from_pretrained(model_name).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# Load the pre-trained PyCaret model
best_model = load_model(r"C:\Users\jeiso\Documents\Maestria\Semestre #1\Reto\MentalRiskES-2025\models\logistic_regressor_beto\lr_task2")

# Define the directory path
directory = r"C:\Users\jeiso\Documents\Maestria\Semestre #1\Reto\MentalRiskES-2025\data\trial\task2\subjects"

# Define category mapping
mapping = {
    'betting': 0,
    'trading': 1,
    'lootboxes': 2,
    'onlinegaming': 3
}


def get_longformer_embedding(text, max_length=512): # max_length --> BETO = 512, LongForme = 4096
    """
    Generate embeddings using Longformer for a given text.

    Args:
        text (str): The input text to encode.
        max_length (int): Maximum sequence length for tokenization.

    Returns:
        numpy.ndarray: The averaged embedding vector.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


def process_incremental_messages(user_id, messages):
    """
    Process user messages incrementally, generating embeddings and making predictions.

    Args:
        user_id (str): The identifier of the user.
        messages (list): List of message strings.
    """
    history = ""
    print(f"\nProcessing user {user_id}...")

    for i in range(len(messages)):
        history = " [SEP] ".join(messages[:i + 1])
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
    Main function to iterate over JSON files and process user messages.
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

    df_emissions = pd.read_csv('emissions\emissions.csv')

    print("\nDatos de emisiones formateados:")
    for index, row in df_emissions.iterrows():
        print(f'"duración": {row["duration"]},')
        print(f'"emisiones": {row["emissions"]},')
        print(f'"energía_cpu": {row["cpu_energy"]},')
        print(f'"energía_gpu": {row["gpu_energy"]},')
        print(f'"energía_ram": {row["ram_energy"]},')
        print(f'"energía_consumida": {row["energy_consumed"]},')
        print(f'"número_de_cpu": {row["cpu_count"]},')
        print(f'"cuenta_de_gpu": {row["gpu_count"]},')
        print(f'"cpu_model": "{row["cpu_model"]}",')
        print(f'"gpu_model": "{row["gpu_model"]}",')
        print(f'"tamaño_total_de_ram": {row["ram_total_size"]},')
        print(f'" código ISO del país": "{row["country_iso_code"]}"')

if __name__ == "__main__":
    main()


