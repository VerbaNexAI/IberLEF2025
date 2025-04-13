import requests
from requests.adapters import HTTPAdapter, Retry
from typing import List, Dict
import json
import os
import pandas as pd
from codecarbon import EmissionsTracker
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel, LongformerTokenizer, LongformerModel
from pycaret.classification import load_model, predict_model
import re
import emoji
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import time

# Configuración inicial
URL = "http://s3-ceatic.ujaen.es:8036"  # Se eliminó el espacio inicial
TOKEN = "258d7e20e4b175baeba402128baa9155"
ENDPOINT_GET_MESSAGES_TRIAL = URL + "/{TASK}/getmessages_trial/{TOKEN}"
ENDPOINT_SUBMIT_DECISIONS_TRIAL = URL + "/{TASK}/submit_trial/{TOKEN}/{RUN}"
BASE_DIR = Path(__file__).resolve().parent.parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DatasetColumnTransformer:
    """Clase para limpieza y preprocesamiento de texto"""

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
        if not isinstance(text, str):
            return ""
        text = text.lower()  # Normalizar a minúsculas
        text = self.url_pattern.sub("[URL]", text)  # Reemplazar URLs
        text = self.email_pattern.sub("[EMAIL]", text)  # Reemplazar emails
        text = self.user_pattern.sub("[USER]", text)  # Reemplazar menciones
        text = self.html_pattern.sub("", text)  # Eliminar HTML
        text = self.markdown_pattern.sub("", text)  # Eliminar markdown
        text = emoji.demojize(text, language='es')  # Convertir emojis a texto
        text = re.sub(r':(\w+):', r' \1 ', text)  # Normalizar emojis convertidos
        text = re.sub(r'\s+', ' ', text).strip()  # Eliminar espacios múltiples
        text = self.double_quotes_pattern.sub('"', text)  # Normalizar comillas dobles
        text = self.single_quotes_pattern.sub("'", text)   # Normalizar comillas simples
        text = self.punctuation_pattern.sub(r'\1\1\1', text)  # Reducir puntuación excesiva
        words = word_tokenize(text)  # Tokenización
        words = [self.lemmatizer.lemmatize(word) for word in words]  # Lematización
        return " ".join(words)


class Client:
    """Cliente unificado para manejar ambas tareas simultáneamente"""

    def __init__(self, token: str, number_of_runs: int, tracker: EmissionsTracker):
        self.token = token
        self.number_of_runs = number_of_runs
        self.tracker = tracker
        self.user_histories = {}  # Almacena historial por usuario
        self.relevant_cols = [  # Columnas para reporte de emisiones
            'duration', 'emissions', 'cpu_energy', 'gpu_energy',
            'ram_energy', 'energy_consumed', 'cpu_count', 'gpu_count',
            'cpu_model', 'gpu_model', 'ram_total_size', 'country_iso_code'
        ]

        # Inicializar componentes de procesamiento de texto
        self.text_cleaner = DatasetColumnTransformer()

        # Cargar modelos para Task 1 (Detección de riesgo)
        self.tokenizer_task1 = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
        self.model_task1 = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased").to(device)
        self.model_task1.eval()
        self.best_model_task1 = load_model(str(BASE_DIR / "models" / "logistic_regressor_beto" / "lr_task1"))

        # Cargar modelos para Task 2 (Clasificación de categorías)
        self.tokenizer_task2 = LongformerTokenizer.from_pretrained("PlanTL-GOB-ES/longformer-base-4096-bne-es")
        self.model_task2 = LongformerModel.from_pretrained("PlanTL-GOB-ES/longformer-base-4096-bne-es").to(device)
        self.model_task2.eval()
        self.best_model_task2 = load_model(str(BASE_DIR / "models" / "random_forest_long" / "rf_task2"))

        # Mapeos de etiquetas
        self.risk_mapping = {'low risk': 0, 'high risk': 1}
        self.category_mapping = {
            'betting': 0,
            'trading': 1,
            'lootboxes': 2,
            'onlinegaming': 3
        }

    def get_embeddings(self, text: str, task: str):
        """Genera embeddings según el modelo de cada tarea"""
        if task == "task1":
            inputs = self.tokenizer_task1(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        else:
            inputs = self.tokenizer_task2(text, return_tensors="pt", truncation=True, padding="max_length", max_length=4096)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model_task1(**inputs) if task == "task1" else self.model_task2(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    def get_messages(self, retries: int, backoff: float) -> List[Dict]:
        """Obtiene mensajes de ambas tareas desde el servidor"""
        messages = []
        for task in ["task1", "task2"]:
            session = requests.Session()
            retry_strategy = Retry(total=retries, backoff_factor=backoff, status_forcelist=[500, 502, 503, 504])
            session.mount('https://', HTTPAdapter(max_retries=retry_strategy))
            response = session.get(ENDPOINT_GET_MESSAGES_TRIAL.format(TASK=task, TOKEN=self.token))
            if response.status_code == 200:
                messages.extend(json.loads(response.content))
            else:
                print(f"Error obteniendo mensajes de {task}: {response.text}")
        return messages

    def process_user_messages(self, user_nick: str):
        """Procesa el historial de mensajes de un usuario para ambas tareas"""
        messages = self.user_histories.get(user_nick, [])
        cleaned_messages = [self.text_cleaner.clean_text(msg) for msg in messages]
        history = " [SEP] ".join(cleaned_messages)

        # Procesar Task 1 (Riesgo)
        embedding_task1 = self.get_embeddings(history, "task1")
        df_task1 = pd.DataFrame([embedding_task1],
                                columns=[f"feature_{i + 1}" for i in range(embedding_task1.shape[0])])
        df_task1["nick"] = user_nick
        risk_pred = int(predict_model(self.best_model_task1, data=df_task1)["prediction_label"].iloc[0])

        # Procesar Task 2 (Categoría)
        embedding_task2 = self.get_embeddings(history, "task2")
        df_task2 = pd.DataFrame([embedding_task2],
                                columns=[f"feature_{i + 1}" for i in range(embedding_task2.shape[0])])
        df_task2["nick"] = user_nick
        category_pred = int(predict_model(self.best_model_task2, data=df_task2)["prediction_label"].iloc[0])

        return risk_pred, category_pred

    def submit_decisions(self, messages: List[Dict], emissions: Dict, retries: int, backoff: float):
        """Envía predicciones para ambas tareas al servidor"""
        decisions = {f"run{run}": {"task1": {}, "task2": {}} for run in range(self.number_of_runs)}

        # Generar predicciones para todos los usuarios
        for msg in messages:
            user_nick = msg["nick"]
            risk_pred, category_pred = self.process_user_messages(user_nick)
            for run in range(self.number_of_runs):
                decisions[f"run{run}"]["task1"][user_nick] = risk_pred
                decisions[f"run{run}"]["task2"][user_nick] = list(self.category_mapping.keys())[int(category_pred)]

        # Configurar sesión para las solicitudes POST
        session = requests.Session()
        retry_strategy = Retry(total=retries, backoff_factor=backoff, status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retry_strategy))

        for run in range(self.number_of_runs):
            # Preparar payload para Task1 y Task2
            payload_task1 = {
                "predictions": decisions[f"run{run}"]["task1"],
                "emissions": emissions
            }
            payload_task2 = {
                "predictions": decisions[f"run{run}"]["task1"],
                "types": decisions[f"run{run}"]["task2"],
                "emissions": emissions
            }

            print(f"\n{'=' * 30} Task1 Payload (Run {run}) {'=' * 30}")
            print(json.dumps(payload_task1, indent=2))
            print(f"\n{'=' * 30} Task2 Payload (Run {run}) {'=' * 30}")
            print(json.dumps(payload_task2, indent=2))
            print("=" * 80 + "\n")

            for task, payload in [("task1", payload_task1), ("task2", payload_task2)]:
                # Serializamos el payload y lo enviamos dentro de una lista
                response = session.post(
                    ENDPOINT_SUBMIT_DECISIONS_TRIAL.format(TASK=task, TOKEN=self.token, RUN=run),
                    json=[json.dumps(payload)]
                )
                print(f"POST - {task} - Run {run} - Status: {response.status_code}")

    def run(self, retries: int, backoff: float):
        """
        Flujo principal de ejecución:
         - Se asegura de enviar las 3 runs de Task1 y las 3 runs de Task2 para la ronda actual
         - Solo se hace un nuevo GET cuando se han enviado las 6 solicitudes de POST para la ronda
        """
        processed_rounds = set()
        while True:
            messages = self.get_messages(retries, backoff)
            if not messages:
                print("No hay más mensajes para procesar")
                break

            # Asumimos que todos los mensajes de la ronda tienen el mismo número de ronda
            current_round = messages[0]["round"]
            if current_round in processed_rounds:
                # Ya se procesó esta ronda; espera un poco antes de volver a preguntar
                time.sleep(1)
                continue

            print(f"----------------------- Processing round {current_round} -----------------------")
            processed_rounds.add(current_round)

            # Actualizar historiales
            for msg in messages:
                user = msg["nick"]
                self.user_histories.setdefault(user, []).append(msg["message"])

            # Medir emisiones
            self.tracker.start()
            _ = self.tracker.stop()  # Finaliza la medición (demostración)
            df_emissions = pd.read_csv("emissions/emissions.csv")
            emissions_data = df_emissions.iloc[-1][self.relevant_cols].to_dict()

            # Enviar las 6 solicitudes de POST (3 runs para Task1 y 3 runs para Task2)
            self.submit_decisions(messages, emissions_data, retries, backoff)

            # Esperar un momento antes de solicitar la siguiente ronda
            time.sleep(1)


def initialize_client(token: str):
    """Inicializa el cliente con configuración de emisiones"""
    tracker = EmissionsTracker(
        save_to_file=True,
        log_level="WARNING",
        tracking_mode="process",
        output_dir=".",
        allow_multiple_runs=True
    )
    return Client(token, number_of_runs=3, tracker=tracker)


if __name__ == '__main__':
    client = initialize_client(TOKEN)
    client.run(retries=5, backoff=0.1)



