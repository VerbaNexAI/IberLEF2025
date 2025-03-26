import os
import json
import random
import string
from openai import OpenAI

# --------------------- Configuration ---------------------
NUM_USERS = 70  # Number of users to generate
API_KEY = "sk-a4f13b8fa9804817addfed2268aca5d2"  # Replace with your DeepSeek API key
BASE_URL = "https://api.deepseek.com"  # Base URL for the DeepSeek API
MODEL = "deepseek-chat"  # Model to use as per the documentation

# Local file paths
PROMPT_PATH = r'C:\Users\jeiso\Documents\Maestria\Semestre #1\Reto\MentalRiskES-2025\data\prompt.txt'      # Path to the text file containing the prompt
OUTPUT_FOLDER = r'C:\Users\jeiso\Documents\Maestria\Semestre #1\Reto\MentalRiskES-2025\data\test_dataset\generated_users'   # Folder where JSON files will be saved

# --------------------- Listas de Juegos y Tópicos (10 elementos cada una) ---------------------
lista_juegos = [
    "Genshin Impact",
    "FIFA Ultimate Team",
    "NBA 2K",
    "Fortnite",
    "CS:GO",
    "Call of Duty",
    "League of Legends",
    "Valorant",
    "clash of clans",
    "overwatch",
    "free fire",
    "dota 2",
    "world of warcraft",
    "heroes of the storm",
    "Pokemon Go",
    "Apex Legends",
    "Rocket League",
    "Rainbow Six Siege",
    "GTA V"
]

lista_topicos = [
    "FOMO",
    "Grinding excesivo",
    "Eventos temporales",
    "Pérdida de control",
    "Manipulación psicológica",
    "Estrategias de marketing engañoso",
    "Apuestas virtuales",
    "Obsesión por las colecciones",
    "Inversión desmedida de tiempo",
    "Estrés por el juego competitivo",
    "Desgaste emocional",
    "Riesgo de endeudamiento",
    "Trastornos de comportamiento",
    "Tildeo",
    "frustración desmedida"

]

# --------------------- Preparar Carpeta de Salida ---------------------
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Carpeta '{OUTPUT_FOLDER}' creada.")

# --------------------- Leer el Prompt Base desde el Archivo ---------------------
try:
    with open(PROMPT_PATH, "r", encoding="utf-8") as file:
        base_prompt_text = file.read()
except Exception as e:
    print(f"Error al leer el archivo del prompt: {e}")
    exit(1)

# --------------------- Inicializar Cliente DeepSeek ---------------------
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# --------------------- Función para Generar una Cadena Aleatoria ---------------------
def generar_cadena_aleatoria(longitud=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=longitud))


# --------------------- Generar Usuarios y Guardar Respuestas ---------------------
for i in range(1, NUM_USERS + 1):
    # Generar un identificador único aleatorio para variar el prompt
    identificador_unico = f"[Iteración {i}: {generar_cadena_aleatoria()}]"

    # Seleccionar aleatoriamente un juego y un tópico
    juego_seleccionado = random.choice(lista_juegos)
    topico_seleccionado = random.choice(lista_topicos)

    # Crear un prompt único para cada iteración agregando el identificador, el juego y el tópico
    unique_prompt_text = (
            base_prompt_text +
            "\n\n" + identificador_unico +
            f"\nJuego a mencionar: {juego_seleccionado}" +
            f"\nTópico a desarrollar: {topico_seleccionado}"
    )

    # Variar ligeramente la temperatura para aumentar la aleatoriedad
    temperatura = 0.7 + random.uniform(-0.1, 0.1)

    # Realizar la solicitud a la API de DeepSeek
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un generador de datos sintéticos para entrenar un modelo que detecta problemas relacionados con el gambling en videojuegos. "
                    "Genera una conversación para un único usuario en formato JSON. El output debe ser una lista de objetos, "
                    "cada uno con exactamente las claves 'message', 'date' y 'platform'. No incluyas otras claves ni metadatos. "
                    "El campo 'date' debe estar en el formato DD.MM.YYYY HH:MM:SS UTC+01:00, y 'platform' debe ser 'Telegram', 'Reddit' o 'Twitch'."
                )
            },
            {"role": "user", "content": unique_prompt_text}
        ],
        stream=False,
        temperature=temperatura
    )

    # Recuperar el contenido generado en la respuesta.
    content = response.choices[0].message.content

    # Eliminar delimitadores Markdown si están presentes (```json ... ```)
    if content.startswith("```json"):
        content = content.replace("```json", "").replace("```", "").strip()

    # Intentar interpretar el contenido como JSON.
    try:
        json_data = json.loads(content)
    except json.JSONDecodeError:
        json_data = {"generated": content}

    # Guardar la respuesta en un archivo con nombre secuencial (por ejemplo, user0001.json, user0002.json, ...)
    filename = os.path.join(OUTPUT_FOLDER, f"user{i:03d}.json")
    with open(filename, "w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, indent=4, ensure_ascii=False)
    print(f"Usuario {i} guardado en: {filename}")