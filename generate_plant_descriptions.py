import os
import json
import time
from google import genai
from google.genai import types
from tqdm import tqdm
import dotenv

dotenv.load_dotenv(".env.local")

# 1. Налаштування Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-2.5-flash"


def generate_description(species_name):
    # Використовуємо системну інструкцію для стабільності ролі
    system_instr = """You are a professional botanist. Provide a concise, visual description 
    of the plant species for a CV model. Focus on:
    1. Growth habit (tree, shrub, herb).
    2. Specific leaf shape, color, and margin.
    3. Flower morphology and typical color.
    Use purely visual terms. Avoid history, medical uses, or geography. 
    Strictly limit to 40-50 words."""

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            config=types.GenerateContentConfig(
                system_instruction=system_instr,
                temperature=0.0  # Для ідентичних результатів
            ),
            contents=f"Describe this species: {species_name}"
        )
        return response.text.strip()
    except Exception as e:
        print(f"\nError for {species_name}: {e}")
        return None

def main():
    # Шляхи до файлів
    prefix = "/Users/stankutnyk/Downloads/"
    test_images_path = prefix+"plantnet_300k/images_test"
    species_names_path = prefix+"plantnet_300k/plantnet300K_species_names.json"
    output_file = "test_species_descriptions.json"

    # Завантажуємо мапінг ID -> Latin Name
    with open(species_names_path, 'r') as f:
        id_to_name = json.load(f)

    # Визначаємо унікальні ID видів, які є в папці images_test
    test_species_ids = [d for d in os.listdir(test_images_path) if os.path.isdir(os.path.join(test_images_path, d))]

    print(f"Found {len(test_species_ids)} unique species in images_test.")

    descriptions = {}

    # Якщо файл вже існує, завантажуємо, щоб не витрачати квоту API
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            descriptions = json.load(f)

    for s_id in tqdm(test_species_ids):
        if s_id in descriptions:
            continue  # Пропускаємо вже оброблені

        latin_name = id_to_name.get(s_id)
        if not latin_name:
            continue

        desc = generate_description(latin_name)
        if desc:
            descriptions[s_id] = {
                "latin_name": latin_name,
                "description": desc
            }
            # Зберігаємо після кожного запиту про всяк випадок
            with open(output_file, 'w') as f:
                json.dump(descriptions, f, indent=4)

        # Обмеження Gemini Free Tier (15 RPM для flash). Робимо паузу.
        time.sleep(4)

    print(f"Done! Descriptions saved to {output_file}")


if __name__ == '__main__':
    main()