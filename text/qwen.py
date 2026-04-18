import sqlite3
import os
from mlx_lm import load, generate

def retrieve_context(species_name, genus_name, family_name):
    """
    Retrieves botanical context for a plant from the local SQLite database.

    Attempts a hierarchical fallback search: first looking for the specific
    species context, then the genus context, and finally the family context.
    This ensures that even if specific species data is missing, general
    information about its broader taxonomy can be provided to the LLM.

    Args:
        species_name (str): The scientific name of the species.
        genus_name (str): The scientific name of the genus.
        family_name (str): The scientific name of the family.

    Returns:
        str: The retrieved text context, or a default message if no data is found locally.
    """
    # Use absolute path to DB to ensure it works from anywhere
    base_dir = os.path.dirname(__file__)
    db_path = os.path.join(base_dir, './plants_knowledge.db')
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Try to find the Species
    cursor.execute("SELECT full_text, url FROM plant_info WHERE name = ?", (species_name,))
    result = cursor.fetchone()
    if result:
        return f"Specific info for {species_name}: {result[0]}", result[1], "Species"

    # 2. Fallback to Genus
    cursor.execute("SELECT full_text, url FROM plant_info WHERE name = ?", (genus_name,))
    result = cursor.fetchone()
    if result:
        return f"Info for the genus {genus_name} (details for {species_name} are limited): {result[0]}", result[1], "Genus"

    # 3. Fallback to Family
    cursor.execute("SELECT full_text, url FROM plant_info WHERE name = ?", (family_name,))
    result = cursor.fetchone()
    if result:
        return f"General info for the {family_name} family (details for {species_name} are limited): {result[0]}", result[1], "Family"

    return "No local data found.", None, "None"

# Load model globally to avoid reloading on every request
# Path to Qwen (you can use "Qwen/Qwen2.5-1.5B-Instruct" directly with load())
try:
    model, tokenizer = load("Qwen/Qwen3.5-2B")
except Exception as e:
    print(f"Warning: Failed to load Qwen model: {e}")
    model, tokenizer = None, None


def ask_botanist(species, genus, family, audience="Pupil"):
    """
    Acts as a Retrieval-Augmented Generation (RAG) interface to query a local Qwen LLM.

    Retrieves context regarding the plant taxonomy from a local database and
    constructs a strict prompt instructing the LLM to act as a botanical expert.
    The LLM is constrained to output exactly 3 amazing facts focused on culture,
    history, or biology, formatted as a JSON array of strings under 60 words total.

    Args:
        species (str): The scientific name of the species.
        genus (str): The scientific name of the genus.
        family (str): The scientific name of the family.

    Returns:
        str: A JSON-formatted string array of 3 extracted facts, or an error message 
             if the model fails to load.
    """
    if model is None:
        return "Qwen model is not loaded.", None, "None"

    # Get data from our local RAG system
    context, source_url, match_level = retrieve_context(species, genus, family)
    
    if audience == "Pupil":
        sys_msg = "You are a botanical expert that works with pupils."
        user_msg = f"Extract 3 or less bullet-point facts about {species} from this text that will be particularly interesting to your young audience.\nMind of what your young audience knows and what they don't. Keep the language simple and engaging. \n"
    else:
        sys_msg = "You are a botanical expert that works with university students."
        user_msg = f"Extract 3 or less bullet-point facts about {species} from this text that will be particularly interesting to university students. You want to share very complex and scientific information.\n"

    messages = [
        {
            "role": "system",
            "content": sys_msg
        },
        {
            "role": "user",
            "content": user_msg +
                       f"Rules:\n"
                       f"1. Use only the provided context. For each fact provide a quote from source. NEVER include information that can not be directly proven from the source\n"
                       f"2. If there are not enough information for 3 facts, reduce the quantity but do not compromise quality.\n"
                       f"3. Focus on culture, history, or unique biology.\n"
                       f"4. Total facts must be under 60 words.\n"
                       f"5. Output format: [{{\"quote\": \"...\", \"fact\": \"...\"}}, ...]\n"
                       f"Context: {context}"
        }
    ]


    # Qwen 3.5 uses the <|im_start|> and <|im_end|> tokens
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 3. Generate the response
    response = generate(
        model,
        tokenizer,
        prompt=text,
        max_tokens=5120,
    )
    # response = generate(model, tokenizer, prompt=prompt, max_tokens=300)
    return response, source_url, match_level

# Example Usage:
if __name__ == '__main__':
    print(ask_botanist("Abelia chinensis", "Centaurium", "Asteraceae"))
