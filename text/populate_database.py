import pandas as pd
import wikipediaapi
import sqlite3
import time
from tqdm import tqdm

# Initialize Wikipedia API
wiki = wikipediaapi.Wikipedia(
    user_agent="BotanicAI_App_Project (contact@example.com)",
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)


def fetch_wiki_data(title):
    try:
        page = wiki.page(title)
        if page.exists():
            return {
                "summary": page.summary,
                "full_text": page.text,  # This grabs all sections
                "url": page.fullurl
            }
    except Exception as e:
        print(f"\n[ERROR] Fetching {title}: {e}")
    return None


def fill_database(csv_path):
    df = pd.read_csv(csv_path)
    conn = sqlite3.connect('plants_knowledge.db')
    cursor = conn.cursor()

    # Updated Table Schema
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS plant_info
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY,
                       level
                       TEXT,
                       name
                       TEXT
                       UNIQUE,
                       summary
                       TEXT,
                       full_text
                       TEXT,
                       url
                       TEXT,
                       source
                       TEXT
                   )
                   ''')

    # Load existing items to skip
    cursor.execute("SELECT name FROM plant_info")
    processed_items = {row[0] for row in cursor.fetchall()}
    print(f"Skipping {len(processed_items)} already processed items.")

    # Iterate through unique names in the hierarchy
    # We flatten the levels to get a unique list of strings to fetch
    all_names = pd.concat([df['species'], df['genus'], df['family']]).unique()
    all_names = [n for n in all_names if pd.notna(n) and n not in processed_items]

    print(f"Starting fetch for {len(all_names)} new items...")

    for name in tqdm(all_names, desc="Downloading Wiki Data"):
        # Determine the level for logging/metadata (optional but helpful)
        # For simplicity, we just fetch; we can map levels back via the CSV later
        data = fetch_wiki_data(name)

        if data:
            try:
                # We don't know the level here easily without re-checking CSV,
                # so we can leave it as 'pending' or look it up.
                # Let's do a quick lookup in the DF to find its level.
                if name in df['species'].values:
                    level = 'species'
                elif name in df['genus'].values:
                    level = 'genus'
                else:
                    level = 'family'

                cursor.execute('''
                               INSERT
                               OR IGNORE INTO plant_info (level, name, summary, full_text, url, source)
                    VALUES (?, ?, ?, ?, ?, ?)
                               ''', (level, name, data['summary'], data['full_text'], data['url'], 'Wikipedia'))

                conn.commit()
                processed_items.add(name)
            except Exception as e:
                print(f"Error saving {name}: {e}")

        time.sleep(0.1) # Respect Rate Limits

    conn.close()
    print("\nDatabase is now a rich offline repository!")

if __name__ == "__main__":
    fill_database('../data/PT_species.csv')