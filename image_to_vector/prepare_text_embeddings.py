import torch
import pandas as pd
import json
import faiss
import numpy as np
import open_clip
from tqdm import tqdm
import os

def generate_text_embeddings(
    model, 
    tokenizer, 
    device, 
    parquet_path, 
    output_dir, 
    version_name, 
    string_template="{text}", 
    taxonomical_mode=False, 
    batch_size=512
):
    """
    Generates text embeddings for taxa data and saves them to a FAISS index.
    Also records the template used in text_templates_versions.json.
    """
    print(f"Reading taxonomic data from {parquet_path}...")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Could not find {parquet_path}")

    df = pd.read_parquet(parquet_path)
    
    # We always need a unique identifier for the label, usually `species` or `name`
    label_col = 'name' if 'name' in df.columns else 'species'
    if label_col not in df.columns:
        raise ValueError(f"Neither 'name' nor 'species' column found in {parquet_path}")

    # Drop rows where the label is missing
    df = df.dropna(subset=[label_col])
    
    # We want one text representation per unique label
    df_unique = df.drop_duplicates(subset=[label_col])
    
    labels_list = []
    text_to_encode_list = []
    
    # Define taxonomy columns in order
    tax_cols = ['kingdom', 'phylum', 'class', 'order', 'family', 'species']
    
    # We might want to sample text to save to versions file
    sample_text = ""
    max_parts = -1

    for _, row in df_unique.iterrows():
        for organ in ["flower", "leaf", "fruit", "bark"]:
            if organ != "bark":
                organ_string_template = string_template.replace("a photo of", f"a photo of {organ} of a")
            else:
                organ_string_template = string_template.replace("a photo of", f"a photo of the {organ} of a")
            label = row[label_col]
            labels_list.append(label)

            if taxonomical_mode:
                parts = []
                for col in tax_cols:
                    if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
                        parts.append(str(row[col]).strip())
                # If nothing was found, at least use the label
                base_text = " ".join(parts) if parts else str(label)

                final_text = organ_string_template.replace("{text}", base_text).strip()

                if len(parts) > max_parts:
                    max_parts = len(parts)
                    sample_text = final_text
            else:
                base_text = str(label)
                final_text = organ_string_template.replace("{text}", base_text).strip()

                if not sample_text:
                    sample_text = final_text

            text_to_encode_list.append(final_text)

    if len(text_to_encode_list) == 0:
        print("No labels found to encode.")
        return None, None

    print(f"Sample text to encode: '{sample_text}'")
    
    all_text_features = []
    print(f"Encoding {len(text_to_encode_list)} items in batches of {batch_size}...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(text_to_encode_list), batch_size)):
            batch_texts = text_to_encode_list[i:i + batch_size]
            
            # Tokenize and push to device
            text_inputs = tokenizer(batch_texts).to(device)

            # Encode text
            text_features = model.encode_text(text_inputs)
            
            # Normalize embeddings
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Move to CPU explicitly
            all_text_features.append(text_features.cpu().numpy().astype('float32'))

    species_text_features = np.vstack(all_text_features)
    
    # Output paths
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, f"{version_name}.index")
    labels_path = os.path.join(output_dir, f"{version_name}_labels.json")
    versions_path = os.path.join(output_dir, "text_templates_versions.json")
    
    print("Saving features to FAISS index...")
    with open(labels_path, 'w') as f:
        json.dump(labels_list, f, indent=4)
        
    dim = species_text_features.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(species_text_features)
    faiss.write_index(index, index_path)

    # Update versions json
    versions = {}
    if os.path.exists(versions_path):
        with open(versions_path, 'r') as f:
            try:
                versions = json.load(f)
            except json.JSONDecodeError:
                pass
                
    versions[version_name] = sample_text
    with open(versions_path, 'w') as f:
        json.dump(versions, f, indent=4)

    print(f"Success! Created {index_path} with {index.ntotal} vectors.")
    return index_path, labels_path

def generate_bioclip2_text_embeddings():
    DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 512
    # BioCLIP-2 is available under this ID
    MODEL_ID = 'hf-hub:imageomics/bioclip-2'
    print(f"Loading BioCLIP-2 on {DEVICE}...")

    # Load model and tokenizer
    model, _, _ = open_clip.create_model_and_transforms(MODEL_ID)
    tokenizer = open_clip.get_tokenizer(MODEL_ID)
    model.to(DEVICE)
    model.eval()

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(base_dir, "../data"))
    parquet_path = os.path.join(data_dir, "taxa.parquet")

    generate_text_embeddings(
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        parquet_path=parquet_path,
        output_dir=data_dir,
        version_name="bioclip2_text_v4",
        string_template="a photo of {text}",
        taxonomical_mode=True,
        batch_size=BATCH_SIZE
    )

if __name__ == '__main__':
    generate_bioclip2_text_embeddings()
