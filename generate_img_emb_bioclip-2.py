import torch
import json
import os
import faiss
import numpy as np
import open_clip
from PIL import Image
from tqdm import tqdm

# 1. Ініціалізація BioCLIP-2
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# BioCLIP-2 зазвичай доступний під цим ID
MODEL_ID = 'hf-hub:imageomics/bioclip-2'

print(f"Loading BioCLIP-2 on {DEVICE}...")
model, _, preprocess_val = open_clip.create_model_and_transforms(MODEL_ID)
model.to(DEVICE)
model.eval()


def generate_bioclip2_embeddings():
    # Шляхи
    base_path = "/Users/stankutnyk/Downloads/plantnet_300K/images_test"
    names_json = "/Users/stankutnyk/Downloads/plantnet_300K/plantnet300K_species_names.json"
    meta_json = "/Users/stankutnyk/Downloads/plantnet_300K/plantnet300K_metadata.json"
    desc_json = "test_species_descriptions.json"

    with open(names_json, 'r') as f:
        id_to_name = json.load(f)
    with open(meta_json, 'r') as f:
        metadata = json.load(f)
    with open(desc_json, 'r') as f:
        descriptions_data = json.load(f)

    valid_names = {data['latin_name'] for data in descriptions_data.values()}

    tasks = []
    for filename, info in metadata.items():
        if info['split'] != 'test': continue
        name = id_to_name.get(info['species_id'])
        if name in valid_names:
            img_path = os.path.join(base_path, info['species_id'], f"{filename}.jpg")
            if os.path.exists(img_path):
                tasks.append((img_path, name, filename))

    print(f"Processing {len(tasks)} images with BioCLIP-2...")

    image_vectors = []
    new_img_metadata = []
    batch_size = 64

    for i in tqdm(range(0, len(tasks), batch_size)):
        batch = tasks[i:i + batch_size]
        imgs = []
        for path, name, fname in batch:
            imgs.append(preprocess_val(Image.open(path).convert("RGB")))

        batch_tensor = torch.stack(imgs).to(DEVICE)

        with torch.no_grad():
            features = model.encode_image(batch_tensor)
            features /= features.norm(dim=-1, keepdim=True)
            image_vectors.append(features.cpu().numpy())

        for _, name, fname in batch:
            new_img_metadata.append({"filename": fname, "latin_name": name})

    # Збереження
    vectors_np = np.vstack(image_vectors).astype('float32')
    dim = vectors_np.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(vectors_np)

    faiss.write_index(index, "bioclip2_plantnet_test.index")
    with open("bioclip2_img_metadata.json", "w") as f:
        json.dump(new_img_metadata, f, indent=4)

    print(f"Success! Created bioclip2_plantnet_test.index with {index.ntotal} vectors.")


if __name__ == '__main__':
    generate_bioclip2_embeddings()