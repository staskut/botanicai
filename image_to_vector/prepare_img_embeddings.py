import torch
import json
import os
import faiss
import numpy as np
import open_clip
from PIL import Image
from tqdm import tqdm

def generate_bioclip2_embeddings():
    BATCH_SIZE = 128
    DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    # BioCLIP-2 is available under this ID
    MODEL_ID = 'hf-hub:imageomics/bioclip-2'

    print(f"Loading BioCLIP-2 on {DEVICE}...")
    model, _, preprocess_val = open_clip.create_model_and_transforms(MODEL_ID)
    model.to(DEVICE)
    model.eval()

    # Paths
    base_dir = "../data/plantnet_300K"
    names_json = os.path.join(base_dir, "plantnet300K_species_names.json")
    meta_json = os.path.join(base_dir, "plantnet300K_metadata.json")

    with open(names_json, 'r') as f:
        id_to_name = json.load(f)
    with open(meta_json, 'r') as f:
        metadata = json.load(f)

    print("Checking for existing checkpoints...")
    temp_dir = "../data/bioclip2_embeddings_temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    processed_files = set()
    part_idx = 0
    
    # Find existing parts
    for fname in os.listdir(temp_dir):
        if fname.startswith("metadata_part_") and fname.endswith(".json"):
            part_num_str = fname.replace("metadata_part_", "").replace(".json", "")
            if part_num_str.isdigit():
                part_num = int(part_num_str)
                part_idx = max(part_idx, part_num + 1)
                
                with open(os.path.join(temp_dir, fname), "r") as mf:
                    part_meta = json.load(mf)
                    for item in part_meta:
                        processed_files.add(item["image_path"])
                        
    print(f"Found {len(processed_files)} previously processed images.")

    tasks = []

    print("Collecting images...")
    # Go through data/plantnet_300K metadata keeping track of split and class id
    for filename, info in metadata.items():
        split = info['split']
        class_id = info['species_id']
        name = id_to_name.get(class_id, "Unknown")
        
        # Image paths are typically like: images_train/1355868/1000003.jpg
        img_path = os.path.join(base_dir, f"images_{split}", str(class_id), f"{filename}.jpg")
        
        if os.path.exists(img_path) and img_path not in processed_files:
            tasks.append((img_path, filename, split, class_id, name))

    print(f"Processing {len(tasks)} images with BioCLIP-2...")
    if len(tasks) == 0:
        print("No images found. Please check paths and data structure.")
        return

    image_vectors = []
    new_img_metadata = []
    batch_size = BATCH_SIZE
    save_interval = 10000  # Number of images to process before saving a checkpoint

    for i in tqdm(range(0, len(tasks), batch_size)):
        batch = tasks[i:i + batch_size]
        imgs = []
        for path, fname, split, class_id, name in batch:
            imgs.append(preprocess_val(Image.open(path).convert("RGB")))

        batch_tensor = torch.stack(imgs).to(DEVICE)

        with torch.no_grad():
            features = model.encode_image(batch_tensor)
            features /= features.norm(dim=-1, keepdim=True)
            image_vectors.append(features.cpu().numpy())

        for path, fname, split, class_id, name in batch:
            new_img_metadata.append({
                "filename": fname,
                "split": split,
                "class_id": class_id,
                "latin_name": name,
                "image_path": path
            })
            
        # Check if we should save a checkpoint
        if len(new_img_metadata) >= save_interval or (i + len(batch)) >= len(tasks):
            part_vectors = np.vstack(image_vectors).astype('float32')
            np.save(os.path.join(temp_dir, f"vectors_part_{part_idx}.npy"), part_vectors)
            with open(os.path.join(temp_dir, f"metadata_part_{part_idx}.json"), "w") as f:
                json.dump(new_img_metadata, f, indent=4)
            
            part_idx += 1
            image_vectors = []
            new_img_metadata = []

    # Final assembly
    print("Building final representations from parts...")
    all_metadata = []
    index = None

    for p in range(part_idx):
        part_meta_path = os.path.join(temp_dir, f"metadata_part_{p}.json")
        part_vec_path = os.path.join(temp_dir, f"vectors_part_{p}.npy")
        
        if not os.path.exists(part_vec_path) or not os.path.exists(part_meta_path):
            continue
            
        with open(part_meta_path, "r") as f:
            all_metadata.extend(json.load(f))
            
        part_vectors = np.load(part_vec_path)
        
        if index is None:
            dim = part_vectors.shape[1]
            index = faiss.IndexFlatIP(dim)
            
        index.add(part_vectors)

    os.makedirs("../data", exist_ok=True)
    index_path = "../data/bioclip2_plantnet300k.index"
    meta_path = "../data/bioclip2_plantnet300k_metadata.json"

    faiss.write_index(index, index_path)
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=4)

    print(f"Success! Created {index_path} with {index.ntotal} vectors.")

if __name__ == '__main__':
    generate_bioclip2_embeddings()
