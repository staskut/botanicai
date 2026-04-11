import torch
import pandas as pd
import json
import faiss
import open_clip
from tqdm import tqdm
import os
from PIL import Image

# Added CUDA support so it doesn't default to CPU on Windows/Linux machines with NVIDIA GPUs
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_ID = 'hf-hub:imageomics/bioclip-2'

# Global variables for caching
_model = None
_tokenizer = None
_transform = None
_species_text_index = None
_labels_map = None
_verification_index = None
_verification_labels_map = None


def get_model():
    global _model, _tokenizer, _transform
    if _model is None:
        _model, _, _transform = open_clip.create_model_and_transforms(MODEL_ID)
        _tokenizer = open_clip.get_tokenizer(MODEL_ID)
        _model.to(DEVICE)
        _model.eval()
    return _model, _tokenizer, _transform


def get_text_embeddings_and_labels_map(version_name):
    global _species_text_index, _labels_map
    if _species_text_index is not None:
        return _species_text_index, _labels_map

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(base_dir, '../data'))
    
    index_path = os.path.join(data_dir, f'{version_name}.index')
    labels_path = os.path.join(data_dir, f'{version_name}_labels.json')

    if not (os.path.exists(index_path) and os.path.exists(labels_path)):
        raise FileNotFoundError(
            f"FAISS index or labels not found for version '{version_name}'. "
            f"Please run prepare_text_embeddings.py first to generate them."
        )

    print(f"Loading pre-computed species features from FAISS ({version_name})...")
    with open(labels_path, 'r') as f:
        _labels_map = json.load(f)

    _species_text_index = faiss.read_index(index_path)
    return _species_text_index, _labels_map


def get_verification_embeddings_and_labels_map(version_name="verification"):
    global _verification_index, _verification_labels_map
    if _verification_index is not None:
        return _verification_index, _verification_labels_map

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(base_dir, '../data'))
    
    index_path = os.path.join(data_dir, f'{version_name}.index')
    labels_path = os.path.join(data_dir, f'{version_name}_labels.json')

    if not (os.path.exists(index_path) and os.path.exists(labels_path)):
        return None, None

    print(f"Loading pre-computed verification features from FAISS ({version_name})...")
    with open(labels_path, 'r') as f:
        _verification_labels_map = json.load(f)

    _verification_index = faiss.read_index(index_path)
    return _verification_index, _verification_labels_map


def predict_species(image, version_name="bioclip2_text_v2"):
    """
    Predicts the species family, genus, and species name for a given PIL Image.
    Returns: (species_name, genus_name, family_name) or None if prediction fails
    """
    model, _, transform = get_model()
    species_features, unique_labels = get_text_embeddings_and_labels_map(version_name)
    verif_index, verif_labels = get_verification_embeddings_and_labels_map("verification")

    image_input = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features_np = image_features.cpu().numpy().astype('float32')

    scores, indices = _species_text_index.search(image_features_np, k=4)
    abs_diff_1 = scores[0, 0] - scores[0, 1]
    abs_diff_3 = scores[0, 2] - scores[0, 3]
    top_1_conf, top_3_conf, genus_conf, family_conf = _estimate_confidence(abs_diff_1, abs_diff_3)
    response = {
        "top-3": [unique_labels[indices[0, 0]], unique_labels[indices[0, 1]], unique_labels[indices[0, 2]]],
        "top-1_confidence": top_1_conf,
        "top-3_confidence": top_3_conf,
        "genus_confidence": genus_conf,
        "family_confidence": family_conf,
        "verification_label": None
    }
    
    if verif_index is not None and verif_labels is not None:
        v_scores, v_indices = verif_index.search(image_features_np, k=1)
        response["verification_label"] = verif_labels[v_indices[0, 0]]
        
    return response


def _estimate_confidence(abs_diff_1, abs_diff_3):
    top_1_table = pd.read_csv("./data/top_1_diff_decile.csv")
    top_3_table = pd.read_csv("./data/top_3_diff_decile.csv")
    top_1_mask = (top_1_table["diff_from"]<abs_diff_1) & (top_1_table["diff_to"]>abs_diff_1)
    top_1_conf = top_1_table[top_1_mask]["species_match"].iloc[0]
    genus_conf = top_1_table[top_1_mask]["genus_match"].iloc[0]
    family_conf = top_1_table[top_1_mask]["family_match"].iloc[0]
    top_3_conf = top_3_table[(top_3_table["diff_from"]<abs_diff_3) & (top_3_table["diff_to"]>abs_diff_3)]["species_match_top-3"].iloc[0]
    return top_1_conf, top_3_conf, genus_conf, family_conf


if __name__ == "__main__":
    image_path = "../data/plantnet_300K/images_train/1355868/0a342112ddd74ee3ea7918c445e2133fb5b9454d.jpg"
    image = Image.open(image_path)
    response = predict_species(image)
    print(response)