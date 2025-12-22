import torch
import pandas as pd
import json
import faiss
import numpy as np
import open_clip
from tqdm import tqdm

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_ID = 'hf-hub:imageomics/bioclip-2'

model, _, _ = open_clip.create_model_and_transforms(MODEL_ID)
tokenizer = open_clip.get_tokenizer(MODEL_ID)
model.to(DEVICE)
model.eval()


def run_bioclip2_hierarchical_test():
    hierarchy_df = pd.read_csv('species_hierarchy.csv')

    # Завантаження нових метаданих та індексу
    with open('bioclip2_img_metadata.json', 'r') as f:
        img_metadata = json.load(f)

    index = faiss.read_index("bioclip2_plantnet_test.index")
    img_vectors = index.reconstruct_n(0, index.ntotal).astype('float32')

    levels = ['CLASS', 'ORDER', 'FAMILY', 'GENUS', 'species']
    all_results = []
    per_class_metrics = []

    for level in levels:
        print(f"\n### BioCLIP-2 Testing: {level} ###")
        unique_labels = hierarchy_df[level].unique().tolist()

        print(f"Vectorizing {len(unique_labels)} labels...")
        with torch.no_grad():
            if True:
                input_str_template = "a photo of"
            else:
                input_str_template = f"a photo of a plant of {level.lower()}"

            text_inputs = tokenizer([f"{input_str_template} {l}" for l in unique_labels]).to(DEVICE)
            text_features = model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features_np = text_features.cpu().numpy().astype('float32')

        sim_matrix = img_vectors @ text_features_np.T

        correct_top1 = 0
        level_similarities = []
        class_stats = {label: {'hits': 0, 'total': 0, 'sum_sim': 0.0} for label in unique_labels}

        for i, meta in enumerate(img_metadata):
            # Перетворюємо латинську назву для пошуку в CSV (як у вашому фіксі)
            true_species = meta['latin_name'].replace("_", " ")

            row = hierarchy_df[hierarchy_df['species'] == true_species]
            if row.empty: continue

            true_label = row.iloc[0][level]
            pred_idx = np.argmax(sim_matrix[i])
            pred_label = unique_labels[pred_idx]
            max_sim = sim_matrix[i][pred_idx]

            class_stats[true_label]['total'] += 1
            class_stats[true_label]['sum_sim'] += max_sim

            if pred_label == true_label:
                correct_top1 += 1
                class_stats[true_label]['hits'] += 1

            level_similarities.append(max_sim)

        acc = correct_top1 / len(level_similarities)
        avg_sim = np.mean(level_similarities)
        print(f"BioCLIP-2 {level} Accuracy: {acc:.2%}")

        all_results.append({'level': level, 'accuracy': acc, 'avg_similarity': avg_sim})

        for label, stats in class_stats.items():
            if stats['total'] > 0:
                per_class_metrics.append({
                    'taxonomic_level': level,
                    'label_name': label,
                    'accuracy': stats['hits'] / stats['total'],
                    'avg_similarity': stats['sum_sim'] / stats['total'],
                    'sample_count': stats['total']
                })

    # Збереження результатів спеціально для BioCLIP-2
    pd.DataFrame(all_results).to_csv('bioclip2_summary.csv', index=False)
    pd.DataFrame(per_class_metrics).to_csv('bioclip2_per_class.csv', index=False)
    print("\n[+] BioCLIP-2 Benchmark complete.")


if __name__ == '__main__':
    run_bioclip2_hierarchical_test()