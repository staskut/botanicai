import json
import os
import faiss
import numpy as np
import pandas as pd
from scipy.special import softmax

def load_data(img_index_path, img_meta_path, text_index_path, text_labels_path, taxa_parquet_path, split_filter=None,
              filter_by_classes=None, num_random_classes=0, random_seed=42):
    """
    Loads pre-computed FAISS indices (images, text embeddings), metadata, and taxonomic hierarchy.
    
    Args:
        filter_by_classes (list): If True, filters the loaded text_index to include only the species
                                      found within the loaded img_meta.
        num_random_classes (int): Number of additional random classes to include in the text_index.
        random_seed (int): Random seed for reproducibility when sampling random classes.

    Returns:
        img_vectors (np.ndarray): Image vectors extracted from the FAISS index.
        img_meta (list): List of dictionaries containing image metadata (e.g. split, latin_name).
        text_index (faiss.Index): FAISS index containing the text embeddings for species.
        text_labels (list): Ordered list of species names corresponding to text_index.
        taxa_df (pd.DataFrame): DataFrame containing the taxonomic hierarchy.
    """
    print(f"Loading Image Index: {img_index_path}")
    img_index = faiss.read_index(img_index_path)
    img_vectors = img_index.reconstruct_n(0, img_index.ntotal)
    
    print(f"Loading Image Metadata: {img_meta_path}")
    with open(img_meta_path, 'r') as f:
        img_meta = json.load(f)

    if split_filter:
        split_mask = np.array([el["split"] in split_filter for el in img_meta])
        img_vectors = img_vectors[split_mask]
        img_meta = [el for el in img_meta if el["split"] in split_filter]

    print(f"Loading Text Index: {text_index_path}")
    text_index = faiss.read_index(text_index_path)

    print(f"Loading Text Labels: {text_labels_path}")
    with open(text_labels_path, 'r') as f:
        text_labels = json.load(f)

    if filter_by_classes or num_random_classes > 0:
        print("Filtering text embeddings...")
        selected_classes = set()
        
        if filter_by_classes:
            selected_classes = filter_by_classes
                    
        if num_random_classes > 0:
            rng = np.random.default_rng(random_seed)
            available_classes = [lbl for lbl in text_labels if lbl not in selected_classes]
            num_to_sample = min(num_random_classes, len(available_classes))
            if num_to_sample > 0:
                sampled = rng.choice(available_classes, size=num_to_sample, replace=False)
                selected_classes.update(sampled)
                
        # Reconstruct filtered index
        indices_to_keep = [i for i, lbl in enumerate(text_labels) if lbl in selected_classes]
        
        if len(indices_to_keep) < len(text_labels):
            all_text_vectors = text_index.reconstruct_n(0, text_index.ntotal)
            filtered_vectors = all_text_vectors[indices_to_keep]
            
            dim = filtered_vectors.shape[1]
            new_text_index = faiss.IndexFlatIP(dim)
            new_text_index.add(filtered_vectors)
            
            text_index = new_text_index
            text_labels = [text_labels[i] for i in indices_to_keep]
            print(f"Filtered text index down to {text_index.ntotal} vectors.")

    print(f"Loading Taxonomic Data: {taxa_parquet_path}")
    taxa_df = pd.read_parquet(taxa_parquet_path)
    
    return img_vectors, img_meta, text_index, text_labels, taxa_df


def run_similarity_search(img_vectors, text_index, k=100):
    """
    Finds the top-K matching text embeddings for each image vector.
    
    Returns:
        scores (np.ndarray): Shape (N, k). Inner product similarity scores.
        indices (np.ndarray): Shape (N, k). FAISS indices for the text embeddings.
    """
    print(f"Running similarity search for top-{k} matches...")
    scores, indices = text_index.search(img_vectors, k)
    return scores, indices


def _get_taxa_dict(taxa_df):
    """Helper function to create O(1) lookup dictionary for taxonomic ranks."""
    # Ensure distinct names, use first entry if duplications exist
    df_unique = taxa_df.drop_duplicates(subset=['species'])
    
    # We map 'species' to its corresponding upper ranks
    return df_unique.set_index('species')[['genus', 'family', 'order', 'class']].to_dict('index')


def inference_vanilla(scores, indices, text_labels, taxa_df):
    """
    Strategy 1: Always predict the species with the highest similarity.
    Outputs the top-1 species and perfectly propagates its parent taxonomy.
    """
    taxa_map = _get_taxa_dict(taxa_df)
    predictions = []
    
    for i in range(len(indices)):
        top_idx = indices[i][0]
        pred_species = text_labels[top_idx]
        ranks = taxa_map.get(pred_species, {})
        
        predictions.append({
            'species': pred_species,
            'genus': ranks.get('genus'),
            'family': ranks.get('family'),
            'order': ranks.get('order'),
            'class': ranks.get('class')
        })
        
    return predictions


def inference_threshold(scores, indices, text_labels, taxa_df, threshold=0.25):
    """
    Strategy 2: Predict top-1 species if its score >= threshold.
    Otherwise, fall back to predicting only the Genus (and its ancestors) of the top species.
    """
    taxa_map = _get_taxa_dict(taxa_df)
    predictions = []
    
    for i in range(len(indices)):
        top_idx = indices[i][0]
        top_score = scores[i][0]
        top_species = text_labels[top_idx]
        
        ranks = taxa_map.get(top_species, {})
        
        # If score is below threshold, species prediction is None (fallback)
        pred_species = top_species if top_score >= threshold else None
        
        predictions.append({
            'species': pred_species,
            'genus': ranks.get('genus'),
            'family': ranks.get('family'),
            'order': ranks.get('order'),
            'class': ranks.get('class')
        })
        
    return predictions


def inference_top_n_softmax(scores, indices, text_labels, taxa_df, n=5, species_threshold=0.5):
    """
    Strategy 3: Apply Softmax weighting to the top-N species similarity scores.
    Aggregates probability by common Genus. 
    Predicts the best species if max_species_prob >= species_threshold, 
    otherwise falls back to the Genus with the highest aggregated probability.
    """
    taxa_map = _get_taxa_dict(taxa_df)
    
    # Pre-build reverse lookup to find higher ranks from genus (pick the first available species for that genus)
    genus_to_ranks = {}
    for sp, ranks in taxa_map.items():
        g = ranks.get('genus')
        if pd.notna(g) and g not in genus_to_ranks:
            genus_to_ranks[g] = {'family': ranks.get('family'), 'order': ranks.get('order'), 'class': ranks.get('class')}

    predictions = []
    for i in range(len(indices)):
        top_n_scores = scores[i][:n]
        top_n_indices = indices[i][:n]
        
        # Apply softmax across top N
        probs = softmax(top_n_scores)
        
        genus_probs = {}
        best_species = None
        best_species_prob = -1
        
        # Track best species and aggregate genus probabilities
        for p, idx in zip(probs, top_n_indices):
            sp = text_labels[idx]
            g = taxa_map.get(sp, {}).get('genus')
            
            if pd.notna(g):
                genus_probs[g] = genus_probs.get(g, 0.0) + p
                
            if p > best_species_prob:
                best_species_prob = p
                best_species = sp
                
        # Find best genus by sum of probabilities
        best_genus = max(genus_probs.items(), key=lambda x: x[1])[0] if genus_probs else None
        
        pred = {}
        if best_species_prob >= species_threshold:
            # High confidence in species, propagate from species
            pred['species'] = best_species
            ranks = taxa_map.get(best_species, {})
            pred['genus'] = ranks.get('genus')
            pred['family'] = ranks.get('family')
            pred['order'] = ranks.get('order')
            pred['class'] = ranks.get('class')
        else:
            # Low confidence in species, fallback to best aggregated Genus
            pred['species'] = None
            pred['genus'] = best_genus
            ranks = genus_to_ranks.get(best_genus, {})
            pred['family'] = ranks.get('family')
            pred['order'] = ranks.get('order')
            pred['class'] = ranks.get('class')
            
        predictions.append(pred)
        
    return predictions


def inference_margin_threshold(scores, indices, text_labels, taxa_df, margin_threshold=0.05):
    """
    Strategy 4: Predict species if the difference in similarity between the 1st and 2nd top prediction
    is >= margin_threshold. Otherwise, fall back to predicting only the Genus of the top prediction.
    """
    taxa_map = _get_taxa_dict(taxa_df)
    predictions = []
    
    for i in range(len(indices)):
        top_idx = indices[i][0]
        top_score = scores[i][0]
        second_score = scores[i][1] if len(scores[i]) > 1 else 0.0
        
        top_species = text_labels[top_idx]
        ranks = taxa_map.get(top_species, {})
        
        # If score diff is below threshold, species prediction is None (fallback)
        pred_species = top_species if (top_score - second_score) >= margin_threshold else None
        
        predictions.append({
            'species': pred_species,
            'genus': ranks.get('genus'),
            'family': ranks.get('family'),
            'order': ranks.get('order'),
            'class': ranks.get('class')
        })
        
    return predictions


def evaluate(predictions, img_meta, taxa_df, split_filter=None):
    """
    Evaluates predictions against ground truth labels inside img_meta.
    Tracks metric accuracy per taxonomic rank separately for each dataset split (train/val/test).
    Calculates coverage (how often species is successfully predicted vs falling back).
    
    Args:
        predictions (list): List of metric dictionaries output by inference_* functions.
        img_meta (list): Ground truth metadata corresponding to predictions.
        taxa_df (pd.DataFrame): Taxonomic dataframe.
        split_filter (list): Optional list of splits to evaluate (e.g. ['train', 'val', 'test']).
        
    Returns:
        tuple: (summary_df, detailed_df)
            - summary_df (pd.DataFrame): Aggregated metrics grouped by dataset split.
            - detailed_df (pd.DataFrame): Row-by-row matching results including ground truth
              taxonomic ranks to allow custom aggregations (e.g., `detailed_df.groupby('gt_family').mean()`).
    """
    taxa_map = _get_taxa_dict(taxa_df)
    
    results = {
        'split': [],
        'gt_species': [],
        'gt_genus': [],
        'gt_family': [],
        'gt_order': [],
        'gt_class': [],
        'species_match': [],
        'species_predicted': [],
        'genus_match': [],
        'family_match': [],
        'order_match': [],
        'class_match': []
    }
    
    for i, meta in enumerate(img_meta):
        split = meta.get('split', 'train')  # default to train if missing
        
        if split_filter and split not in split_filter:
            continue
            
        gt_species = meta.get('latin_name').replace('_', ' ')
        gt_ranks = taxa_map.get(gt_species, {})
        
        gt_genus = gt_ranks.get('genus')
        gt_family = gt_ranks.get('family')
        gt_order = gt_ranks.get('order')
        gt_class = gt_ranks.get('class')
        
        pred = predictions[i]
        
        # Determine exact matches
        def is_match(pred_val, gt_val):
            if pd.isna(pred_val) or pd.isna(gt_val) or pred_val is None or gt_val is None:
                return None
            return pred_val == gt_val

        results['split'].append(split)
        results['gt_species'].append(gt_species)
        results['gt_genus'].append(gt_genus)
        results['gt_family'].append(gt_family)
        results['gt_order'].append(gt_order)
        results['gt_class'].append(gt_class)
        
        results['species_match'].append(is_match(pred.get('species'), gt_species))
        results['species_predicted'].append(pred.get('species') is not None)
        results['genus_match'].append(is_match(pred.get('genus'), gt_genus))
        results['family_match'].append(is_match(pred.get('family'), gt_family))
        results['order_match'].append(is_match(pred.get('order'), gt_order))
        results['class_match'].append(is_match(pred.get('class'), gt_class))
        
    detailed_df = pd.DataFrame(results)
    
    if len(detailed_df) == 0:
        print("Warning: No matching splits found for evaluation.")
        return pd.DataFrame(), pd.DataFrame()

    # Aggregate by split
    metrics = []
    for split, group in detailed_df.groupby('split'):
        count = len(group)
        metrics.append({
            'Split': split,
            'Count': count,
            'Species Accuracy': group['species_match'].mean(),
            'Species Coverage': group['species_predicted'].mean(),
            'Genus Accuracy': group['genus_match'].mean(),
            'Family Accuracy': group['family_match'].mean(),
            'Order Accuracy': group['order_match'].mean(),
            'Class Accuracy': group['class_match'].mean()
        })
        
    summary_df = pd.DataFrame(metrics).set_index('Split')
    return summary_df, detailed_df


if __name__ == '__main__':
    import pandas as pd

    # 1. Load Everything
    img_index = "../data/bioclip2_plantnet300k.index"
    img_meta = "../data/bioclip2_plantnet300k_metadata.json"
    text_index = "../data/bioclip2_text_species.index"
    text_labels = "../data/bioclip2_text_species_labels.json"
    taxa = "../data/taxa.parquet"
    # 1. Load Everything
    # Example to filter text embeddings to only those categories present in the images + N random ones:
    # img_vecs, metas, text_idx, labels, taxa_df = load_data(
    #     img_index, img_meta, text_index, text_labels, taxa,
    #     filter_by_img_classes=True, num_random_classes=1000
    # )
    img_vecs, metas, text_idx, labels, taxa_df = load_data(img_index, img_meta, text_index, text_labels, taxa)

    # 2. Get Top-100 Matches
    scores, indices = run_similarity_search(img_vecs, text_idx, k=100)

    # 3. Try a strategy (e.g. threshold strategy)
    preds = inference_threshold(scores, indices, labels, taxa_df, threshold=0.30)

    # 4. Evaluate to retrieve summary metrics and the detailed DataFrame
    summary, details = evaluate(preds, metas, taxa_df, split_filter=['train', 'val', 'test'])

    display(summary)

    # Example: Check which Families performed the worst
    family_metrics = details.groupby('gt_family')[['species_match', 'genus_match']].mean()


if __name__=="main()":
    img_index = "../data/bioclip2_plantnet300k.index"
    img_meta = "../data/bioclip2_plantnet300k_metadata.json"
    text_index = "../data/bioclip2_text_v2.index"
    text_labels = "../data/bioclip2_text_v2_labels.json"
    taxa = "../data/taxa.parquet"
    load_data(img_index, img_meta, text_index, text_labels, taxa, filter_by_classes=None, )