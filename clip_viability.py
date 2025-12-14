import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from io import BytesIO
import numpy as np

# --- 1. CONFIGURATION & PREPARATION ---
MODEL_ID = "openai/clip-vit-base-patch32"
print(f"Loading model: {MODEL_ID}...")

try:
    model = CLIPModel.from_pretrained(MODEL_ID)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Device configuration (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# --- 2. TEST DATA SETUP ---
# 10 Plant Species with 3 label types each: Name, Description, Fact
# Image URLs are selected from public sources (Wikimedia/Unsplash) to represent the species.

plants_data = [
    {
        "id": "rosa_canina",
        "labels": {
            "name": "Rosa canina",
            "desc": "Багаторічний чагарник з рожевими квітами та шипами (Dog Rose).",
            "fact": "Плоди шипшини дуже багаті на вітамін С, використовуються для чаю."
        },
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/Rosa_canina_blatt_2005.05.26_11.50.13.jpg/640px-Rosa_canina_blatt_2005.05.26_11.50.13.jpg"
    },
    {
        "id": "helianthus_annuus",
        "labels": {
            "name": "Helianthus annuus",
            "desc": "Висока однорічна рослина з великими жовтими суцвіттями (Sunflower).",
            "fact": "Ця рослина проявляє геліотропізм — повороти слідом за сонцем."
        },
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/A_sunflower.jpg/640px-A_sunflower.jpg"
    },
    {
        "id": "monstera_deliciosa",
        "labels": {
            "name": "Monstera deliciosa",
            "desc": "Тропічна ліана з великим розсіченим зеленим листям.",
            "fact": "У природі монстера дає їстівні плоди, схожі на качан кукурудзи зі смаком ананаса."
        },
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/Monstera_deliciosa_DSCN4590.jpg/640px-Monstera_deliciosa_DSCN4590.jpg"
    },
    {
        "id": "lavandula_angustifolia",
        "labels": {
            "name": "Lavandula angustifolia",
            "desc": "Кущик з вузьким листям і ароматними фіолетовими квітами.",
            "fact": "Лаванда широко використовується в парфумерії та для заспокоєння нервової системи."
        },
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/Lamiales_-_Lavandula_angustifolia_-_5.jpg/640px-Lamiales_-_Lavandula_angustifolia_-_5.jpg"
    },
    {
        "id": "quercus_robur",
        "labels": {
            "name": "Quercus robur",
            "desc": "Могутнє листяне дерево з лопатевим листям і жолудями (English Oak).",
            "fact": "Дуб може жити понад 1000 років і вважається символом сили."
        },
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/New_oak_leaves_with_female_flowers.jpg/640px-New_oak_leaves_with_female_flowers.jpg"
    },
    {
        "id": "dionaea_muscipula",
        "labels": {
            "name": "Dionaea muscipula",
            "desc": "Хижа рослина з пастками, що закриваються при дотику (Venus Flytrap).",
            "fact": "Венерина мухоловка ловить комах, щоб компенсувати нестачу азоту в ґрунті."
        },
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/Venus_flytrap_%28Dionaea_muscipula%29_Zagreb_Botanical_Garden_1060.jpg/640px-Venus_flytrap_%28Dionaea_muscipula%29_Zagreb_Botanical_Garden_1060.jpg"
    },
    {
        "id": "ficus_elastica",
        "labels": {
            "name": "Ficus elastica",
            "desc": "Дерево з великим, шкірястим, блискучим овальним листям (Rubber Fig).",
            "fact": "Сік цієї рослини містить латекс, який раніше використовували для виробництва гуми."
        },
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Ficus_elastica_2.jpg/640px-Ficus_elastica_2.jpg"
    },
    {
        "id": "mentha_piperita",
        "labels": {
            "name": "Mentha piperita",
            "desc": "Трав'яниста рослина з сильним свіжим ароматом і зубчастим листям (Peppermint).",
            "fact": "Ментол, що міститься в м'яті, активує рецептори, що відчувають холод."
        },
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Mentha_piperita_0zz.jpg/640px-Mentha_piperita_0zz.jpg"
    },
    {
        "id": "acer_saccharum",
        "labels": {
            "name": "Acer saccharum",
            "desc": "Дерево з п'ятилопатевим листям, що стає яскраво-червоним восени (Sugar Maple).",
            "fact": "З соку цього клена виготовляють знаменитий кленовий сироп."
        },
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/Acer_saccharum_1-jgreenlee_%285098070608%29.jpg/640px-Acer_saccharum_1-jgreenlee_%285098070608%29.jpg"
    },
    {
        "id": "solanum_lycopersicum",
        "labels": {
            "name": "Solanum lycopersicum",
            "desc": "Рослина з червоними соковитими їстівними плодами (Tomato).",
            "fact": "Ботанічно помідор є ягодою, хоча в кулінарії його розглядають як овоч."
        }
        ,"image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/Tomato_je.jpg/640px-Tomato_je.jpg"
    }
]

# Flatten all labels for similarity search pool (Total 30 labels)
all_labels = []
label_metadata = [] # To keep track which plant/type matches the label

for plant in plants_data:
    for lbl_type, text in plant["labels"].items():
        all_labels.append(text)
        label_metadata.append({
            "plant_id": plant["id"],
            "type": lbl_type,
            "text": text
        })

print(f"Prepared {len(all_labels)} text labels and {len(plants_data)} test images.")

# --- 3. HELPER FUNCTIONS ---

def load_image(url):
    """Downloads image from URL and converts to PIL Image."""
    try:
        header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(url, headers=header, stream=True, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except Exception as e:
        print(f"Failed to load image {url}: {e}")
        # Return a dummy black image in case of failure to keep script running
        return Image.new("RGB", (224, 224), color="black")

# --- 4. ENCODING & INFERENCE ---

# 4.1 Encode Texts (Zero-Shot Prototypes)
print("Encoding text labels...")
# Processing text: truncation=True, padding=True
text_inputs = processor(text=all_labels, return_tensors="pt", padding=True, truncation=True).to(device)

with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)

# Normalize text features for cosine similarity
text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# 4.2 Process Images and Compare
results = []

print("\nStarting evaluation loop...")
print("-" * 60)

for plant in plants_data:
    plant_id = plant["id"]
    print(f"Processing image for: {plant['labels']['name']}...")
    
    # Load and preprocess image
    image = load_image(plant["image_url"])
    image_inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
    
    # Normalize image features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # 4.3 Calculator Similarity (Cosine Similarity)
    # image_features: [1, 512], text_features: [30, 512] -> [1, 30]
    # We will use raw cosine similarity for ranking 'matches'.
    cosine_sim = (image_features @ text_features.T).squeeze(0).cpu().numpy()

    # Get Top-5 indices
    # argsort returns lowest to highest, so we take last 5 and reverse
    top_5_indices = cosine_sim.argsort()[-5:][::-1]

    # --- 5. EVALUATION ---
    print(f"  > Top-5 Predictions for {plant_id}:")
    
    is_correct_top1 = False
    is_correct_top5 = False

    for rank, idx in enumerate(top_5_indices):
        score = cosine_sim[idx]
        matched_meta = label_metadata[idx]
        is_match = (matched_meta["plant_id"] == plant_id)
        
        marker = "[CORRECT]" if is_match else ""
        print(f"    {rank+1}. {score:.4f} | {matched_meta['type'].upper()}: {matched_meta['text'][:60]}... {marker}")

        if is_match:
            is_correct_top5 = True
            if rank == 0:
                is_correct_top1 = True

    results.append({
        "plant": plant_id,
        "top1": is_correct_top1,
        "top5": is_correct_top5
    })
    print("-" * 60)

# --- 6. SUMMARY ---
print("\n=== SUMMARY RESULTS ===")
total = len(results)
top1_acc = sum(1 for r in results if r["top1"]) / total * 100
top5_acc = sum(1 for r in results if r["top5"]) / total * 100

print(f"Total Species Tested: {total}")
print(f"Top-1 Accuracy (Exact Match): {top1_acc:.1f}%")
print(f"Top-5 Accuracy (Retrieval):   {top5_acc:.1f}%")

if top5_acc > 80:
    print("\nCONCLUSION: CLIP-Enhanced Retrieval looks VIABLE for this task.")
else:
    print("\nCONCLUSION: CLIP performance is mixed. Consider fine-tuning or better prompting.")
