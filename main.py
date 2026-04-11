import os
import json
import pandas as pd
import gradio as gr
from image_to_vector.inference_bioclip import predict_species
from text.qwen import ask_botanist

# --- GLOBALS & DATA LOADING ---
# Load taxa lookup mapping globally for O(1) performance
base_dir = os.path.dirname(os.path.abspath(__file__))
taxa_path = os.path.join(base_dir, 'data/taxa.parquet')

VERIFICATION_WHITELIST = """a photo of a plant
a photo of a plant organ
a photo of a fruit
a photo of a seed
a photo of a bud
a photo of a flower
a photo of an inflorescence
a photo of a leaf
# a photo of bark
# a photo of a dried plant
a photo of a plant part""".split("\n")

try:
    taxa_df = pd.read_parquet(taxa_path)
    # Determine the column to use as the unique label identifier
    label_col = 'species'
    
    # Create lookup dict: map label to 'genus' and 'family'
    TAXA_MAP = taxa_df.drop_duplicates(subset=[label_col]).set_index(label_col)[['genus', 'family']].to_dict('index')
    print(f"Successfully loaded taxa lookup with {len(TAXA_MAP)} entries.")

except Exception as e:
    print(f"Warning: Could not load data/taxa.parquet. Lookups will default to 'Unknown'.\n{e}")
    TAXA_MAP = {}


def process_plant(image):
    """
    Gradio interface core handler that processes plant images.
    """
    if image is None:
        return "Please upload an image.", None

    try:
        # Run inference pipeline
        response = predict_species(image)
        
        # response contains: 'top-3', 'top-1_confidence', 'top-3_confidence', 'genus_confidence', 'family_confidence'
        top_species = response.get("top-3", ["Unknown"])[0]
        
        # Look up taxonomic hierarchy based on the highest-confidence predicted species
        ranks = TAXA_MAP.get(top_species, {})
        genus = ranks.get('genus', 'Unknown')
        family = ranks.get('family', 'Unknown')
        
        # Format markdown text readout
        prediction_text = ""
        verif_label = response.get("verification_label")
        print(f"Verification label: {verif_label}")
        if verif_label not in VERIFICATION_WHITELIST:
            prediction_text += f"> ⚠️ **WARNING: Bad Photo Detected.** The botanical results below are likely incorrect.\n\n"

        prediction_text += (
             f"**Identified Species:** {top_species}\n\n"
             f"**Genus:** {genus}\n\n"
             f"**Family:** {family}\n"
        )
        
        # Generate facts using Qwen LLM
        try:
            botanist_response = ask_botanist(top_species, genus, family)
            
            # Try to safely parse the JSON array if the LLM outputted it perfectly
            try:
                import ast
                # Clean up potential markdown wrapper from LLM
                cleaned = botanist_response.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:].rstrip("`").strip()
                elif cleaned.startswith("```"):
                    cleaned = cleaned[3:].rstrip("`").strip()

                try:
                    facts_list = json.loads(cleaned)
                except json.JSONDecodeError:
                    facts_list = ast.literal_eval(cleaned)

                if isinstance(facts_list, list) and len(facts_list) > 0 and isinstance(facts_list[0], dict):
                    blocks = []
                    for i, item in enumerate(facts_list):
                        quote = item.get("quote", "").strip()
                        fact = item.get("fact", "").strip()
                        
                        block = f"#### 🌿 Fact {i+1}\n"
                        if quote:
                            block += f"> *\"{quote}\"*\n>\n"
                        block += f"> {fact}\n"
                        blocks.append(block)
                        
                    facts_text = "\n<br>\n\n".join(blocks)
                elif isinstance(facts_list, list):
                    facts_text = "\n".join([f"- {fact}" for fact in facts_list])
                else:
                    facts_text = botanist_response
            except Exception:
                # If it's not proper JSON, just output the raw text the LLM returned
                facts_text = botanist_response
                
        except Exception as e:
            facts_text = f"*Botanist is currently unavailable: {str(e)}*"
            
        final_output = f"{prediction_text}\n### Botanist's Facts:\n{facts_text}"
        
        # Create dictionary for gr.Label (Gradio normalizes floats to 0%-100% bars if they are 0.0 - 1.0)
        confidences = {
            f"Species: {top_species}": float(response.get('top-1_confidence', 0.0)),
            f"One of: {response.get('top-3')}": float(response.get('top-3_confidence', 0.0)),
            f"Genus: {genus}": float(response.get('genus_confidence', 0.0)),
            f"Family: {family}": float(response.get('family_confidence', 0.0)),
        }
        
        return final_output, confidences

    except Exception as e:
        return f"Error during identification: {str(e)}", None


# --- GR.BLOCKS UI LAYOUT ---
with gr.Blocks(title="BotanicAI - Plant Identifier") as demo:
    gr.Markdown("# 🌿 BotanicAI - Plant Identifier")
    gr.Markdown(
        "Upload a photo of a plant to identify its species, view confidence metrics, and learn amazing facts about it."
    )

    with gr.Row():
        # Input column
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Plant Image")
            submit_btn = gr.Button("Identify Plant", variant="primary")
            
        # Output column
        with gr.Column(scale=2):
            text_output = gr.Markdown(label="Identification & Facts")
            confidences_output = gr.Label(label="Confidences", num_top_classes=4)

    submit_btn.click(
        fn=process_plant,
        inputs=image_input,
        outputs=[text_output, confidences_output]
    )

if __name__ == "__main__":
    demo.launch()