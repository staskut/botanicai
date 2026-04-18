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


def format_facts(botanist_response, source_url):
    """Parses JSON text and generates colorful HTML cards with hidden quote accordions."""
    try:
        import ast
        cleaned = botanist_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:].rstrip("`").strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:].rstrip("`").strip()

        # Attempt graceful rescue for truncated LLM outputs
        # 1. Check for unclosed string literal quotes
        if cleaned.count('"') % 2 != 0:
            cleaned += '"'
            
        # 2. Check for unclosed json dictionary brackets
        open_curly = cleaned.count('{')
        close_curly = cleaned.count('}')
        if open_curly > close_curly:
            cleaned += '}' * (open_curly - close_curly)
            
        # 3. Check for unclosed list brackets
        open_square = cleaned.count('[')
        close_square = cleaned.count(']')
        if open_square > close_square:
            cleaned += ']' * (open_square - close_square)

        try:
            facts_list = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback to ast parsing if json is too strict 
            facts_list = ast.literal_eval(cleaned)

        if isinstance(facts_list, list) and len(facts_list) > 0 and isinstance(facts_list[0], dict):
            blocks = []
            for i, item in enumerate(facts_list):
                quote = item.get("quote", "").strip()
                fact = item.get("fact", "").strip()
                
                card_html = f"""
                <div style="background-color: #f0f7f4; border-left: 5px solid #2e8b57; padding: 15px; margin-bottom: 15px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <div style="font-size: 1.1em; font-weight: bold; color: #1e5c3a; margin-bottom: 8px;">
                        🌿 Fact {i+1}
                    </div>
                    <div style="font-size: 1.05em; margin-bottom: 12px; color: #333;">
                        {fact}
                    </div>
                    <details style="cursor: pointer; background-color: #e6f2eb; padding: 8px; border-radius: 4px;">
                        <summary style="font-weight: 500; color: #2e8b57; user-select: none;">View Source Quote</summary>
                        <div style="margin-top: 8px; font-style: italic; color: #555; padding-left: 10px; border-left: 3px solid #2e8b57;">
                            "{quote}"
                        </div>
                    </details>
                </div>
                """
                blocks.append(card_html)
                
            if source_url:
                source_html = f"""
                <div style="margin-top: 15px; padding: 10px; background-color: #eef2f5; border-radius: 4px; font-size: 0.9em;">
                    🔗 <b>Source:</b> <a href="{source_url}" target="_blank" style="color: #0366d6; text-decoration: none;">{source_url}</a>
                </div>
                """
                blocks.append(source_html)

            return "\n".join(blocks)
        elif isinstance(facts_list, list):
            return "<br>".join([f"<li>{f}</li>" for f in facts_list])
        else:
            return f"<pre style='white-space: pre-wrap; font-family: inherit;'>{botanist_response}</pre>"
    except Exception as e:
         return f"<div style='padding: 10px; background: #fff3f3; border-left: 4px solid #ff4444;'><p><i>Displaying raw output (JSON parse failed):</i></p><pre style='white-space: pre-wrap; font-family: inherit;'>{botanist_response}</pre></div>"


def process_plant(image):
    """
    Gradio interface core handler that processes plant images.
    """
    if image is None:
        yield "Please upload an image.", None, "", ""
        return

    try:
        # Run inference pipeline
        response = predict_species(image)
        
        # response contains: 'top-3', 'top-1_confidence', 'top-3_confidence', 'genus_confidence', 'family_confidence'
        top_species = response.get("top-3", ["Unknown"])[0]
        
        # Look up taxonomic hierarchy based on the highest-confidence predicted species
        ranks = TAXA_MAP.get(top_species, {})
        genus = ranks.get('genus', 'Unknown')
        family = ranks.get('family', 'Unknown')
        
        # Format markdown text readout (only used for warnings now)
        prediction_text = ""
        verif_label = response.get("verification_label")
        print(f"Verification label: {verif_label}")
        if verif_label is not None and verif_label not in VERIFICATION_WHITELIST:
            prediction_text += f"> ⚠️ **WARNING: Bad Photo Detected.** The botanical results below are likely incorrect.\n\n"
        
        # Create dictionary for gr.Label (Gradio normalizes floats to 0%-100% bars if they are 0.0 - 1.0)
        top_1_in_top_60 = response.get("top_1_in_top_60", False)
        
        confidences = {}
        if top_1_in_top_60:
            confidences[f"Species: {top_species}"] = float(response.get('top-1_confidence', 0.0))
        else:
            confidences[f"One of: {response.get('top-3')}"] = float(response.get('top-3_confidence', 0.0))
            
        confidences[f"Genus: {genus}"] = float(response.get('genus_confidence', 0.0))
        confidences[f"Family: {family}"] = float(response.get('family_confidence', 0.0))
        
        # Determine Title State
        if top_1_in_top_60:
            facts_header = f"### 📝 Botanist's Facts (Identified Species: {top_species})"
        else:
            facts_header = f"### 📝 Botanist's Facts (Top-3 Scenario: displaying facts for {top_species})"
        
        # Step 1: Push classification results while loading the first LLM
        yield prediction_text, confidences, facts_header, "<i>Gathering Pupil Facts... (this may take a moment pending your local MLX GPU queue)</i>", "<i>Waiting for Pupil job to finish...</i>"
        
        # Step 2: Generate Pupil Facts 
        try:
            bot_pupil, pupil_source, match_level = ask_botanist(top_species, genus, family, audience="Pupil")
            pupil_html = format_facts(bot_pupil, pupil_source)
            if match_level != "None":
                 facts_header += f"\n<div style='font-size: 0.9em; font-style: italic; color: #555;'>Data retrieved at <b>{match_level}</b> level</div>"
        except Exception as e:
            pupil_html = f"<span style='color:red;'>Botanist (Pupil) unavailable: {str(e)}</span>"
            
        yield prediction_text, confidences, facts_header, pupil_html, "<i>Gathering Student Facts...</i>"
            
        # Step 3: Generate Student Facts
        try:
            bot_student, student_source, _ = ask_botanist(top_species, genus, family, audience="Student")
            student_html = format_facts(bot_student, student_source)
        except Exception as e:
            student_html = f"<span style='color:red;'>Botanist (Student) unavailable: {str(e)}</span>"

        yield prediction_text, confidences, facts_header, pupil_html, student_html

    except Exception as e:
        yield f"Error during identification: {str(e)}", None, "### 📝 Botanist's Facts", "", ""


# --- GR.BLOCKS UI LAYOUT ---
with gr.Blocks(title="BotanicAI - Plant Identifier") as demo:
    gr.Markdown("# 🌿 BotanicAI - Plant Identifier")
    gr.Markdown(
        "Upload a photo of a plant to identify its species, view confidence metrics, and learn amazing facts about it."
    )

    with gr.Row():
        # Output column (Classification Results Top-Left)
        with gr.Column(scale=1):
            text_output = gr.Markdown(label="Identification Results")
            confidences_output = gr.Label(label="Confidences", num_top_classes=4)
            
        # Input column (Image Upload Top-Right)
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Plant Image", height=300)
            submit_btn = gr.Button("Identify Plant", variant="primary")
            
    # Full Width Facts Section at the bottom
    with gr.Row():
        with gr.Column(scale=1):
            facts_title = gr.Markdown("### 📝 Botanist's Facts")
            with gr.Tabs():
                with gr.Tab("Pupil"):
                    pupil_output = gr.HTML(label="Pupil Facts")
                with gr.Tab("Student"):
                    student_output = gr.HTML(label="Student Facts")

    submit_btn.click(
        fn=process_plant,
        inputs=image_input,
        outputs=[text_output, confidences_output, facts_title, pupil_output, student_output]
    )

if __name__ == "__main__":
    demo.launch()