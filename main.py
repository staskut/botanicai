import os
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fasthtml.common import *
import dotenv

dotenv.load_dotenv(".env.local")

app, rt = fast_app()

API_KEY = os.environ.get("PLANTNET_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
API_URL = "https://my-api.plantnet.org/v2/identify/all"

# Initialize LangChain model
gemini_model = None
if GEMINI_API_KEY:
    gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=GEMINI_API_KEY)

@rt('/')
def get():
    return Titled("Plant Species Identifier",
        Container(
            P("Upload a photo of a plant to find out what it is!"),
            Form(
                Label("Select Image:", Input(type="file", name="image", accept="image/*", required=True)),
                Button("Identify Plant", type="submit"),
                action="/identify", method="post", enctype="multipart/form-data"
            )
        )
    )

@rt('/identify')
async def post(image: UploadFile):
    if not API_KEY:
        return Titled("Error", P("PLANTNET_API_KEY environment variable is not set."))

    content = await image.read()
    if not content:
        return Titled("Error", P("Uploaded file is empty."))
    
    # helper to pass multiple files/organs if needed, but here simple 1-1
    files = [
        ('images', (image.filename, content, image.content_type))
    ]
    data = {
        'organs': 'auto'
    }
    params = {
        'api-key': API_KEY,
        'include-related-images': 'false',
        'lang': 'en'
    }

    try:
        response = requests.post(API_URL, files=files, data=data, params=params)
        response.raise_for_status()
        results = response.json()
    except requests.exceptions.RequestException as e:
        error_msg = f"API Request failed: {str(e)}"
        if response is not None:
             try:
                 error_msg += f"\nDetails: {response.text}"
             except:
                 pass
        return Titled("Error", Pre(error_msg))

    # Parse results
    predictions = results.get('results', [])
    if not predictions:
        return Titled("No Results", P("No matching species found."))

    # Create list of cards for predictions
    cards = []
    best_match = None
    
    for i, pred in enumerate(predictions[:3]): # Top 3
        species = pred.get('species', {})
        score = pred.get('score', 0)
        common_names = species.get('commonNames', [])
        scientific_name = species.get('scientificNameWithoutAuthor', 'Unknown')
        
        if i == 0:
            best_match = {
                "scientific_name": scientific_name,
                "common_name": common_names[0] if common_names else scientific_name
            }

        card = Card(
            H3(f"{scientific_name} ({score:.1%})"),
            P(f"Common Names: {', '.join(common_names) if common_names else 'N/A'}")
        )
        cards.append(card)

    # Gemini Description (via LangChain)
    description_section = []
    if gemini_model and best_match:
        try:
            prompt = ChatPromptTemplate.from_template(
                "Identify this plant: {scientific_name} ({common_name}). Provide a fun, short description and care tips. Keep it under 200 words."
            )
            chain = prompt | gemini_model | StrOutputParser()
            description_text = chain.invoke(best_match)
            
            description_section = [
                H2("AI Description & Care Tips"),
                Div(description_text, cls="description-box")
            ]
        except Exception as e:
            description_section = [P(f"Could not generate description: {str(e)}")]

    return Titled("Identification Results",
        H2(f"Results for {image.filename}"),
        *description_section,
        H2("Top Matches"),
        *cards,
        A("Try Another", href="/")
    )

serve()
