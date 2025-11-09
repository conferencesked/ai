from fastapi import FastAPI, Request
from pydantic import BaseModel
from openai import OpenAI
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Initialize OpenAI client
client = OpenAI()

# Define category list (same as before)
categories = [
    "Engineering", "Energy", "Design", "Data Mining", "Computing", "Computer software and applications",
    "Biotechnology", "Bioinformatics", "Biomedical Engineering", "Artificial Intelligence", "Architecture",
    "Water", "Waste Management", "Soil", "Physics", "Oceanography", "Meteorology", "Genetics", "GIS",
    "Environment", "Ecology", "Earth Sciences", "Chemistry", "Biology", "Biodiversity", "Astronomy",
    "Archaeology", "Aquaculture", "Agriculture", "Marketing", "Management", "Human Resources", "Economics",
    "E-commerce", "Business Ethics", "Business", "Banking and finance", "Statistics", "Mathematics",
    "European Studies", "Asian Studies", "American Studies", "African Studies", "Women's studies", "Violence",
    "Urban studies", "Tourism", "Sport science", "Sustainable development", "Spirituality",
    "Sexuality and eroticism", "Public Policy", "Poverty", "Memory", "Leadership", "Identity", "Human Rights",
    "HIV/AIDS", "Globalization", "Gender studies", "GLBT Studies", "Film studies", "Discourse",
    "Disaster Management", "Culture", "Creativity", "Conflict resolution", "Communications and Media",
    "Children and Youth", "Women's history", "Sociology", "Social Sciences", "Religious studies", "Psychology",
    "Politics", "Poetry", "Philosophy", "Occupational Science", "Music", "Museums and heritage",
    "Multidisciplinary Studies", "Local Government", "Literature", "Linguistics", "Language", "Islamic Studies",
    "Interdisciplinary studies", "Information science", "History", "English", "Arts", "Art History",
    "Anthropology", "Animal Sciences", "Health and Medicine", "Law", "Education", "Engineering and Technology",
    "Physical and life sciences", "Business and Economics", "Mathematics and statistics", "Regional Studies",
    "Interdisciplinary", "Social Sciences", "Forestry", "Information Technology", "Internet and World Wide Web",
    "Manufacturing", "Military", "Mining", "Nanotechnology and Smart Materials", "Polymers and Plastics",
    "Renewable Energy", "Robotics", "Space Environment and Aviation Technology", "Systems Engineering",
    "Transport", "E-learning", "Higher Education", "Lifelong Learning", "Teaching and Learning",
    "Justice and legal studies", "Alternative Health", "Cardiology", "Dentistry", "Dermatology",
    "Disability and Rehabilitation", "Family Medicine", "Food Safety", "Gastroenterology", "Gerontology", "Health",
    "Infectious diseases", "Medical ethics", "Medicine and Medical Science", "Neurology", "Nursing",
    "Nutrition and Dietetics", "Oncology", "Palliative Care", "Psychiatry", "Public Health", "Radiology",
    "Reproductive Medicine and Women's Health", "Social Work", "Surgery", "Veterinary Science",
    "Image Processing", "Virtual Reality", "Other", "Mechanical and Aerospace Engineering",
    "Electronics and Electric Engineering"
]

# ⚙️ Precompute category embeddings once
print("🔄 Generating category embeddings (first-time only)...")
category_embeddings = []
for cat in categories:
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=cat
    ).data[0].embedding
    category_embeddings.append(emb)
category_embeddings = np.array(category_embeddings)
print("✅ Category embeddings loaded successfully.")


# 🧾 Request Model
class EventData(BaseModel):
    event_name: str = ""
    content: str = ""
    keywords: str = ""


# 🧠 Helper Function
def get_top_categories(event_text: str):
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=event_text
    ).data[0].embedding
    emb = np.array(emb)
    similarities = np.dot(category_embeddings, emb) / (
        np.linalg.norm(category_embeddings, axis=1) * np.linalg.norm(emb)
    )
    top_indices = similarities.argsort()[-3:][::-1]
    top_results = [{"category": categories[i], "score": float(similarities[i])} for i in top_indices]
    return top_results


# 🧩 Endpoint
@app.post("/classify")
def classify(event: EventData):
    text = f"{event.event_name} {event.content} {event.keywords}".strip()
    if not text:
        return {"error": "Please provide event_name, content, or keywords."}
    results = get_top_categories(text)
    return {"top_categories": results}


# 🧩 Health check endpoint
@app.get("/")
def health():
    return {"status": "OK", "message": "Conference Category Classifier is running."}
