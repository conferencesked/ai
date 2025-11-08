from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Load a pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and efficient

# Define the category list
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

# Encode category list
category_embeddings = model.encode(categories, convert_to_tensor=True)

# Read the Excel file
file_path = "event_info_date.xlsx"  # Replace with the actual file path
df = pd.read_excel(file_path)

# Function to find the most relevant categories
def find_categories(row):
    event_text = f"{row['event_name']} {row['content']}"
    event_embedding = model.encode(event_text, convert_to_tensor=True)
    similarities = util.cos_sim(event_embedding, category_embeddings)[0]
    top_indices = similarities.argsort(descending=True)[:3]  # Get top 3 categories
    return ", ".join([categories[i] for i in top_indices])

# Apply the function to the dataframe
df["Category"] = df.apply(find_categories, axis=1)

# Save the updated file
output_path = "updated_file_with_categories.xlsx"
df.to_excel(output_path, index=False)

print(f"Updated file saved to {output_path}")
