from flask import Flask, request, jsonify
from openai import OpenAI
import numpy as np

app = Flask(__name__)

client = OpenAI()

# Define categories
categories = [
    "Engineering", "Energy", "Design", "Artificial Intelligence", "Business", "Health"
    # (shortened for demo — use your full list)
]

# Precompute category embeddings (load from file if available)
category_embeddings = []
for cat in categories:
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=cat
    ).data[0].embedding
    category_embeddings.append(emb)
category_embeddings = np.array(category_embeddings)
print("✅ Category embeddings ready.")


@app.route("/ai/", methods=["GET"])
def health():
    return jsonify({"status": "OK", "message": "Conference Classifier running"})


@app.route("/ai/classify", methods=["POST"])
def classify():
    data = request.get_json(force=True)
    event_name = data.get("event_name", "")
    content = data.get("content", "")
    keywords = data.get("keywords", "")

    text = f"{event_name} {content} {keywords}".strip()
    if not text:
        return jsonify({"error": "Please provide event_name, content, or keywords"}), 400

    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding
    emb = np.array(emb)

    similarities = np.dot(category_embeddings, emb) / (
        np.linalg.norm(category_embeddings, axis=1) * np.linalg.norm(emb)
    )

    top_indices = similarities.argsort()[-3:][::-1]
    results = [{"category": categories[i], "score": float(similarities[i])} for i in top_indices]

    return jsonify({"top_categories": results})


if __name__ == "__main__":
    # IMPORTANT: Use port 8080 for Choreo
    app.run(host="0.0.0.0", port=8080)
