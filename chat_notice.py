import os
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

FOLDER_PATH = "notices_txt"
MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"

model = SentenceTransformer(MODEL_NAME)

def load_documents(folder_path):
    docs = []
    filenames = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            path = os.path.join(folder_path, file)
            with open(path, "r", encoding="utf-8") as f:
                docs.append(f.read())
                filenames.append(file)
    return docs, filenames

def embed_text(text):
    return model.encode([text])[0]

def find_best_doc(question, docs, doc_embeddings):
    question_embedding = embed_text(question)
    similarities = cosine_similarity([question_embedding], doc_embeddings)[0]
    best_idx = np.argmax(similarities)
    return docs[best_idx]

def chunk_text(text, size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        chunks.append(chunk)
        if i + size >= len(words):
            break
    return chunks

def find_best_chunk(question, chunks):
    question_embedding = embed_text(question)
    chunk_embeddings = [embed_text(chunk) for chunk in chunks]
    similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]
    top_idx = int(np.argmax(similarities))
    return chunks[top_idx]

def extract_relevant_section(text, keywords=["effets indésirables", "effets secondaires"]):
    pattern = "|".join(re.escape(k) for k in keywords)
    matches = re.finditer(pattern, text.lower())
    sections = []
    for match in matches:
        start = match.start()
        section = text[start:start + 3000]  # extrait un bout raisonnable
        sections.append(section)
    return sections if sections else [text]

def ask_ollama(question, context):
    prompt = f"""
[INST]
Tu es un assistant **francophone** chargé de répondre à des questions sur les médicaments en te basant **exclusivement sur le contenu de la notice fourni ci-dessous**.

🧾 Consignes obligatoires :
- Réponds **en français uniquement**.
- Ne donne une réponse que **si l’information figure dans la notice**.
- ❌ N’invente rien. Tu peux **résumer légèrement**, mais tu dois **rester fidèle à la notice**.
- ❌ Ne fais **aucune référence à la notice elle-même**, à ses sections, ni à un professionnel de santé.
- Si l’information demandée n’est pas dans le contexte, réponds : **"L'information demandée n’est pas disponible dans le contexte fourni."**

Notice :
{context}
[FIN DU CONTEXTE]

Question : {question}
[/INST]
"""
    payload = {
        "model": "nous-hermes2",
        "prompt": prompt,
        "temperature": 0.2,
        "top_p": 0.9,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        if response.status_code == 200:
            return response.json()["response"].strip()
        return f"Erreur Ollama ({response.status_code}) : {response.text}"
    except Exception as e:
        return f"Erreur de connexion à Ollama : {e}"

# 🌐 Interface Streamlit
st.set_page_config(page_title="Assistant Médicament 💊", layout="wide")
st.title("🤖 Assistant Médical basé sur les notices")

question = st.text_input("Pose ta question ici (en français)")

if question:
    with st.spinner("🔍 Recherche du document pertinent..."):
        documents, filenames = load_documents(FOLDER_PATH)
        doc_embeddings = [embed_text(doc) for doc in documents]
        best_doc = find_best_doc(question, documents, doc_embeddings)

    if best_doc:
        st.success("📄 Document trouvé")
        with st.expander("📑 Afficher un extrait de la notice"):
            st.text(best_doc[:2000])  # Pour inspection

        with st.spinner("🔬 Recherche du passage le plus pertinent..."):
            sections = extract_relevant_section(best_doc)
            all_chunks = []
            for section in sections:
                all_chunks.extend(chunk_text(section))
            best_chunk = find_best_chunk(question, all_chunks)

        with st.spinner("💬 Génération de la réponse..."):
            response = ask_ollama(question, best_chunk)
        st.markdown("### 🧠 Réponse")
        st.write(response)
    else:
        st.warning("Aucun document pertinent trouvé pour cette question.")