import os
import json
import numpy as np
import boto3
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# === CONFIG AWS ===
REGION = "us-east-1"
TEXT_MODEL_ID = "amazon.titan-text-lite-v1"
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"

# === CONSTANTES ===
FOLDER_PATH = "notices_txt"
EMBEDDINGS_PATH = "embeddings.json"
MAX_CONTEXT_CHARS = 15000  # Limite stricte de caractères

# === INIT BOTO3 CLIENT ===
bedrock = boto3.client("bedrock-runtime", region_name=REGION)

# === FONCTIONS UTILITAIRES ===
def embed_text_bedrock(text):
    body = {"inputText": text}
    response = bedrock.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=json.dumps(body),
        contentType="application/json"
    )
    result = json.loads(response["body"].read())
    return result["embedding"]

def get_top_filename(question_vector, embedded_docs):
    filenames = list(embedded_docs.keys())
    vectors = np.array(list(embedded_docs.values()))
    question_vec = np.array(question_vector).reshape(1, -1)
    sims = cosine_similarity(question_vec, vectors)[0]
    top_index = np.argmax(sims)
    return filenames[top_index]

def load_context_from_file(filename):
    path = os.path.join(FOLDER_PATH, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None

def build_prompt(question, context):
    # Tronquer le contexte si nécessaire
    context = context[:MAX_CONTEXT_CHARS]
    return f"""Tu es un assistant médical francophone. Réponds à la question en utilisant uniquement le contexte fourni.

Question: {question}

Contexte:
{context}

Réponse concise:"""

def ask_titan_qa(prompt):
    body = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount":1024,
            "temperature": 0.3,
            "topP": 0.9
        }
    }
    response = bedrock.invoke_model(
        modelId=TEXT_MODEL_ID,
        body=json.dumps(body),
        contentType="application/json"
    )
    result = json.loads(response["body"].read())
    return result["results"][0]["outputText"].strip()

# === APPLICATION STREAMLIT ===
st.set_page_config(page_title="Assistant Médical Simple", layout="wide")
st.title("💊 Assistant Médical (Version Simple)")

@st.cache_data(show_spinner=False)
def load_embeddings():
    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

embedded_docs = load_embeddings()

question = st.text_input("Posez votre question médicale:")

if question:
    with st.spinner("Recherche de la notice la plus pertinente..."):
        question_vector = embed_text_bedrock(question)
        top_filename = get_top_filename(question_vector, embedded_docs)
        context = load_context_from_file(top_filename)

    if context:
        with st.spinner("Préparation de la réponse..."):
            prompt = build_prompt(question, context)
            response = ask_titan_qa(prompt)
            
            st.subheader("Réponse:")
            st.write(response)
            
            with st.expander("Notice utilisée"):
                st.write(top_filename)
                st.text(context[:1000] + "...")  # Aperçu du contexte
    else:
        st.error("Aucune notice trouvée pour répondre à la question.")