import os
import json
import numpy as np
import boto3
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# === CONFIG AWS ===
REGION = "us-east-1"
TEXT_MODEL_ID = "amazon.titan-text-premier-v1:0"
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"

# === CONSTANTES ===
FOLDER_PATH = "notices_txt"
EMBEDDINGS_PATH = "embeddings.json"
MAX_INPUT_TOKENS = 4000
AVG_CHARS_PER_TOKEN = 4

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

def get_top_passages(question_vector, embedded_passages, top_k=3):
    keys = list(embedded_passages.keys())
    vectors = np.array(list(embedded_passages.values()))
    question_vec = np.array(question_vector).reshape(1, -1)
    sims = cosine_similarity(question_vec, vectors)[0]
    top_indices = np.argpartition(sims, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(-sims[top_indices])]
    return [(keys[i], sims[i]) for i in top_indices]

def load_context_passages(top_passages):
    contexts = []
    for key, sim in top_passages:
        parts = key.split("::")
        if len(parts) == 2:
            doc, para_idx = parts
            para_idx = int(para_idx)
        else:
            doc = parts[0]
            para_idx = None

        path = os.path.join(FOLDER_PATH, doc)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                if para_idx is not None:
                    paragraphs = split_paragraphs(f.read())
                    if para_idx < len(paragraphs):
                        contexts.append((doc, paragraphs[para_idx], sim))
                else:
                    contexts.append((doc, f.read(), sim))
    return contexts

def split_paragraphs(text, max_chars=1200):
    paragraphs = []
    current = ""
    for line in text.splitlines():
        if line.strip() == "":
            if current:
                paragraphs.append(current.strip())
                current = ""
        else:
            current += " " + line.strip()
        if len(current) >= max_chars:
            paragraphs.append(current.strip())
            current = ""
    if current:
        paragraphs.append(current.strip())
    return paragraphs

def build_prompt(question, context_chunks, chat_history=None):
    history_context = ""
    if chat_history:
        history_context = "\n\nContexte de la conversation précédente:\n"
        for msg in chat_history[-2:]:
            history_context += f"Q: {msg['question']}\nR: {msg['response']}\n"
    combined_context = "\n\n".join([f"{c}" for c in context_chunks])

    prompt = f"""Tu es un assistant médical francophone. Réponds uniquement en français. Tiens compte de l'historique si présent.

{history_context}

Nouvelle question: {question}

Contexte documentaire pertinent:
{combined_context}

Réponds "Je ne sais pas" si le contexte ne contient pas la réponse. Sinon, donne une réponse claire et concise:"""
    return prompt

def ask_titan(prompt):
    body = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 512,
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

# === GESTION DE L'HISTORIQUE ===
if 'history' not in st.session_state:
    st.session_state.history = []

# === APPLICATION STREAMLIT ===
st.set_page_config(page_title="Assistant Médical", layout="wide")
st.title("💊 Assistant Médical Conversationnel")

@st.cache_data(show_spinner=False)
def load_embeddings():
    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

embedded_passages = load_embeddings()

# Affichage de l'historique
for msg in st.session_state.history:
    with st.chat_message("user"):
        st.write(msg['question'])
    with st.chat_message("assistant"):
        st.write(msg['response'])

if prompt := st.chat_input("Posez votre question médicale:"):
    with st.chat_message("user"):
        st.write(prompt)

    question_vec = embed_text_bedrock(prompt)

    # Recherche de nouveaux passages
    with st.spinner("Recherche des passages les plus pertinents..."):
        top_passages = get_top_passages(question_vec, embedded_passages, top_k=3)
        context_entries = load_context_passages(top_passages)

    # Par défaut, considérer que ce n'est pas un nouveau médicament
    is_new_question = False

    # Vérifier si un passage a une similarité >0.6
    for doc, passage, sim in context_entries:
        if sim > 0.5:
            is_new_question = True
            break

    if is_new_question:
        st.info("La question correspond à un nouveau contexte pertinent.")
        selected_contexts = []
        total_chars = 0
        for doc, passage, sim in context_entries:
            if total_chars + len(passage) > MAX_INPUT_TOKENS * AVG_CHARS_PER_TOKEN:
                break
            selected_contexts.append(f"(Document: {doc}, Similarité: {sim:.2f})\n{passage}")
            total_chars += len(passage)

        current_context = selected_contexts
        current_docs = [doc for doc, _, _ in context_entries]

    else:
        st.info("Pas de contexte suffisamment pertinent : réutilisation du contexte précédent.")
        if st.session_state.history:
            last = st.session_state.history[-1]
            current_context = last["context_chunks"]
            current_docs = last["documents"]
        else:
            st.warning("Aucun contexte précédent disponible. Utilisation du contexte trouvé par défaut.")
            selected_contexts = []
            total_chars = 0
            for doc, passage, sim in context_entries:
                if total_chars + len(passage) > MAX_INPUT_TOKENS * AVG_CHARS_PER_TOKEN:
                    break
                selected_contexts.append(f"(Document: {doc}, Similarité: {sim:.2f})\n{passage}")
                total_chars += len(passage)
            current_context = selected_contexts
            current_docs = [doc for doc, _, _ in context_entries]

    prompt_text = build_prompt(
        prompt,
        current_context,
        st.session_state.history
    )

    with st.spinner("Génération de la réponse..."):
        response_text = ask_titan(prompt_text)

        st.session_state.history.append({
            "question": prompt,
            "response": response_text,
            "documents": current_docs,
            "context_chunks": current_context
        })

        if len(st.session_state.history) > 5:
            st.session_state.history = st.session_state.history[-5:]

        with st.chat_message("assistant"):
            st.write(response_text)
