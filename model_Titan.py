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
MAX_TOKENS = 10000
TOKEN_RATIO = 3.5

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

def get_top_filenames(question_vector, embedded_docs, top_k=3):
    filenames = list(embedded_docs.keys())
    vectors = np.array(list(embedded_docs.values()))
    question_vec = np.array(question_vector).reshape(1, -1)
    sims = cosine_similarity(question_vec, vectors)[0]
    top_indices = np.argpartition(sims, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(-sims[top_indices])]
    return [filenames[i] for i in top_indices], [sims[i] for i in top_indices]

def load_context_from_files(filenames):
    contexts = []
    for filename in filenames:
        path = os.path.join(FOLDER_PATH, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                contexts.append(f.read())
    return contexts

def chunk_text(text, max_chars=None):
    if max_chars is None:
        max_chars = round(MAX_TOKENS * TOKEN_RATIO)
    max_chars = int(max_chars)
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

def build_prompt(question, chunks, chat_history=None):
    prompts = []
    history_context = ""
    
    if chat_history:
        history_context = "\n\nContexte de la conversation précédente:\n"
        for msg in chat_history[-3:]:  # Garder les 3 derniers échanges
            history_context += f"Q: {msg['question']}\nR: {msg['response']}\n"
    
    for i, chunk in enumerate(chunks):
        prompt = f"""Tu es un assistant médical francophone, réponds uniquement en francais. Tiens compte de l'historique de conversation suivant pour répondre à la nouvelle question.

{history_context}

Nouvelle question: {question}

Contexte documentaire (partie {i+1}/{len(chunks)}):
{chunk}

Réponds "Je ne sais pas" si le contexte ne contient pas la réponse. Sinon, donne une réponse concise en tenant compte de l'historique:"""
        prompts.append(prompt)
    return prompts

def ask_titan_qa(prompts):
    responses = []
    for prompt in prompts:
        body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 512,
                "temperature": 0.5,
                "topP": 0.9
            }
        }
        response = bedrock.invoke_model(
            modelId=TEXT_MODEL_ID,
            body=json.dumps(body),
            contentType="application/json"
        )
        result = json.loads(response["body"].read())
        responses.append(result["results"][0]["outputText"].strip())
    return responses

def find_best_response(responses):
    valid_responses = [r for r in responses if r.lower() not in ["je ne sais pas", "je ne trouve pas"]]
    if not valid_responses:
        return "Je n'ai pas trouvé de réponse dans les documents consultés.", -1
    return valid_responses[0], responses.index(valid_responses[0])

# === GESTION DE L'HISTORIQUE ===
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_docs' not in st.session_state:
    st.session_state.current_docs = None
if 'current_context' not in st.session_state:
    st.session_state.current_context = None

def add_to_history(question, response, documents, context):
    st.session_state.history.append({
        'question': question,
        'response': response,
        'documents': documents,
        'context': context
    })

# === APPLICATION STREAMLIT ===
st.set_page_config(page_title="Assistant Médical", layout="wide")
st.title("💊 Assistant Médical Conversationnel")

@st.cache_data(show_spinner=False)
def load_embeddings():
    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

embedded_docs = load_embeddings()

# Affichage de l'historique de chat
for msg in st.session_state.history:
    with st.chat_message("user"):
        st.write(msg['question'])
    with st.chat_message("assistant"):
        st.write(msg['response'])

# Sidebar pour les détails
with st.sidebar:
    st.header("Détails techniques")
    if st.session_state.history:
        last_query = st.session_state.history[-1]
        st.write("**Dernière question:**", last_query['question'])
        st.write("**Documents utilisés:**")
        for doc in last_query['documents']:
            st.write(f"- {doc}")
        
        if st.checkbox("Afficher le contexte complet"):
            st.text_area("Contexte:", last_query['context'], height=300)

# Gestion de la nouvelle question
if prompt := st.chat_input("Posez votre question médicale:"):
    with st.chat_message("user"):
        st.write(prompt)
    
    # Vérifie si c'est un suivi de conversation
    is_follow_up = len(st.session_state.history) > 0 and st.session_state.current_docs is not None
    
    if not is_follow_up:
        with st.spinner("Recherche des notices pertinentes..."):
            question_vector = embed_text_bedrock(prompt)
            top_filenames, sim_scores = get_top_filenames(question_vector, embedded_docs, top_k=3)
            contexts = load_context_from_files(top_filenames)
            full_context = "\n\n".join(contexts)
            st.session_state.current_docs = [f"{f} (similarité: {s:.2f})" for f, s in zip(top_filenames, sim_scores)]
            st.session_state.current_context = full_context
    else:
        full_context = st.session_state.current_context
    
    chunks = chunk_text(full_context)
    
    with st.spinner("Analyse en cours..."):
        chat_history = st.session_state.history if is_follow_up else None
        prompts = build_prompt(prompt, chunks, chat_history)
        responses = ask_titan_qa(prompts)
        best_response, best_chunk_idx = find_best_response(responses)
        
        add_to_history(prompt, best_response, st.session_state.current_docs, full_context)
        
        with st.chat_message("assistant"):
            st.write(best_response)
            
            with st.expander("Détails de la réponse"):
                st.write("**Documents de référence:**")
                for doc in st.session_state.current_docs:
                    st.write(f"- {doc}")
                
                if best_chunk_idx >= 0:
                    st.write("**Extrait pertinent:**")
                    st.text(chunks[best_chunk_idx][:500] + "...")