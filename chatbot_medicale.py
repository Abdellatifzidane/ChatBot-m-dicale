import os
import json
import boto3
import requests
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

# --- CONFIG AWS & OpenSearch ---
region = 'us-east-1'  # Modifie ta région
service = 'es'
opensearch_host = 'votre-endpoint-opensearch'  # ex: search-xxx.us-east-1.es.amazonaws.com

# AWS Auth pour OpenSearch
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)

# OpenSearch client
client = OpenSearch(
    hosts=[{'host': opensearch_host, 'port': 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

# Bedrock client (embedding + génération)
bedrock = boto3.client('bedrock-runtime', region_name=region)

INDEX_NAME = 'notices-vector-index'

# --- Fonctions utilitaires ---

def create_index():
    index_body = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 512
            }
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 768
                }
            }
        }
    }
    if not client.indices.exists(INDEX_NAME):
        client.indices.create(index=INDEX_NAME, body=index_body)
        print(f"Index '{INDEX_NAME}' créé.")
    else:
        print(f"Index '{INDEX_NAME}' existe déjà.")

def get_embedding(text):
    response = bedrock.invoke_model(
        modelId="amazon.titan-tc-large",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text})
    )
    result = json.loads(response['body'].read())
    return result['embedding']

def index_chunk(doc_id, text, embedding):
    body = {"text": text, "embedding": embedding}
    client.index(index=INDEX_NAME, id=doc_id, body=body)

def chunk_text(text, size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        chunks.append(chunk)
        if i + size >= len(words):
            break
    return chunks

def load_documents(folder):
    docs = []
    files = []
    for file in os.listdir(folder):
        if file.endswith('.txt'):
            path = os.path.join(folder, file)
            with open(path, 'r', encoding='utf-8') as f:
                docs.append(f.read())
                files.append(file)
    return docs, files

def search_similar(question_embedding, top_k=3):
    query = {
        "size": top_k,
        "query": {
            "knn": {
                "field": "embedding",
                "query_vector": question_embedding,
                "k": top_k,
                "num_candidates": 100
            }
        }
    }
    res = client.search(index=INDEX_NAME, body=query)
    hits = res['hits']['hits']
    return [hit['_source']['text'] for hit in hits]

def generate_answer(question, contexts):
    context_text = "\n\n".join(contexts)
    prompt = f"""
Tu es un assistant médical francophone. Utilise uniquement les informations suivantes extraites de la notice :

{context_text}

Réponds clairement à la question suivante :

{question}
"""
    payload = {
        "modelId": "anthropic.claude-v2",
        "contentType": "application/json",
        "accept": "application/json",
        "body": json.dumps({
            "prompt": prompt,
            "maxTokensToSample": 512,
            "temperature": 0.2
        })
    }
    response = bedrock.invoke_model(**payload)
    output = json.loads(response['body'].read())
    return output['completions'][0]['text']

# --- Pipeline complet ---

def index_all_documents(folder_path):
    print("Création de l'index OpenSearch si besoin...")
    create_index()
    docs, files = load_documents(folder_path)
    doc_counter = 0
    for doc, filename in zip(docs, files):
        chunks = chunk_text(doc)
        print(f"Indexation de {len(chunks)} chunks du fichier {filename}...")
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            doc_id = f"{filename}-{i}"
            index_chunk(doc_id, chunk, embedding)
            doc_counter += 1
            if doc_counter % 10 == 0:
                print(f"{doc_counter} chunks indexés...")
    print("Indexation terminée.")

def answer_question(question):
    print("Embedding de la question...")
    q_emb = get_embedding(question)
    print("Recherche des passages pertinents dans OpenSearch...")
    relevant_chunks = search_similar(q_emb, top_k=3)
    print("Génération de la réponse...")
    answer = generate_answer(question, relevant_chunks)
    return answer

# --- Exemple d'utilisation ---

if __name__ == "__main__":
    FOLDER_PATH = "notices_txt"  # dossier avec tes fichiers .txt

    # Indexe les documents (faire une fois au début)
    index_all_documents(FOLDER_PATH)

    # Test question
    question = "Puis-je prendre du paracétamol avec de l'alcool ?"
    response = answer_question(question)
    print("\nRéponse du chatbot :\n", response)
