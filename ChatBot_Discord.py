import discord
from discord.ext import commands
from discord.ui import Button, View
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import boto3

#  CONFIGURATION AWS ET TITAN 
REGION = "us-east-1"
TEXT_MODEL_ID = "amazon.titan-text-premier-v1:0"
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
FOLDER_PATH = "notices_txt"
EMBEDDINGS_PATH = "embeddings.json"
MAX_CONTEXT_CHARS = 9000

bedrock = boto3.client("bedrock-runtime", region_name=REGION)

#  FONCTIONS UTILITAIRES 
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
    return "\n\n".join(contexts)[:MAX_CONTEXT_CHARS]

def build_single_prompt(question, context, chat_history=None):
    history_context = ""
    if chat_history:
        history_context = "\n\nVoici l'historique récent de la conversation :\n"
        for msg in chat_history[-1:]:
            history_context += f"Patient : {msg['question']}\nAssistant : {msg['response']}\n"

    prompt = f"""
Tu es un assistant médical francophone, à la fois chaleureux, empathique et très professionnel.
Tu analyses exclusivement les notices des médicaments fournies pour répondre.
Utilise un ton respectueux, bienveillant et rassurant.
Formule des réponses claires, complètes, polies et faciles à comprendre, même pour des non-professionnels de santé.

Si la question est ambiguë (ex: "oui", "pourquoi ?", "et ?"), reformule à partir du dernier échange pour aider le patient.
Si l'information n'est pas présente dans les notices fournies, réponds :
"Je suis désolé, je ne trouve pas cette information dans les notices médicales que j'ai consultées. Veux-tu que je cherche autre chose ?"

{history_context}

Voici les extraits des notices médicales que tu as pu consulter :
{context}

Dernière question du patient : {question}

Ta réponse en français :
"""
    return prompt

def ask_titan_single(prompt):
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
    return result["results"][0]["outputText"].strip()

#  INITIALISATION DISCORD BOT 
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
    embedded_docs = json.load(f)

user_histories = {}

# COMMANDE !ask 
@bot.command()
async def ask(ctx, *, question):
    user_id = ctx.author.id
    history = user_histories.get(user_id, [])

    status_msg = await ctx.send(f"🔍 Je regarde les notices pour toi, une seconde...")

    question_vector = embed_text_bedrock(question)
    top_filenames, sim_scores = get_top_filenames(question_vector, embedded_docs, top_k=3)
    full_context = load_context_from_files(top_filenames)

    prompt = build_single_prompt(question, full_context, chat_history=history)
    best_response = ask_titan_single(prompt)

    documents = [f"{f} (similarité: {s:.2f})" for f, s in zip(top_filenames, sim_scores)]
    history.append({
        'question': question,
        'response': best_response,
        'documents': documents,
        'context': full_context
    })
    user_histories[user_id] = history

    embed = discord.Embed(title="💊 BOT Médical", description=best_response, color=0x00ff99)
    embed.add_field(name="📂 Notices consultées", value="\n".join(documents), inline=False)
    embed.set_footer(text="Clique sur le bouton pour afficher le contexte complet.")

    class ContextView(View):
        def __init__(self, context):
            super().__init__()
            self.context = context

        @discord.ui.button(label="Voir le contexte complet", style=discord.ButtonStyle.primary)
        async def show_context(self, interaction: discord.Interaction, button: Button):
            preview = self.context[:1900] + ("..." if len(self.context) > 1900 else "")
            await interaction.response.send_message(f"📜 Contexte utilisé :\n```{preview}```", ephemeral=False)

    await status_msg.edit(content=None, embed=embed, view=ContextView(full_context))

#  COMMANDE !reset 
@bot.command()
async def reset(ctx):
    user_id = ctx.author.id
    user_histories[user_id] = []
    await ctx.send("🔄 J'ai bien réinitialisé ton historique. Pose-moi une nouvelle question quand tu veux !")

# COMMANDE !history 
@bot.command()
async def history(ctx):
    user_id = ctx.author.id
    history = user_histories.get(user_id, [])
    if not history:
        await ctx.send("📝 Ton historique est vide.")
    else:
        text = "\n".join([f"Q: {msg['question']}\nR: {msg['response']}" for msg in history[-5:]])
        await ctx.send(f"📝 Voici tes derniers échanges :\n```{text[:1900]}```")

# DÉMARRAGE DU BOT


TOKEN = "MTM2NDg0NzEwMTI0NjQzOTQ5NQ.GisVgw.seRvo1Utx5REzSZVhUYjpvPBnVkYB6lNeDrgbg"
bot.run(TOKEN)
