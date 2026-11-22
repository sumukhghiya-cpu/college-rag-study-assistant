import os
import time
import requests
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from google import genai

st.set_page_config(page_title="SyllaBuddy", page_icon="📚")

# If Google retires this model, change the name here (the error message tells you the new one)
MODEL_NAME = "gemini-3.8-flash"

# ---------- 1. LOAD + CHUNK ----------
# notes.txt is written so that each paragraph is one self-contained chunk
# (every paragraph names its subject and unit), so we split on blank lines.
def load_chunks(path="notes.txt"):
    text = open(path, encoding="utf-8").read()
    return [c.strip() for c in text.split("\n\n") if c.strip()]

# ---------- 2. EMBED (runs locally, no API key) ----------
@st.cache_resource
def build_index():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunks = load_chunks()
    vectors = model.encode(chunks, normalize_embeddings=True)
    return model, chunks, vectors

# ---------- 3. RETRIEVE (cosine similarity) ----------
def retrieve(question, model, chunks, vectors, k=4):
    q = model.encode([question], normalize_embeddings=True)[0]
    scores = vectors @ q
    top = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in top]

# ---------- 4. PROMPTS (the prompt engineering part) ----------
NOT_FOUND = "I don't know based on the syllabus documents."

PROMPTS = {
    "v1 - basic": "Context:\n{context}\n\n{task}Question: {question}\nAnswer:",

    "v2 - role + grounding rule": (
        "You are an assistant that answers questions about a student's semester syllabus and assignments.\n"
        "Take every syllabus fact (subjects, units, topics, marks, assignments) ONLY from the context below. "
        f"If a fact is not in the context, say \"{NOT_FOUND}\"\n\n"
        "Context:\n{context}\n\n{task}Question: {question}\nAnswer:"
    ),

    "v3 - format + few-shot": (
        "You are a careful study assistant for a B.Tech CSE student.\n"
        "Rules:\n"
        "- Take every syllabus fact (subjects, units, topics, marks, assignments) ONLY from the context.\n"
        f"- If a fact is not in the context, reply exactly: {NOT_FOUND}\n"
        "- Keep answers short (2-4 sentences or a tight list).\n"
        "- End with 'Source: chunk <number>'.\n\n"
        "Example:\n"
        "Question: Which subject covers DNS and HTTP?\n"
        "Answer: Computer Networks (BCO 011A) covers DNS, email, HTTP, FTP, SNMP and the World Wide Web in Unit 5, "
        "the Application Layer. Source: chunk 2\n\n"
        "Context:\n{context}\n\n{task}Question: {question}\nAnswer:"
    ),

    "v4 - SyllaBuddy persona": (
        "You are SyllaBuddy, a friendly, slightly witty senior who has survived this semester and helps juniors "
        "understand their syllabus. Be warm and encouraging, use at most one emoji, and keep humour light.\n"
        "Rules:\n"
        "- Take every syllabus fact (subjects, units, topics, marks, assignments) ONLY from the context.\n"
        f"- If a fact is not in the context, say: {NOT_FOUND} (you may add a friendly line after it).\n"
        "- Keep answers short and clear.\n"
        "- End with 'Source: chunk <number>'.\n\n"
        "Context:\n{context}\n\n{task}Question: {question}\nAnswer:"
    ),
}

MODES = {
    "Chat": "",
    "Explain simply": (
        "Task: Explain the topic the student asks about in very simple language with one everyday analogy, "
        "like explaining to a beginner. You may use general knowledge to explain the concept itself, "
        "but which subject/unit it belongs to must come only from the context.\n\n"
    ),
    "Quiz me": (
        "Task: Create 3 multiple-choice questions (4 options each) on the topic the student names, based on topics "
        "listed in the context. Put the correct answers at the end under 'Answers'.\n\n"
    ),
    "Assignment guide": (
        "Task: The student is asking about an assignment question. Do NOT write the full solution. "
        "Give a short guide: which unit it belongs to, the key concepts to cover, and a suggested structure "
        "for the answer. Take question text and marks only from the context.\n\n"
    ),
}

def ask_llm(prompt, api_key, retries=4):
    client = genai.Client(api_key=api_key)
    last_error = None
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            return resp.text
        except Exception as e:
            last_error = e
            # 503 / 429 = Google is busy or rate limiting: wait a bit and retry
            if "503" in str(e) or "429" in str(e) or "UNAVAILABLE" in str(e):
                time.sleep(3 * (attempt + 1))
                continue
            raise
    raise last_error

GROQ_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"   # if Groq retires it, the app auto-picks another chat model

def ask_groq(prompt, api_key):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    model = st.session_state.get("groq_model", GROQ_MODEL)

    def call(m):
        return requests.post(
            f"{GROQ_URL}/chat/completions", headers=headers, timeout=60,
            json={"model": m, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3},
        )

    r = call(model)
    # model retired/renamed -> pick an available chat model automatically and retry once
    if r.status_code in (400, 404) and "model" in r.text.lower():
        data = requests.get(f"{GROQ_URL}/models", headers=headers, timeout=30).json().get("data", [])
        ids = [m["id"] for m in data]
        bad = ("guard", "whisper", "tts", "safeguard", "embed")
        cands = [i for i in ids if ("llama" in i or "gpt-oss" in i) and not any(b in i for b in bad)]
        if cands:
            model = cands[0]
            st.session_state.groq_model = model
            r = call(model)
    if r.status_code == 429:           # rate limit: wait and retry once
        time.sleep(5)
        r = call(model)
    if r.status_code != 200:
        raise RuntimeError(f"Groq error {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]

def ask_any(provider, prompt, api_key):
    if provider.startswith("Groq"):
        return ask_groq(prompt, api_key)
    return ask_llm(prompt, api_key)

# ---------- 5. CHATBOT UI ----------
st.title("📚 SyllaBuddy")
st.caption(
    "A RAG chatbot that knows my semester syllabus and Python assignments. "
    "Try: *What is in Unit 4 of Operating Systems?* | *Which assignment covers NumPy?* | "
    "*Which subjects do I have?*"
)

provider = st.sidebar.selectbox("LLM provider", ["Groq (free)", "Gemini"])
key_name = "GROQ_API_KEY" if provider.startswith("Groq") else "GEMINI_API_KEY"
api_key = st.sidebar.text_input(f"{provider.split()[0]} API key", type="password", value=os.getenv(key_name, ""))
version = st.sidebar.selectbox("Prompt version", list(PROMPTS.keys()), index=3)
mode = st.sidebar.radio("Mode", list(MODES.keys()))
k = st.sidebar.slider("Chunks to retrieve (k)", 1, 6, 4)
if st.sidebar.button("Clear chat"):
    st.session_state.messages = []

model, chunks, vectors = build_index()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("sources"):
            with st.expander("Retrieved chunks"):
                for i, (c, sc) in enumerate(m["sources"], 1):
                    st.markdown(f"**chunk {i}** (similarity {sc:.2f})")
                    st.write(c)

question = st.chat_input("Ask about your syllabus or assignments")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    results = retrieve(question, model, chunks, vectors, k)
    context = "\n\n".join(f"[chunk {i+1}] {c}" for i, (c, _) in enumerate(results))

    history = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages[-7:-1]
    )
    prompt = PROMPTS[version].format(context=context, question=question, task=MODES[mode])
    if history:
        prompt = "Previous conversation:\n" + history + "\n\n" + prompt

    with st.chat_message("assistant"):
        if not api_key:
            answer = "Add your API key in the sidebar first."
        else:
            try:
                answer = ask_any(provider, prompt, api_key)
            except Exception as e:
                answer = f"Error calling {provider}: {e}"
        st.write(answer)
        with st.expander("Retrieved chunks"):
            for i, (c, sc) in enumerate(results, 1):
                st.markdown(f"**chunk {i}** (similarity {sc:.2f})")
                st.write(c)
        with st.expander("Final prompt sent to the LLM"):
            st.code(prompt)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": results})
