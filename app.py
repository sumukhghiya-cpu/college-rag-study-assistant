import os
import time
import requests
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from google import genai

st.set_page_config(page_title="SyllaBuddy | Study smarter", page_icon="📚", layout="wide")

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

# Frontend-only styling: the RAG pipeline and provider functions above remain unchanged.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --sb-bg:#0b1020; --sb-panel:#111a2e; --sb-line:rgba(148,163,184,.16); --sb-muted:#9aa9c2; --sb-text:#edf3ff; --sb-accent:#8b7cff; --sb-accent2:#53c7e8; }
.stApp { background: radial-gradient(ellipse at 75% -10%, rgba(91,78,190,.20), transparent 38%), var(--sb-bg); color:var(--sb-text); }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#10182b 0%,#0d1425 100%); border-right:1px solid var(--sb-line); }
[data-testid="stSidebar"] > div { padding-top:1.5rem; }
.block-container { max-width:1120px; padding-top:2.2rem; padding-bottom:3rem; }
.sb-brand { display:flex; align-items:center; gap:14px; margin:0 0 1.5rem 0; }
.sb-logo { width:48px; height:48px; border-radius:16px; display:flex; align-items:center; justify-content:center; font-size:24px; background:linear-gradient(135deg,#7567f8,#36b9d7); box-shadow:0 8px 28px rgba(103,91,238,.25); }
.sb-brand h1 { font-family:'Manrope',sans-serif; font-size:30px; line-height:1.1; margin:0; color:#f6f8ff; letter-spacing:-1.2px; }
.sb-brand p { margin:5px 0 0 0; color:var(--sb-muted); font-size:13px; }
.sb-hero { padding:26px 28px; border:1px solid rgba(139,124,255,.24); border-radius:24px; background:linear-gradient(115deg,rgba(36,47,83,.92),rgba(20,32,55,.88) 55%,rgba(33,55,78,.82)); box-shadow:0 18px 55px rgba(0,0,0,.16); margin-bottom:24px; }
.sb-eyebrow { color:#a9a0ff; text-transform:uppercase; letter-spacing:1.8px; font-size:11px; font-weight:700; margin-bottom:10px; }
.sb-hero h2 { font-family:'Manrope',sans-serif; font-size:29px; line-height:1.25; margin:0 0 9px 0; color:#f6f8ff; letter-spacing:-.8px; }
.sb-hero p { color:#b9c6dc; margin:0; font-size:14px; line-height:1.65; max-width:700px; }
.sb-pill { display:inline-block; border:1px solid rgba(120,220,204,.25); background:rgba(52,211,153,.08); color:#80e2c4; border-radius:99px; padding:5px 10px; font-size:11px; font-weight:600; margin-top:16px; }
.sb-section-label { color:#aab7cd; text-transform:uppercase; letter-spacing:1.5px; font-size:11px; font-weight:700; margin:18px 0 10px 0; }
.sb-card { height:100%; min-height:112px; border:1px solid var(--sb-line); border-radius:17px; background:linear-gradient(145deg,rgba(24,36,59,.95),rgba(17,26,44,.95)); padding:17px 18px; }
.sb-card .emoji { font-size:21px; margin-bottom:10px; }
.sb-card h4 { color:#eef3ff; font-size:14px; margin:0 0 5px 0; font-weight:700; }
.sb-card p { color:#9eacc3; font-size:12px; line-height:1.5; margin:0; }
[data-testid="stChatMessage"] { background:rgba(20,30,50,.72); border:1px solid var(--sb-line); border-radius:18px; padding:17px 19px; margin-bottom:13px; }
[data-testid="stChatMessage"] p { line-height:1.7; }
[data-testid="stChatInput"] { border:1px solid rgba(139,124,255,.35); border-radius:18px; background:#121c31; box-shadow:0 8px 30px rgba(0,0,0,.18); }
[data-testid="stChatInput"] textarea { color:#f1f5ff !important; }
.stButton > button { border-radius:12px; border:1px solid rgba(139,124,255,.28); background:rgba(119,105,238,.10); color:#dcd8ff; font-weight:600; transition:all .18s ease; }
.stButton > button:hover { border-color:#8b7cff; background:rgba(139,124,255,.20); color:white; transform:translateY(-1px); }
.stSelectbox > div > div, .stTextInput > div > div, .stSlider > div { border-radius:11px; }
[data-testid="stExpander"] { border:1px solid var(--sb-line); border-radius:13px; background:rgba(15,24,42,.55); }
hr { border-color:var(--sb-line); }
.sb-footer { color:#71819c; text-align:center; font-size:11px; margin-top:32px; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#aab7cd; }
</style>
""", unsafe_allow_html=True)

# Sidebar branding and controls
with st.sidebar:
    st.markdown("<div class='sb-brand'><div class='sb-logo'>📚</div><div><h1>SyllaBuddy</h1><p>Your semester, made simpler.</p></div></div>", unsafe_allow_html=True)
    st.markdown('<div class="sb-section-label">AI settings</div>', unsafe_allow_html=True)
    provider = st.selectbox("LLM provider", ["Groq (free)", "Gemini"])
    key_name = "GROQ_API_KEY" if provider.startswith("Groq") else "GEMINI_API_KEY"
   api_key = st.secrets.get(key_name, os.getenv(key_name, ""))
    st.markdown("---")
    st.markdown('<div class="sb-section-label">Personalise your session</div>', unsafe_allow_html=True)
    version = st.selectbox("Prompt style", list(PROMPTS.keys()), index=3)
    mode = st.radio("Learning mode", list(MODES.keys()), horizontal=False)
    k = st.slider("Retrieved context chunks", 1, 6, 4)
    st.markdown("---")
    if st.button("↻  Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("suggested_question", None)
        st.rerun()
    st.markdown('<div class="sb-footer">Built for curious minds · Powered by RAG</div>', unsafe_allow_html=True)

model, chunks, vectors = build_index()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Main header
st.markdown("""
<div class='sb-hero'>
  <div class='sb-eyebrow'>✦ Your personal academic sidekick</div>
  <h2>Study smarter. Understand deeper.</h2>
  <p>Ask questions about your semester syllabus and assignments. SyllaBuddy finds relevant material from your notes and turns it into clear, helpful answers.</p>
  <span class='sb-pill'>● Knowledge base connected</span>
</div>
""", unsafe_allow_html=True)

# Suggested prompts are shown only before the first message.
if not st.session_state.messages:
    st.markdown("<div class='sb-section-label'>Not sure where to start? Try asking</div>", unsafe_allow_html=True)
    suggestions = [
        ("🧠", "Explore a unit", "What is in Unit 4 of Operating Systems?"),
        ("🐍", "Python assignments", "Which assignment covers NumPy?"),
        ("📖", "My subjects", "Which subjects do I have?"),
        ("🎯", "Test my knowledge", "Create a quiz on Computer Networks."),
    ]
    c1, c2 = st.columns(2)
    for idx, (emoji, title, prompt_text) in enumerate(suggestions):
        with (c1 if idx % 2 == 0 else c2):
            st.markdown(f"<div class='sb-card'><div class='emoji'>{emoji}</div><h4>{title}</h4><p>{prompt_text}</p></div>", unsafe_allow_html=True)
            if st.button("Ask this  →", key=f"suggest_{idx}", use_container_width=True):
                st.session_state.suggested_question = prompt_text
                st.rerun()

# Render conversation history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("sources"):
            with st.expander("📎 View retrieved context"):
                for i, (c, sc) in enumerate(m["sources"], 1):
                    st.markdown(f"**Retrieved passage {i}** · Similarity `{sc:.2f}`")
                    st.write(c)

question = st.chat_input("Ask SyllaBuddy about your syllabus or assignments…")
if not question and st.session_state.get("suggested_question"):
    question = st.session_state.pop("suggested_question")

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
        with st.expander("📎 View retrieved context"):
            for i, (c, sc) in enumerate(results, 1):
                st.markdown(f"**Retrieved passage {i}** · Similarity `{sc:.2f}`")
                st.write(c)
        with st.expander("⚙️ View final prompt sent to the LLM"):
            st.code(prompt)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": results})

st.markdown("<div class='sb-footer'>SyllaBuddy · Learn at your own pace</div>", unsafe_allow_html=True)
