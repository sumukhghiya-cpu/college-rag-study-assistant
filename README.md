# 📚 SyllaBuddy

### Your AI-powered College Study Assistant

SyllaBuddy is a mini Retrieval-Augmented Generation (RAG) chatbot that helps students explore their college syllabus and assignments using natural-language questions.

## ✨ Features

- Ask questions about your college syllabus and assignments.
- Choose between Groq and Gemini.
- Four different prompt-engineering versions.
- Chat, Explain Simply, Quiz Me and Assignment Guide modes.
- Adjustable document retrieval (top-k chunks).
- View retrieved information and similarity scores.

## 🛠️ Tech Stack

- Python
- Streamlit
- Sentence Transformers
- NumPy
- Groq API
- Google Gemini API

## ⚙️ How It Works

1. Loads academic content from `notes.txt`.
2. Splits content into smaller chunks.
3. Converts chunks into embeddings using Sentence Transformers.
4. Retrieves relevant chunks using cosine similarity.
5. Combines retrieved context with engineered prompts.
6. Generates answers using Groq or Gemini.

## 🚀 Getting Started

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

Enter your Groq or Gemini API key in the sidebar.

## 📌 Project Status

Initial Prototype – Developed as part of a Prompt Engineering academic project.

## 🔮 Future Improvements

- PDF and DOCX document uploads.
- Persistent vector storage.
- Improved document retrieval.
- Better source attribution.

---

**Built with Python, RAG and Prompt Engineering.**
