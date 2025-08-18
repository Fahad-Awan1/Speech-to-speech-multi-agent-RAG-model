import os
import io
import time
import tempfile
from pathlib import Path
import logging

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from pydub import AudioSegment
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import docx2txt
from pypdf import PdfReader

# --- Configuration ---
# Use gTTS for Text-to-Speech as it's lightweight and better for deployment.
# Coqui TTS is powerful but requires heavy dependencies (PyTorch) which can
# exceed memory limits on free hosting platforms like Streamlit Community Cloud.
USE_COQUI_TTS = False

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# ---------------------------
# API Key Management
# ---------------------------
# Load environment variables from a .env file if it exists
load_dotenv()


def get_secret(key: str) -> str:
    """
    Retrieves a secret key from environment variables or Streamlit's secrets manager.
    Priority:
    1. Environment variable (e.g., from a local .env file).
    2. Streamlit secrets manager (for deployment).
    """
    secret_value = os.getenv(key)
    if secret_value:
        return secret_value

    # Fallback to Streamlit secrets if not found in env
    try:
        return st.secrets.get(key)
    except Exception:
        return None


ASSEMBLYAI_API_KEY = get_secret("ASSEMBLYAI_API_KEY")
GROQ_API_KEY = get_secret("GROQ_API_KEY")

# Check for essential API keys and stop if not found
if not ASSEMBLYAI_API_KEY:
    st.error(
        "❌ ASSEMBLYAI_API_KEY not found. Please set it in your .env file or Streamlit secrets."
    )
    st.stop()
if not GROQ_API_KEY:
    st.error(
        "❌ GROQ_API_KEY not found. Please set it in your .env file or Streamlit secrets."
    )
    st.stop()


# Conditionally import TTS libraries based on configuration
if USE_COQUI_TTS:
    try:
        from TTS.api import TTS
    except ImportError:
        st.error(
            "Coqui TTS library not found. Please install it with 'pip install TTS'."
        )
        st.stop()
else:
    from gtts import gTTS


# ---------------------------
# Caching for Expensive Models
# ---------------------------
@st.cache_resource
def load_sentence_transformer_model(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
):
    """Loads and caches the SentenceTransformer model."""
    logging.info(f"Loading SentenceTransformer model: {model_name}")
    return SentenceTransformer(model_name)


@st.cache_resource
def load_tts_model():
    """Loads and caches the TTS model if Coqui is used."""
    if USE_COQUI_TTS:
        logging.info("Loading Coqui TTS model...")
        # Note: This will download the model on the first run, which can take time.
        return TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
    return None


# ---------------------------
# Utility Functions
# ---------------------------
def load_text_from_filelike(filename: str, data: bytes) -> str:
    """
    Extracts text content from various file types (PDF, DOCX, TXT, MD).
    Handles temporary file creation and cleanup securely.
    """
    suffix = Path(filename).suffix.lower()
    text = ""
    tmp_path = None
    try:
        if suffix in (".pdf", ".docx"):
            # Create a temporary file to be read by libraries that need a file path
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            if suffix == ".pdf":
                reader = PdfReader(tmp_path)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            elif suffix == ".docx":
                text = docx2txt.process(tmp_path) or ""
        elif suffix in (".txt", ".md"):
            text = data.decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Error processing file {filename}: {e}")
        return ""
    finally:
        # Clean up the temporary file if it was created
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return text


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    """
    Splits a long text into smaller chunks with a specified overlap.
    """
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += size - overlap
    return chunks


# ---------------------------
# Core Logic Classes
# ---------------------------
class SimpleFAISS:
    """A simple wrapper for FAISS vector search."""

    def __init__(self, embedder):
        self.embedder = embedder
        self.texts = []
        self.sources = []
        self.index = None
        self.dim = self.embedder.get_sentence_embedding_dimension()

    def build(self, docs: list[tuple[str, str]]):
        """Builds the FAISS index from a list of document chunks."""
        self.texts = [t for t, _ in docs]
        self.sources = [s for _, s in docs]

        logging.info(f"Generating embeddings for {len(self.texts)} chunks...")
        embeddings = self.embedder.encode(
            self.texts, normalize_embeddings=True, convert_to_numpy=True
        )

        logging.info("Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings.astype(np.float32))
        logging.info("FAISS index built successfully.")

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Performs a similarity search on the FAISS index."""
        if self.index is None:
            raise RuntimeError(
                "Index is not built. Please upload documents and build the index first."
            )

        query_embedding = self.embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )

        distances, indices = self.index.search(query_embedding.astype(np.float32), k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.texts):
                results.append(
                    {
                        "text": self.texts[idx],
                        "source": self.sources[idx],
                        "score": float(score),
                    }
                )
        return results


class AssemblyAI_ASR:
    """Handles Speech-to-Text using the AssemblyAI API."""

    def __init__(self):
        self.api_key = ASSEMBLYAI_API_KEY
        self.base_url = "https://api.assemblyai.com/v2"
        self.headers = {"authorization": self.api_key}

    def _upload_audio(self, wav_bytes: bytes) -> str:
        """Uploads audio data and returns the audio URL."""
        upload_endpoint = f"{self.base_url}/upload"
        response = requests.post(upload_endpoint, headers=self.headers, data=wav_bytes)
        response.raise_for_status()
        return response.json()["upload_url"]

    def transcribe(self, wav_bytes: bytes) -> str:
        """Transcribes audio by uploading and polling for results."""
        logging.info("Uploading audio to AssemblyAI...")
        audio_url = self._upload_audio(wav_bytes)

        transcript_endpoint = f"{self.base_url}/transcript"
        data = {"audio_url": audio_url}
        response = requests.post(transcript_endpoint, json=data, headers=self.headers)
        response.raise_for_status()
        transcript_id = response.json()["id"]

        polling_endpoint = f"{self.base_url}/transcript/{transcript_id}"
        logging.info("Transcription started. Polling for results...")

        while True:
            result = requests.get(polling_endpoint, headers=self.headers).json()
            if result["status"] == "completed":
                logging.info("Transcription completed.")
                return result["text"] or ""
            elif result["status"] == "error":
                logging.error(f"AssemblyAI transcription failed: {result['error']}")
                raise RuntimeError(f"Transcription failed: {result['error']}")
            time.sleep(3)


class RAGGroq:
    """Handles response generation using Groq and RAG."""

    def __init__(self, model: str, temperature: float = 0.2):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = model
        self.temperature = temperature

    def generate(self, query: str, contexts: list[dict]) -> str:
        """Generates an answer based on the query and retrieved contexts."""
        if not contexts:
            return "I could not find any relevant information in the provided documents to answer your question."

        context_block = "\n\n".join(
            f"Context [{i + 1}] from source '{c['source']}':\n{c['text']}"
            for i, c in enumerate(contexts)
        )

        prompt = f"""You are a helpful AI assistant. Your task is to answer the user's question based *only* on the provided context information.
Follow these rules strictly:
1. Base your answer entirely on the text provided in the 'Context' section.
2. Do not use any external knowledge or information you have outside of the provided context.
3. If the answer cannot be found within the provided context, state clearly: "I could not find the answer in the provided documents."
4. Keep your answer concise and to the point.

---
Context:
{context_block}
---

Question: {query}

Answer:"""

        logging.info("Generating answer with Groq LLM...")
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful RAG assistant that answers questions strictly based on the provided context.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Groq API call failed: {e}")
            raise RuntimeError(f"Failed to generate answer from LLM: {e}")


class TTSWrapper:
    """A wrapper for Text-to-Speech synthesis."""

    def __init__(self):
        self.tts_model = load_tts_model() if USE_COQUI_TTS else None

    def synth(self, text: str) -> bytes:
        """Synthesizes text into speech (WAV format)."""
        logging.info(f"Synthesizing speech for text: '{text[:50]}...'")
        if USE_COQUI_TTS:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp_path = tmp.name
                self.tts_model.tts_to_file(text=text, file_path=tmp_path)
                with open(tmp_path, "rb") as f:
                    return f.read()
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:  # Use gTTS
            try:
                tts_obj = gTTS(text=text, lang="en")
                mp3_fp = io.BytesIO()
                tts_obj.write_to_fp(mp3_fp)
                mp3_fp.seek(0)

                # Convert MP3 (from gTTS) to WAV for consistent playback
                mp3_audio = AudioSegment.from_file(mp3_fp, format="mp3")
                wav_fp = io.BytesIO()
                mp3_audio.export(wav_fp, format="wav")
                wav_fp.seek(0)
                return wav_fp.getvalue()
            except Exception as e:
                logging.error(f"gTTS failed: {e}")
                raise RuntimeError(f"Failed to synthesize speech: {e}")


# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    st.set_page_config(
        page_title="Speech-to-Speech Doc Assistant", page_icon="🗣️", layout="centered"
    )
    st.title("🗣️ Speech-to-Speech Document Assistant")
    st.caption(
        "Speak a question about your uploaded documents and get a spoken answer back."
    )

    # --- Sidebar for controls ---
    with st.sidebar:
        st.header("1. Upload & Index Documents")
        files = st.file_uploader(
            "Upload PDF, DOCX, or TXT files",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
        )

        build_btn = st.button("Build Index")

        st.divider()
        st.header("2. Configure Settings")
        top_k = st.slider("Number of context chunks to retrieve", 1, 10, 5, key="top_k")
        llm_model = st.selectbox(
            "Groq LLM Model",
            ["llama3-70b-8192", "mixtral-8x7b-32768", "llama3-8b-8192"],
            index=0,
            key="llm_model",
        )

    # --- Session State Initialization ---
    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    # --- Index Building Logic ---
    if build_btn:
        if not files:
            st.warning("Please upload at least one document to build the index.")
        else:
            with st.spinner(
                "Processing documents and building index... This may take a moment."
            ):
                docs = []
                for f in files:
                    try:
                        raw_bytes = f.read()
                        text = load_text_from_filelike(f.name, raw_bytes)
                        if not text or not text.strip():
                            st.warning(f"No text extracted from '{f.name}'. Skipping.")
                            continue

                        chunks = chunk_text(text, size=800, overlap=120)
                        for chunk in chunks:
                            docs.append((chunk, f.name))
                    except Exception as e:
                        st.error(f"Failed to process {f.name}: {e}")

                if not docs:
                    st.error(
                        "No text could be extracted from the uploaded documents. Please check the files."
                    )
                else:
                    embedder = load_sentence_transformer_model()
                    retriever = SimpleFAISS(embedder)
                    retriever.build(docs)
                    st.session_state.retriever = retriever
                    st.success(
                        f"✅ Index built successfully from {len(docs)} chunks across {len(files)} file(s)."
                    )

    st.divider()

    # --- Main Interaction Area ---
    st.header("3. Ask a Question")

    if not st.session_state.retriever:
        st.info(
            "Please upload documents and build the index using the sidebar to get started."
        )
        return

    # Initialize core components once index is ready
    try:
        asr = AssemblyAI_ASR()
        tts = TTSWrapper()
        generator = RAGGroq(model=st.session_state.llm_model, temperature=0.2)
    except Exception as e:
        st.error(f"Failed to initialize core components: {e}")
        st.stop()

    # Audio recorder widget
    audio_bytes = audio_recorder(
        text="Click the microphone to ask your question",
        pause_threshold=2.0,
        sample_rate=16_000,
        icon_size="2x",
    )

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

        try:
            with st.spinner("1/4 - Transcribing your question..."):
                query_text = asr.transcribe(audio_bytes)
            st.info(f'❓ You asked: *"{query_text}"*')

            if not query_text.strip():
                st.warning("Could not detect any speech. Please try again.")
                return

            with st.spinner("2/4 - Searching for relevant context..."):
                contexts = st.session_state.retriever.search(
                    query_text, k=st.session_state.top_k
                )

            with st.expander("🔎 View Retrieved Context"):
                if not contexts:
                    st.write("No relevant context found.")
                for i, c in enumerate(contexts, 1):
                    st.markdown(
                        f"**{i}. Source: `{c['source']}`** (Score: {c['score']:.3f})"
                    )
                    st.markdown(f"> {c['text'][:500]}...")
                    st.markdown("---")

            with st.spinner("3/4 - Generating the answer..."):
                answer = generator.generate(query_text, contexts)

            with st.spinner("4/4 - Synthesizing the spoken answer..."):
                out_wav = tts.synth(answer)

            st.markdown("### 🔊 Here is your answer:")
            st.audio(out_wav, format="audio/wav")
            with st.expander("📝 View Text Answer"):
                st.write(answer)

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    main()
