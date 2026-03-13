import base64
import time

import streamlit as st
from PyPDF2 import PdfReader

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ---------------------------
# Configuración general
# ---------------------------
st.set_page_config(page_title="EAFIT RAG Lab", layout="wide")
st.title("EAFIT RAG Lab")
st.caption("Comparación empírica entre Zero-shot y RAG con Groq + Streamlit")

# ---------------------------
# Secrets
# ---------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)
if not GROQ_API_KEY:
    st.error("No encontré GROQ_API_KEY en Streamlit Secrets.")
    st.stop()

# ---------------------------
# Sidebar: hiperparámetros
# ---------------------------
st.sidebar.header("Hiperparámetros")

MODEL_OPTIONS = {
    "Llama-3-70b": "llama-3.3-70b-versatile",
    "Mixtral-8x7b": "mixtral-8x7b-32768",
}

selected_label = st.sidebar.selectbox("Model Select", list(MODEL_OPTIONS.keys()))
selected_model = MODEL_OPTIONS[selected_label]

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
chunk_size = st.sidebar.slider("Chunk Size", 20, 2000, 800, 20)
top_k = st.sidebar.slider("Top-K", 1, 10, 3, 1)

# Parámetros default para RAG estándar
DEFAULT_CHUNK_SIZE = 500
DEFAULT_TOP_K = 3
DEFAULT_TEMPERATURE = 0.2

st.sidebar.markdown("---")
st.sidebar.write("**OCR / visión:**")
st.sidebar.write("Modelo sugerido por la guía: `llama-3.2-11b-vision-preview`")
st.sidebar.write("Si no está disponible en tu cuenta, usa otro modelo de visión soportado por Groq.")

# ---------------------------
# Utilidades
# ---------------------------
def get_llm(model_name: str, temp: float):
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model_name,
        temperature=temp,
    )

def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)

def image_to_data_url(uploaded_file) -> str:
    mime_type = uploaded_file.type or "image/png"
    file_bytes = uploaded_file.read()
    encoded = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"

def extract_text_from_image_with_groq(uploaded_file) -> str:
    """
    OCR aproximado usando un modelo de visión de Groq.
    Si el modelo no está disponible en tu cuenta, cámbialo.
    """
    vision_model = "llama-3.2-11b-vision-preview"
    data_url = image_to_data_url(uploaded_file)

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=vision_model,
        temperature=0.0,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Extrae TODO el texto visible de la imagen. "
                        "Devuelve únicamente el texto extraído, respetando saltos de línea. "
                        "Si hay tablas, reprodúcelas en texto plano."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                },
            ],
        }
    ]

    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)

def split_into_chunks(text: str, csize: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=csize,
        chunk_overlap=max(20, int(csize * 0.15)),
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks if chunk.strip()]

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def build_vectorstore(docs):
    embeddings = load_embeddings()
    return FAISS.from_documents(docs, embeddings)

def run_zero_shot(question: str, model_name: str, temp: float) -> str:
    llm = get_llm(model_name, temp)
    prompt = f"""
Responde de forma clara y precisa la siguiente pregunta.

Pregunta:
{question}
"""
    response = llm.invoke(prompt)
    return response.content

def run_rag(question: str, retriever_docs, model_name: str, temp: float) -> str:
    llm = get_llm(model_name, temp)
    context = "\n\n".join([doc.page_content for doc in retriever_docs])

    prompt = f"""
Responde usando SOLO el contexto proporcionado.
Si la respuesta no está en el contexto, di explícitamente:
"No encontré la respuesta en el documento cargado."

Contexto:
{context}

Pregunta:
{question}
"""
    response = llm.invoke(prompt)
    return response.content

# ---------------------------
# Interfaz principal
# ---------------------------
uploaded_file = st.file_uploader(
    "Sube un archivo PDF o una imagen con texto",
    type=["pdf", "png", "jpg", "jpeg", "webp"]
)

question = st.text_area("Escribe tu pregunta", height=100)

if st.button("Procesar y comparar"):
    if uploaded_file is None:
        st.warning("Debes subir un archivo.")
        st.stop()

    if not question.strip():
        st.warning("Debes escribir una pregunta.")
        st.stop()

    ext = uploaded_file.name.lower()

    with st.spinner("Extrayendo texto del archivo..."):
        try:
            if ext.endswith(".pdf"):
                raw_text = extract_text_from_pdf(uploaded_file)
                ingestion_method = "PyPDF2"
            else:
                raw_text = extract_text_from_image_with_groq(uploaded_file)
                ingestion_method = "Groq Vision OCR"
        except Exception as e:
            st.error(f"Error durante la ingesta/OCR: {e}")
            st.stop()

    if not raw_text.strip():
        st.error("No se pudo extraer texto del archivo.")
        st.stop()

    with st.spinner("Construyendo índices para RAG estándar y RAG optimizado..."):
        # RAG estándar
        default_docs = split_into_chunks(raw_text, DEFAULT_CHUNK_SIZE)
        default_vectorstore = build_vectorstore(default_docs)
        default_retrieved_docs = default_vectorstore.similarity_search(question, k=DEFAULT_TOP_K)

        # RAG optimizado
        optimized_docs = split_into_chunks(raw_text, chunk_size)
        optimized_vectorstore = build_vectorstore(optimized_docs)
        optimized_retrieved_docs = optimized_vectorstore.similarity_search(question, k=top_k)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("LLM Simple")
        st.caption("Inferencia sin contexto")
        start = time.time()
        try:
            zero_shot_answer = run_zero_shot(question, selected_model, DEFAULT_TEMPERATURE)
        except Exception as e:
            zero_shot_answer = f"Error en LLM Simple: {e}"
        zero_time = time.time() - start
        st.write(zero_shot_answer)
        st.caption(f"Tiempo: {zero_time:.2f} s")
        st.caption(f"temperature={DEFAULT_TEMPERATURE}")

    with col2:
        st.subheader("RAG Estándar")
        st.caption("RAG con parámetros default")
        start = time.time()
        try:
            rag_default_answer = run_rag(
                question,
                default_retrieved_docs,
                selected_model,
                DEFAULT_TEMPERATURE
            )
        except Exception as e:
            rag_default_answer = f"Error en RAG Estándar: {e}"
        rag_default_time = time.time() - start
        st.write(rag_default_answer)
        st.caption(f"Tiempo: {rag_default_time:.2f} s")
        st.caption(
            f"chunk_size={DEFAULT_CHUNK_SIZE} | top_k={DEFAULT_TOP_K} | temperature={DEFAULT_TEMPERATURE}"
        )

    with col3:
        st.subheader("RAG Optimizado")
        st.caption("RAG con el ajuste del Sidebar")
        start = time.time()
        try:
            rag_optimized_answer = run_rag(
                question,
                optimized_retrieved_docs,
                selected_model,
                temperature
            )
        except Exception as e:
            rag_optimized_answer = f"Error en RAG Optimizado: {e}"
        rag_optimized_time = time.time() - start
        st.write(rag_optimized_answer)
        st.caption(f"Tiempo: {rag_optimized_time:.2f} s")
        st.caption(
            f"chunk_size={chunk_size} | top_k={top_k} | temperature={temperature}"
        )

    st.markdown("---")
    st.subheader("Diagnóstico del pipeline")

    st.write(f"**Método de ingesta/OCR:** {ingestion_method}")
    st.write(f"**Modelo de generación:** {selected_model}")
    st.write(
        f"**Parámetros RAG estándar:** chunk_size={DEFAULT_CHUNK_SIZE}, "
        f"top_k={DEFAULT_TOP_K}, temperature={DEFAULT_TEMPERATURE}"
    )
    st.write(
        f"**Parámetros RAG optimizado:** chunk_size={chunk_size}, "
        f"top_k={top_k}, temperature={temperature}"
    )
    st.write(f"**Cantidad de chunks RAG estándar:** {len(default_docs)}")
    st.write(f"**Cantidad de chunks RAG optimizado:** {len(optimized_docs)}")

    with st.expander("Texto extraído"):
        st.text(raw_text[:12000])

    with st.expander("Chunks recuperados para RAG estándar"):
        for i, doc in enumerate(default_retrieved_docs, start=1):
            st.markdown(f"**Fragmento {i}**")
            st.write(doc.page_content)
            st.markdown("---")

    with st.expander("Chunks recuperados para RAG optimizado"):
        for i, doc in enumerate(optimized_retrieved_docs, start=1):
            st.markdown(f"**Fragmento {i}**")
            st.write(doc.page_content)
            st.markdown("---")
