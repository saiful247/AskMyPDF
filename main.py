import os
import streamlit as st
import time
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import requests
from google.cloud import vision
from PIL import Image
from io import BytesIO
from google.oauth2 import service_account
import json

# Load environment variables
load_dotenv()

# Load Google Vision API Credentials from Streamlit Secrets
credentials_dict = json.loads(st.secrets["GOOGLE_VISION_CREDENTIALS"])
credentials = service_account.Credentials.from_service_account_info(
    credentials_dict)

client = vision.ImageAnnotatorClient(credentials=credentials)


def extract_text_from_image(image_source):
    if image_source.startswith("http"):
        response = requests.get(image_source)
        img = BytesIO(response.content)
        image = vision.Image(content=img.getvalue())
    elif os.path.exists(image_source):
        with open(image_source, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
    else:
        raise ValueError("Invalid image source!")

    response = client.text_detection(image=image)
    texts = response.text_annotations

    return texts[0].description.strip() if texts else None


st.title("📚 Study Support System")
st.sidebar.title("📂 Upload PDF Files")

# Allow multiple PDF uploads
pdf_files = st.sidebar.file_uploader("Upload up to 2 PDFs", type=[
                                     "pdf"], accept_multiple_files=True)
process_pdf = st.sidebar.button("Process PDFs")

# Directory to store vector index
file_path = "models/qa_with_pdf"
os.makedirs(file_path, exist_ok=True)

# Initialize session state for PDF processing
if "pdf_processed" not in st.session_state:
    st.session_state["pdf_processed"] = False

main_placeholder = st.empty()

# Process uploaded PDFs
if process_pdf and pdf_files:
    doc_chunks = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join("uploaded_docs", pdf_file.name)
        os.makedirs("uploaded_docs", exist_ok=True)

        with open(pdf_path, "wb") as f:
            f.write(pdf_file.read())

        loader = PyPDFLoader(pdf_path)
        main_placeholder.text(f"📄 Loading data from {pdf_file.name}...")

        data = loader.load()
        doc_splitter = RecursiveCharacterTextSplitter(
            chunk_size=520, chunk_overlap=50)

        main_placeholder.text(f"✂️ Splitting data from {pdf_file.name}...")
        doc_chunks.extend(doc_splitter.split_documents(data))

    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorindex_gemini = FAISS.from_documents(doc_chunks, embedding)

    main_placeholder.text("🔄 Embedding Vector Started Building...")
    time.sleep(2)

    vectorindex_gemini.save_local(file_path)

    st.session_state["pdf_processed"] = True
    st.success("✅ PDFs processed successfully! You can now ask questions.")

# User Query Input or Image Upload
st.header("💡 Ask Your Question")

# Text Input Section
query = st.text_input("Enter your question here",
                      disabled=st.session_state.get("image_uploaded", False))

# Image Upload Section (Below the text input)
image_file = st.file_uploader(
    "Or upload an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

# "Ask" Button Logic
if st.button("Ask"):
    # Prevent submission if PDFs are not processed
    if not st.session_state["pdf_processed"]:
        st.error("⚠️ Please upload and process a PDF before asking a question!")
        st.stop()

    # Ensure only one input is used
    if query and image_file:
        st.warning(
            "⚠️ You can either type a question OR upload an image, not both.")
        st.stop()

    # Process Query from Text Input
    if query:
        st.session_state["image_uploaded"] = False
        final_query = query

    # Process Query from Image Upload
    elif image_file:
        st.session_state["image_uploaded"] = True
        image_path = os.path.join("uploaded_docs", image_file.name)
        with open(image_path, "wb") as f:
            f.write(image_file.read())

        extracted_text = extract_text_from_image(image_path)

        if extracted_text:
            st.success("✅ Text extracted successfully from the image!")
            final_query = extracted_text
        else:
            st.error("❌ No text found in the image. Try another image.")
            st.stop()

    else:
        st.warning("⚠️ Please enter a question or upload an image!")
        st.stop()

    # Ensure the vector index exists before querying
    if final_query and os.path.exists(file_path):
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorindex_gemini = FAISS.load_local(
            file_path, embedding, allow_dangerous_deserialization=True)

        retriever = vectorindex_gemini.as_retriever()
        qa_chain = load_qa_chain(llm, chain_type="stuff")

        chain = RetrievalQA(
            combine_documents_chain=qa_chain, retriever=retriever)
        result = chain({"query": final_query})

        st.header("🤖 Answer")
        st.subheader(result["result"])
