# ---------- Imports ----------
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
import re
import json


# ---------- PDF Loader ----------
def load_pdf(path):
    """
    Extract text from PDF safely
    """
    text = ""

    try:
        reader = PdfReader(path)

        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted

    except Exception as e:
        print(f"Error reading PDF: {e}")

    return text


# ---------- Resume Parsing ----------
def extract_resume_details(text):
    """
    Extract name, email and skills from resume text
    """

    # Email extraction
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    emails = re.findall(email_pattern, text)

    # Name extraction (assume first line)
    name = text.split("\n")[0].strip()

    # Skills database
    skills_database = [
        "python",
        "java",
        "c++",
        "machine learning",
        "data analysis",
        "sql",
        "deep learning",
        "nlp",
        "pandas",
        "numpy",
        "tensorflow",
        "pytorch"
    ]

    found_skills = []

    for skill in skills_database:
        if skill.lower() in text.lower():
            found_skills.append(skill)

    resume_data = {
        "name": name,
        "email": emails[0] if emails else "Not found",
        "skills": found_skills
    }

    return resume_data


# ---------- Job Match Score ----------
def calculate_match_score(candidate_skills):

    required_skills = [
        "python",
        "machine learning",
        "data analysis",
        "sql",
        "pandas"
    ]

    matched = 0

    for skill in required_skills:
        if skill in candidate_skills:
            matched += 1

    score = (matched / len(required_skills)) * 100

    return round(score, 2)


# ---------- Save JSON ----------
def save_to_json(data):
    """
    Save analysis report
    """

    with open("resume_report.json", "w") as file:
        json.dump(data, file, indent=4)

    print("Resume report saved to resume_report.json")


# ---------- Text Chunking ----------
def chunk_text(text):
    """
    Split text into smaller chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    return splitter.split_text(text)


# ---------- Store Embeddings ----------
def store_embeddings(chunks):
    """
    Generate embeddings and store in ChromaDB
    """

    if not chunks:
        print("No chunks to embed. Check your PDF content.")
        return

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("resume_kb")

    embeddings = [embedding_model.embed_query(chunk) for chunk in chunks]

    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings
    )

    print("Embeddings stored successfully!")


# ---------- Main ----------
if __name__ == "__main__":

    pdf_path = "sample_resume.pdf"

    print("Loading PDF...")
    text = load_pdf(pdf_path)

    print("Text length:", len(text))

    if not text.strip():
        print("No text extracted from PDF.")
        exit()

    # Extract resume details
    print("Extracting resume details...")
    resume_data = extract_resume_details(text)

    # Calculate match score
    score = calculate_match_score(resume_data["skills"])
    resume_data["match_score"] = score

    # Recommendation logic
    if score >= 80:
        recommendation = "Strong Candidate"
    elif score >= 50:
        recommendation = "Consider for Interview"
    else:
        recommendation = "Not Recommended"

    resume_data["recommendation"] = recommendation

    print("\nResume Analysis")
    print("----------------")
    print("Name:", resume_data["name"])
    print("Email:", resume_data["email"])
    print("Skills Found:", resume_data["skills"])
    print("Match Score:", score)
    print("Recommendation:", recommendation)

    # Save JSON report
    save_to_json(resume_data)

    # Continue embedding pipeline
    print("\nChunking text...")
    chunks = chunk_text(text)

    print("Storing embeddings...")
    store_embeddings(chunks)