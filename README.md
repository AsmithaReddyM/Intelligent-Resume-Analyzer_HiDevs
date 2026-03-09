# Intelligent Resume Analyzer

An AI-powered resume processing pipeline that extracts text from resumes, splits the content into meaningful chunks, generates semantic embeddings using Hugging Face models, and stores them in ChromaDB for efficient semantic search.

This project demonstrates the core pipeline used in modern Retrieval-Augmented Generation (RAG) systems.

---

## Features

* PDF resume text extraction
* Smart text chunking
* Embedding generation using Hugging Face
* Vector database storage with ChromaDB
* Fast semantic retrieval-ready pipeline
* Modular and beginner-friendly code

---

## Tech Stack

* Python
* PyPDF2
* LangChain Text Splitters
* HuggingFace Embeddings
* Sentence Transformers
* ChromaDB

---

## Project Structure

```
Intelligent-Resume-Analyzer/
│
├── main.py                # Main pipeline script
├── requirements.txt       # Dependencies
├── sample_resume.pdf      # Sample input resume
├── chroma_db/             # Vector database (auto-created)
└── README.md              # Project documentation
```

---

## How It Works

The pipeline follows these steps:

1. Load Resume

   * Reads PDF using PyPDF2

2. Text Extraction

   * Extracts raw text from all pages

3. Chunking

   * Splits text into overlapping chunks

4. Embedding Generation

   * Uses `sentence-transformers/all-MiniLM-L6-v2`

5. Vector Storage

   * Stores embeddings in ChromaDB

This forms the foundation of an industry-level RAG system.

---

## How to Run

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Add your resume

Place your resume in the project root and name it:

```
sample_resume.pdf
```

Important:

* The file must be a text-based PDF
* Scanned image PDFs will not work

---

### Step 3: Run the pipeline

```bash
python main.py
```

---

## Expected Output

You should see output similar to:

```
Loading PDF...
Text length: XXX
Chunking text...
Storing embeddings...
Embeddings stored successfully!
```

After running:

* The `chroma_db/` folder will be created
* Resume content will be stored as embeddings
* The system becomes ready for semantic search

---

## Sample Use Cases

This pipeline can be extended for:

* Resume screening systems
* Candidate-job matching
* HR automation tools
* Semantic resume search
* RAG-based recruitment assistants

---

## Future Improvements

* Semantic search interface
* Resume-job match scoring
* LLM-powered query system
* Streamlit web UI
* Multi-resume support
* Cloud deployment

---

## Author

Asmitha Reddy M

---

## Notes

This project is built for learning and demonstration of:

* Document processing
* Embeddings pipeline
* Vector databases
* RAG system foundations
