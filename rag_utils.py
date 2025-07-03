from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_pdf_document(file_path):
    document_loader = PyPDFLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True
    )
    return text_processor.split_documents(raw_documents)

def find_related_documents(query, vector_database):
    # return vector_database.similarity_search(query, k=2)
    return vector_database.max_marginal_relevance_search(query, k=2, fetch_k=5, lambda_mult=0.6)


def ProcessDocuments(document_path: str) -> str:

    loaded_doc = load_pdf_document(document_path)
    chunked_doc = chunk_documents(loaded_doc)

    
    vector_database = Chroma(
        persist_directory=f"./chroma_store/{document_path.split("\\")[-1].split(".")[0]}",
        embedding_function=embedding_model
    )

    vector_database.add_documents(chunked_doc)


def generate_context(query: str, file: str):

    ProcessDocuments(file)

    
    vector_database = Chroma(
        persist_directory=f"./chroma_store/{file.split("\\")[-1].split(".")[0]}",
        embedding_function=embedding_model
    )

    relevant_docs = find_related_documents(query, vector_database)
    context_text = "\n".join([doc.page_content for doc in relevant_docs])
 
    return query, context_text

