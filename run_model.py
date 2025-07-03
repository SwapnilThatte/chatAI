from model import get_gemma
from rag_utils import generate_context


def generate_response(history=[], temperature: float=0.0, top_k=None, top_p=None):
    
    gemma_model = get_gemma(temperature=temperature, top_k=top_k, top_p=top_p)
   
    response = gemma_model.invoke(history).content

    return response

def generate_RAG_response(query: str, file_path, history=[]):
    gemma_model = get_gemma()
    query, context = generate_context(query, file_path)

    history[-1] = {"role" : "user", "content" : f"INSTRUCTION: Answer the query with given context in mind.\nQUERY: {query}\n\nCONTEXT : {context}"}
    
    response = gemma_model.invoke(history).content

    return response