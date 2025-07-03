from api_key import GOOGLE_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI



def get_gemma(temperature=0, top_k=40, top_p=0.95):

    gemma_model = ChatGoogleGenerativeAI(
            model="gemma-3-12b-it",
            temperature=temperature,
            max_tokens=2048,
            timeout=None,
            max_retries=2,
            google_api_key=GOOGLE_API_KEY,
            top_k=top_k,
            top_p=top_p 
    )

    return gemma_model