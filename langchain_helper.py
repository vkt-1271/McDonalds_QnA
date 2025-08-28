import os
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

vector_db_file_path = 'faiss_index'

# Model Loading
llm = GoogleGenerativeAI(google_api_key=os.environ["GOOGLE_API_KEY"], model="gemini-2.5-pro")

# Embedding
instructor_embeddings = HuggingFaceEmbeddings(model_name= "hkunlp/instructor-large")

# Data Loading and Vector Database (to keep embeddings)
def create_vector_db():
    loader = CSVLoader(file_path='McDonalds_faqs.csv', source_column='prompt')
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vectordb.save_local(vector_db_file_path)


def get_qna_chain():
    vectordb = FAISS.load_local(vector_db_file_path, instructor_embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(score_threshold=0.7)

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type='stuff',
                                        retriever=retriever,
                                        input_key='query',
                                        return_source_documents=True)

    return chain

if __name__ == "__main__":
    chain = get_qna_chain()

    print(chain('do you have juice'))