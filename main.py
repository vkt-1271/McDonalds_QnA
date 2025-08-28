import streamlit as st
from langchain_helper import create_vector_db, get_qna_chain

st.title("McDonald's USA FAQs")

question = st.text_input("Question: ")

if question:
    chain = get_qna_chain()
    response = chain(question)

    st.header("Answer: ")
    st.write(response["result"])