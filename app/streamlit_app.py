import streamlit as st

st.set_page_config(page_title="Internal AI Sales Assistant")
st.title("Internal AI Sales Assistant")

question = st.text_input("Ask a sales or application question", placeholder="Type your question here...")

if st.button("Submit"):
    if question.strip():
        st.write(question)
    else:
        st.info("Please enter a question.")
