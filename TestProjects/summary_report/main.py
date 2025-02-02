import os
import streamlit as st
import pickle
import time
import numpy as np
from langchain import OpenAI
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
import faiss

from dotenv import load_dotenv
load_dotenv() # get environment variables from .env

st.title('News Research')
st.sidebar.title('News Articles URLs')

urls = []
for i in range(3):
    url = st.sidebar.text_input(f'URL {i+1}')
    urls.append(url)

process_url_clicked = st.sidebar.button('Process URLs')

file_path = 'faiss_store_sentence_transformer.pkl'

# Progress bar on the UI
main_placeholder = st.empty()

vector_index = faiss.IndexFlatL2(768)
encoder = SentenceTransformer('all-mpnet-base-v2')
if process_url_clicked:
   loader =  UnstructuredURLLoader(urls=urls)
   main_placeholder.text('Data Loading Started...')
   data = loader.load()

   # Split data into chunks
   text_splitter = RecursiveCharacterTextSplitter(
       separators=['\n\n', '\n', '.', ','],
       chunk_size=4000,
       chunk_overlap=200
   )
   main_placeholder.text('Text Splitter Started...')

   chunks = text_splitter.split_documents(data)
   chunk_text = [chunk.page_content for chunk in chunks]

   #Create Embeddings and stores in vector db
   vectors = encoder.encode(chunk_text)
   dims = vectors.shape[1]

   vector_index = faiss.IndexFlatL2(dims)
   vector_index.add(vectors)
   main_placeholder.text('Embedding Vector Started Building...')
   time.sleep(2)

   # Save the FAISS index to pickle file
   with open(file_path, 'wb') as f:
       pickle.dump(vector_index, f)

llm = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key='os.environ["GITHUB_TOKEN"]',
    model="gpt-4o",
    temperature=0.6,
    max_tokens=4096,
    top_p=1
)


template = """
Context:
{context}

Question:
{search_query}

Answer:
"""

query = main_placeholder.text_input('Question: ')
if query:
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            vectors_store = pickle.load(f)
            vec = encoder.encode(query)
            svec = np.array(vec).reshape(1, -1)
            distances, I = vector_index.search(svec, k=2)
            doc_indices = I[I<len(chunks)].tolist()
            context = '\n'.join([chunk_text[i] for i in doc_indices])
            prompt_template = PromptTemplate(input_variables=["context", "query"], template=template)
            chain = LLMChain(llm=llm, prompt=prompt_template)
            try:
                answer = chain.run({"context": context, "query": query})
                st.header('Answer')
                st.subheader(answer)
            except Exception as e:
                print(f"An error occurred: {e}")