# Imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from pymongo import MongoClient
import os

#document loader
loader=PyMuPDFLoader("your.pdf")
doc=loader.load()
splitter=RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=200)
docs=splitter.split_documents(doc)

#MongoDB vectorstore
import certifi

mongoclient = MongoClient(
    "connection url",
    tlsCAFile=certifi.where()
)
dbname="database"
collection_name="collection"
collection=mongoclient[dbname][collection_name]
db=MongoDBAtlasVectorSearch.from_documents(docs,HuggingFaceEmbeddings(),collection=collection)

os.environ["GROQ_API_KEY"] = "groq api key here"

custom_prompt_template = """Consider the following document excerpt:
{context}
Now, based on this information, answer the following question:
{question}
Provide a concise and accurate response below:
"""

prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])


llm = ChatGroq(
    model_name="mixtral-8x7b-32768",
    temperature=0.1,
    max_new_tokens=512,
)


qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
    chain_type_kwargs={"prompt": prompt},
)

query = "xxx?"
result = qa.invoke(query)
print(result['result'])
