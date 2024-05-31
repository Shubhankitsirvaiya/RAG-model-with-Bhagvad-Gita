from langchain_community.document_loaders import PyPDFLoader
import os
from langchain.schema import prompt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough
import gradio as gr

from langchain_openai import OpenAIEmbeddings,ChatOpenAI

os.environ["OPENAI_API_KEY"] = "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"

loader = PyPDFLoader("bhagavad-gita-in-english-source-file.pdf")
docs = loader.load()


# #chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# #Vector Store
# #APIM call 
embeddings = OpenAIEmbeddings(    
            model="text-embedding-ada-002",
            deployment="text-embedding-ada-002",
            api_key="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",
            chunk_size=16,
            
        )


#Prompt
prompt = """You are an AI trained on the Bhagavad Gita, a sacred Hindu scripture. You provide readings from the text and offer wisdom and guidance based on its teachings. 
Your responses should reflect the spiritual and philosophical nature of the Bhagavad Gita, offering deep insights into life’s questions. 
When asked a question, reference specific verses when appropriate and explain their relevance to the query.
Given below is the context and question of the user.
context = {context}
question = {question}
"""

prompt = ChatPromptTemplate.from_template(prompt)

# create retriever
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

#calling LLM
llm  = ChatOpenAI( 
    
    
    model="gpt-3.5-turbo",
    api_key="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",
    
    
    )


#create RAG chain


rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm




def demo(name):
    return rag_chain.invoke(name).content

demo = gr.Interface(fn=demo, inputs="textbox", outputs="textbox")
    
demo.launch(share=True)




