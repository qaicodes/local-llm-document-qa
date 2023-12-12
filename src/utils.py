from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from src.prompts import qa_template
from src.local_llm import llm

# Wrap prompt template in a PromptTemplate object
def set_qa_prompt():
    prompt = PromptTemplate(template=qa_template,
                            input_variables=['context', 'question']
                            )
    return prompt


def build_vectordb_retriever_qa(llm , prompt, vectordb):
    vectordbqa = RetrievalQA.from_chain_type(llm=llm, 
                                    chain_type='stuff',
                                    retriever=vectordb.as_retriever(search_kwargs={'k':2}),
                                    return_source_documents=True,
                                    chain_type_kwargs={'prompt': prompt})
    return vectordbqa


def setup_vectordbqa():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', 
                                        model_kwargs={'device': 'cpu'})
    vectorstore_db = FAISS.load_local('vectorstore/HP_Books', embeddings)
    qa_prompt = set_qa_prompt()
    vectordbqa = build_vectordb_retriever_qa(llm, qa_prompt, vectorstore_db)
    return vectordbqa        
                                        