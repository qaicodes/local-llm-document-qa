from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings


data_directory = 'HP_Books'

# Load files from directory

loader = DirectoryLoader(data_directory,
                         glob='*.pdf',
                         loader_cls=PyPDFLoader)
document = loader.load()

# Split text into chunks of 500 characters with 50 overlapping characters

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

text  = text_splitter.split_documents(document)

# Load embeddings model

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                   model_kwargs={'device': 'cpu'})

# Build vector database

vectorstore = FAISS.from_documents(text, embeddings)
vectorstore.save_local(f'vectorstore/{data_directory}')


