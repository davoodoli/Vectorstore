from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()


#load the data
loader = TextLoader('note.txt')
text = loader.load()

#embedding layer to vectorize the text
embedding = OpenAIEmbeddings()

#use splitter to to split the text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=50)
documents = text_splitter.split_documents(text)

#use FAISS to store the vectorized data into the vectorstore
db = FAISS.from_documents(documents,embedding)

#do a smiliarity search in the vectorstore
reslult = db.similarity_search(query="Davood")

#use db as a retriever
retriever = db.as_retriever()
result = retriever.invoke("AI with business")

#do a similarity search with getting the similarity scores
scored_result = db.similarity_search_with_score("AI with business")
print(scored_result)


#db.similarity_search_by_vector(embedding_vector)

#save a FAISS vectore store in local storage and loding from there
db.save_local("./vectorstore")
new_df = FAISS.load_local("./vectorstore",embedding,allow_dangerous_deserialization=True)

#creating a Chroma db and doing a similarity search on it
chroma_db = Chroma.from_documents(documents,embedding)
chrom_result = chroma_db.similarity_search("social responsibility")
print(chrom_result[0].page_content)


#saving to the disk
chroma_db = Chroma.from_documents(documents,embedding,persist_directory="./chroma_db")


#load from disk
db2 = Chroma(persist_directory="./chroma_db",embedding_function=embedding)