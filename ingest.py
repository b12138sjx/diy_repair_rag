import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.docstore.document import Document
# txt读取
def load_txt_as_document_list(file_path):
    """
    Load a .txt file and return its content as a list of Document objects.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # Strip newline characters and whitespace from each line
    lines = [line.strip() for line in lines if line.strip()]
    # Convert each line to a Document object
    documents = [Document(page_content=line) for line in lines]
    return documents
# json读取
def load_json_contents(file_path):
    """
    Load a JSON file and return its contents as a list of Document objects.
    """
    documents = []
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

        # 假设 JSON 数据是一个列表，遍历每个元素
        for item in data:
            # 提取每个元素的 'content' 部分
            content = item.get('content', '')  # 使用 .get() 方法避免 KeyError
            documents.append(Document(page_content=content))  # 创建 Document 对象

    return documents
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print("1")
# Load  file
# documents=load_txt_as_document_list("pyl.txt")
documents = load_json_contents("ypy.json")
print("2")
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150,separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""])
texts = text_splitter.split_documents(documents)
print("3")
vector_store = Chroma.from_documents(texts, embeddings, collection_metadata={"hnsw:space": "cosine"}, persist_directory="stores/pet_cosine")
print("Vector Store Created.......")