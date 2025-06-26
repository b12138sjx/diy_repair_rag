import os
import json
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.docstore.document import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import gradio as gr
from main import get_completion  # 导入通义千问调用函数

# 配置参数
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
persist_directory = "stores/pet_cosine"

# 初始化嵌入模型（通用）
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


# 文件处理函数（来自invest.py）
def load_txt_as_document_list(file_path):
    """加载TXT文件并返回Document对象列表"""
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # 去除每行的换行符和空白
    lines = [line.strip() for line in lines if line.strip()]
    # 转换为Document对象
    documents = [Document(page_content=line) for line in lines]
    return documents


def load_json_contents(file_path):
    """加载JSON文件并返回Document对象列表"""
    documents = []
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        # 假设JSON数据是一个列表，遍历每个元素
        for item in data:
            # 提取每个元素的'content'部分
            content = item.get('content', '')
            documents.append(Document(page_content=content))
    return documents


# 向量库构建函数（带进度显示）
def build_vector_store(file, progress=gr.Progress()):
    if file is None:
        return "请上传文件"

    file_path = file.name
    file_ext = os.path.splitext(file_path)[1].lower()

    # 显示进度
    progress(0, desc="开始处理文件...")
    time.sleep(0.5)  # 模拟初始处理时间

    # 根据文件类型加载文档
    try:
        if file_ext == '.txt':
            progress(0.2, desc="正在加载TXT文件...")
            documents = load_txt_as_document_list(file_path)
        elif file_ext == '.json':
            progress(0.2, desc="正在加载JSON文件...")
            documents = load_json_contents(file_path)
        else:
            return "不支持的文件类型，仅支持 .txt 和 .json 文件"
    except Exception as e:
        return f"文件加载错误: {str(e)}"

    progress(0.4, desc="正在分割文本...")
    # 文本分割器
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    progress(0.6, desc="检查向量存储...")
    # 检查向量存储是否已存在
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        progress(0.7, desc="加载现有向量存储...")
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        progress(0.8, desc="添加新文档到向量存储...")
        vector_store.add_documents(texts)
    else:
        progress(0.7, desc="创建新的向量存储...")
        vector_store = Chroma.from_documents(
            texts,
            embeddings,
            collection_metadata={"hnsw:space": "cosine"},
            persist_directory=persist_directory
        )

    # 持久化保存更改
    progress(0.9, desc="保存向量存储...")
    vector_store.persist()

    progress(1.0, desc="完成")
    return "向量库构建完成！"


# 定义提示模板（用于问答）
prompt_template = """Use the following pieces of information to answer the user's question.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
得到答案之后，转化为中文输出
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])


# 创建通义千问LLM包装器
class TongyiQianwenLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "tongyiqianwen"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = get_completion(prompt)
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name": "tongyiqianwen"}


# 初始化LLM
llm = TongyiQianwenLLM()
print("LLM Initialized...")


# 问答函数
def get_response(input):
    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        return "请先构建向量库！"

    # 加载向量数据库
    load_vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = load_vector_store.as_retriever(search_kwargs={"k": 3})

    # 创建检索QA链
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True
    )

    response = qa(input)
    return response['result']


# 示例提示
sample_prompts = ["什么是灰狗的最快速度?", "为什么不能给狗喂巧克力?", "哪些因素可能导致狗狗感到害怕?"]

# 创建Gradio界面
with gr.Blocks(title="家居维修助手") as demo:
    gr.Markdown("# 家居维修助手 - RAG 系统")

    with gr.Tabs():
        # 第一个选项卡：构建向量库
        with gr.Tab("构建知识库"):
            gr.Markdown("### 上传文件构建向量知识库")
            file_input = gr.File(label="选择文件 (.txt 或 .json)", file_types=[".txt", ".json"])
            build_button = gr.Button("构建向量库", variant="primary")
            status_output = gr.Textbox(label="处理状态", lines=3)

            build_button.click(
                fn=build_vector_store,
                inputs=file_input,
                outputs=status_output,
                api_name="build_vector_store"
            )

        # 第二个选项卡：问答界面
        with gr.Tab("智能问答"):
            gr.Markdown("### 基于知识库的智能问答")
            question_input = gr.Textbox(
                label="问题",
                show_label=True,
                max_lines=3,
                placeholder="请输入您的问题..."
            )
            answer_output = gr.Textbox(label="回答", lines=10)
            ask_button = gr.Button("提问", variant="primary")

            gr.Examples(
                examples=sample_prompts,
                inputs=question_input,
                outputs=answer_output,
                fn=get_response,
                cache_examples=False
            )

            ask_button.click(
                fn=get_response,
                inputs=question_input,
                outputs=answer_output,
                api_name="get_answer"
            )

# 启动界面
if __name__ == "__main__":
    demo.launch()