from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import gradio as gr
import os
from main import get_completion  # 导入通义千问调用函数

# 配置参数
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

# 初始化嵌入模型
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 加载向量数据库
load_vector_store = Chroma(persist_directory="stores/pet_cosine", embedding_function=embeddings)
retriever = load_vector_store.as_retriever(search_kwargs={"k": 3})

# 定义提示模板
prompt_template = """Use the following pieces of information to answer the user's question.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
得到答案之后，转化为中文输出
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
# print(prompt)

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

# 示例提示
sample_prompts = ["what is the fastest speed for a greyhound dog?", "Why should we not feed chocolates to the dogs?",
                  "Name two factors which might contribute to why some dogs might get scared?"]


# 响应处理函数
def get_response(input):
    query = input
    chain_type_kwargs = {"prompt": prompt}
    # 创建检索QA链
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True
    )
    response = qa(query)
    return response['result']  # 直接返回回答结果

# test PYL
ins=input()
res=get_response(ins)
print(res)

# # 界面设置 - 修复Gradio参数
# input = gr.Textbox(
#     label="Prompt",
#     show_label=False,
#     max_lines=1,
#     placeholder="Enter your prompt",
# )
#
# # 创建Gradio界面 - 移除过时参数
# iface = gr.Interface(
#     fn=get_response,
#     inputs=input,
#     outputs="text",
#     title="家居维修助手",
#     description="This is a RAG implementation based on Tongyi Qianwen LLM.",
#     examples=sample_prompts,
#     allow_flagging="never"  # 更新为新的参数名称
# )
#
# # 启动界面
# iface.launch()