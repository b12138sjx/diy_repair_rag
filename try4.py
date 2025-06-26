import os
import json
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
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
embeddings = HuggingFaceEmbeddings(
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
# 初始化记忆列表
history = []
# 示例提示
sample_prompts = ["红木维修守则?", "华为手机?", "哪些因素可能导致断电?"]
# 问答函数
def get_response(input):
    global history
    # 构建历史记录字符串
    history_str = "\n".join([f"用户: {h[0]}" for h in history])
    his_prompt = str(history_str)
    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        return "请先构建向量库！"

    # 加载向量数据库
    load_vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = load_vector_store.as_retriever(search_kwargs={"k": 3})

    # 创建检索QA链
    chain_type_kwargs = {"prompt": prompt+"如果用户询问的是prompt里有相关信息的问题\
    那么不需要主动结合以下历史信息。\
    以下是我（用户）告诉你的历史信息，其中“我”均指代用户本人。\
    请在处理信息时，将“我”转换为“您”（用户），并确保不将“我”误判为AI自身。\
    例如若历史信息为“我是奶龙”，请理解为“您是奶龙”（用户是奶龙），而非AI自称。所有以第一人称表述的内容，\
    均为用户的陈述，而非AI的身份信息。注意，用户没有明确提到历史信息需要的部分则无需主动提及\
    请严格遵循上述规则处理以下信息：  "+his_prompt}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True
    )

    # 调用QA链，将 'question' 替换为 'query'
    response = qa({"query": input})
    # 更新记忆列表
    history.append((input, response['result']))
    if len(history) > 10:
        history.pop(0)
    # 生成相关问题
    related_questions_prompt = f"根据问题 '{input}' 和结果 '{response['result']}' \
    生成三个推荐用户询问的相关问题,以换行符分隔"
    related_questions_response = get_completion(related_questions_prompt)
    related_questions = related_questions_response.strip().split('\n')[:3]

    # 将相关问题添加到 sample_prompts 列表
    global sample_prompts
    sample_prompts=related_questions
    print(sample_prompts)
    return response['result']




# 自定义CSS样式
custom_css = """
/* 全局样式 */
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif !important;
    background: linear-gradient(135deg, #f7f3e9 0%, #ede0d3 100%) !important;
}

/* 主标题样式 */
.main-title {
    text-align: center;
    background: linear-gradient(135deg, #8b7355 0%, #6d5a47 100%);
    color: #fff5f0 !important;
    padding: 2rem;
    margin: 0 0 2rem 0;
    border-radius: 15px;
    box-shadow: 0 8px 25px rgba(107, 90, 71, 0.15);
    font-size: 2.5rem !important;
    font-weight: 700 !important;
}

/* 标签页样式 */
.tab-nav {
    background: #faf8f5;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(139, 115, 85, 0.1);
    margin-bottom: 1.5rem;
    border: 1px solid #e8ddd4 !important;
}

.tab-nav button {
    background: transparent !important;
    border: none !important;
    padding: 1rem 2rem !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #6d5a47 !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}

.tab-nav button:hover {
    background: #f0ebe4 !important;
    color: #8b7355 !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #8b7355 0%, #6d5a47 100%) !important;
    color: #fff5f0 !important;
    box-shadow: 0 3px 12px rgba(139, 115, 85, 0.3) !important;
}

/* 卡片容器样式 */
.card-container {
    background: #faf8f5;
    border-radius: 15px;
    padding: 2rem;
    box-shadow: 0 6px 20px rgba(139, 115, 85, 0.08);
    margin-bottom: 1.5rem;
    border: 1px solid #e8ddd4;
}

/* 按钮样式 */
.btn-primary {
    background: linear-gradient(135deg, #8b7355 0%, #6d5a47 100%) !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 12px 30px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    color: #fff5f0 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(139, 115, 85, 0.25) !important;
}

.btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(139, 115, 85, 0.35) !important;
    background: linear-gradient(135deg, #9d8466 0%, #7a6552 100%) !important;
}

/* 输入框样式 */
.input-container {
    border-radius: 12px !important;
    border: 2px solid #e8ddd4 !important;
    background: #faf8f5 !important;
    transition: all 0.3s ease !important;
}

.input-container:focus-within {
    border-color: #8b7355 !important;
    box-shadow: 0 0 0 3px rgba(139, 115, 85, 0.1) !important;
    background: #ffffff !important;
}

/* 输出框样式 */
.output-container {
    background: #f5f1eb;
    border-radius: 12px;
    border: 1px solid #e8ddd4;
    padding: 1rem;
    color: #4a3f36;
}

/* 文件上传区域样式 */
.file-upload {
    border: 2px dashed #c4b5a6 !important;
    border-radius: 12px !important;
    padding: 2rem !important;
    text-align: center !important;
    background: #f5f1eb !important;
    transition: all 0.3s ease !important;
}

.file-upload:hover {
    border-color: #8b7355 !important;
    background: #f0ebe4 !important;
}

/* 示例按钮样式 */
.examples-container button {
    background: #f0ebe4 !important;
    border: 1px solid #e8ddd4 !important;
    border-radius: 20px !important;
    padding: 8px 16px !important;
    color: #6d5a47 !important;
    font-size: 0.9rem !important;
    transition: all 0.3s ease !important;
    margin: 4px !important;
}

.examples-container button:hover {
    background: #8b7355 !important;
    color: #fff5f0 !important;
    border-color: #8b7355 !important;
    transform: translateY(-1px) !important;
}

/* 状态消息样式 */
.status-success {
    background: linear-gradient(135deg, #7d8471 0%, #6b7059 100%);
    color: #fff5f0;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}

.status-error {
    background: linear-gradient(135deg, #a67c52 0%, #8b6f47 100%);
    color: #fff5f0;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}

/* 响应式设计 */
@media (max-width: 768px) {
    .main-title {
        font-size: 2rem !important;
        padding: 1.5rem !important;
    }

    .card-container {
        padding: 1.5rem !important;
        margin: 1rem !important;
    }

    .tab-nav button {
        padding: 0.8rem 1.5rem !important;
        font-size: 1rem !important;
    }
}

/* 加载动画 */
.loading {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid #e8ddd4;
    border-top: 3px solid #8b7355;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* 自定义滚动条 */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #f5f1eb;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: #c4b5a6;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #8b7355;
}
"""

# 创建Gradio界面
with gr.Blocks(title="家居维修助手 - RAG系统", css=custom_css, theme=gr.themes.Soft()) as demo:
    # 主标题
    gr.HTML("""
        <div class="main-title">
            🏠 家居维修助手
            <div style="font-size: 1.2rem; font-weight: 400; margin-top: 0.5rem; opacity: 0.9;">
                基于AI的智能问答与知识库管理系统
            </div>
        </div>
    """)

    with gr.Tabs(elem_classes=["tab-nav"]):
        # 知识库管理页面
        with gr.Tab("📚 知识库管理", elem_classes=["card-container"]):
            gr.HTML("""
                <div style="text-align: center; margin-bottom: 2rem;">
                    <h3 style="color: #4a3f36; margin-bottom: 0.5rem;">📁 上传文档构建知识库</h3>
                    <p style="color: #6d5a47; font-size: 1rem;">支持 TXT 和 JSON 格式文件，系统将自动处理并构建向量索引</p>
                </div>
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(
                        label="📄 选择文件",
                        file_types=[".txt", ".json"],
                        elem_classes=["file-upload"]
                    )

                    build_button = gr.Button(
                        "🔨 构建向量库",
                        variant="primary",
                        elem_classes=["btn-primary"],
                        size="lg"
                    )

                with gr.Column(scale=1):
                    gr.HTML("""
                        <div style="background: #f0ebe4; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #8b7355;">
                            <h4 style="color: #4a3f36; margin-bottom: 1rem;">💡 使用提示</h4>
                            <ul style="color: #6d5a47; line-height: 1.6;">
                                <li>支持 .txt 和 .json 文件格式</li>
                                <li>文件内容将被智能分割和索引</li>
                                <li>可多次上传文件扩展知识库</li>
                                <li>处理完成后即可开始问答</li>
                            </ul>
                        </div>
                    """)

            status_output = gr.Textbox(
                label="📊 处理状态",
                lines=4,
                elem_classes=["output-container"],
                interactive=False
            )

            build_button.click(
                fn=build_vector_store,
                inputs=file_input,
                outputs=status_output,
                api_name="build_vector_store"
            )

        # 智能问答页面
        with gr.Tab("💬 智能问答", elem_classes=["card-container"]):
            gr.HTML("""
                <div style="text-align: center; margin-bottom: 2rem;">
                    <h3 style="color: #4a3f36; margin-bottom: 0.5rem;">🤖 基于知识库的智能问答</h3>
                    <p style="color: #6d5a47; font-size: 1rem;">向AI提问，获得基于您知识库内容的准确回答</p>
                </div>
            """)

            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Group():
                        question_input = gr.Textbox(
                            label="💭 您的问题",
                            show_label=True,
                            max_lines=3,
                            placeholder="请输入您想了解的问题...",
                            elem_classes=["input-container"],
                            container=True
                        )
                        ask_button = gr.Button(
                            "🚀 获取答案",
                            variant="primary",
                            elem_classes=["btn-primary"],
                            size="lg"
                        )
                        answer_output = gr.Textbox(
                            label="🎯 AI回答",
                            lines=12,
                            elem_classes=["output-container"],
                            interactive=False
                        )
                with gr.Column(scale=1):
                    gr.HTML("""
                        <div style="margin: 1.5rem 0;">
                            <h4 style="color: #6d5a47; margin-bottom: 1rem;">🎯 示例问题：</h4>
                        </div>
                    """)
                    gr.Examples(
                        examples=sample_prompts,
                        inputs=question_input,
                        label="点击下方示例快速开始："
                    )
                    gr.HTML("""
                        <div style="background: #ede8e0; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #7d8471; margin-top: 1.5rem;">
                            <h4 style="color: #4a3f36; margin-bottom: 1rem;">✨ 问答技巧</h4>
                            <ul style="color: #6d5a47; line-height: 1.6;">
                                <li>问题描述要具体清晰</li>
                                <li>可以询问操作步骤</li>
                                <li>支持故障诊断问题</li>
                                <li>可以要求详细解释</li>
                            </ul>
                        </div>
                    """)

            ask_button.click(
                fn=get_response,
                inputs=question_input,
                outputs=answer_output,
                api_name="get_answer"
            )

    # 页脚信息
    gr.HTML("""
        <div style="text-align: center; margin-top: 3rem; padding: 2rem; color: #6d5a47; font-size: 0.9rem;">
            <p>🏠 家居维修助手 | 让AI成为您的维修专家</p>
            <p style="margin-top: 0.5rem; opacity: 0.8;">基于RAG技术，提供准确可靠的维修指导</p>
        </div>
    """)

# 启动界面
if __name__ == "__main__":
    import socket


    def find_free_port():
        """找到一个可用的端口"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port


    def try_launch_gradio():
        """尝试启动Gradio应用"""
        ports_to_try = [7860, 7861, 7862, 7863, 7864]

        for port in ports_to_try:
            try:
                print(f"尝试在端口 {port} 启动应用...")
                demo.launch(
                    server_name="127.0.0.1",
                    server_port=port,
                    share=False,
                    inbrowser=True,
                    prevent_thread_lock=False,
                    quiet=True
                )
                print(f"✅ 成功在端口 {port} 启动应用！")
                return
            except Exception as e:
                print(f"❌ 端口 {port} 启动失败: {str(e)}")
                demo.close()  # 确保清理资源
                continue

        # 如果所有预设端口都失败，尝试随机端口
        try:
            free_port = find_free_port()
            print(f"尝试使用随机端口 {free_port}...")
            demo.launch(
                server_name="127.0.0.1",
                server_port=free_port,
                share=False,
                inbrowser=True,
                prevent_thread_lock=False,
                quiet=True
            )
            print(f"✅ 成功在随机端口 {free_port} 启动应用！")
        except Exception as e:
            print(f"❌ 随机端口也失败，尝试共享链接...")
            try:
                demo.launch(
                    share=True,
                    inbrowser=True,
                    prevent_thread_lock=False,
                    quiet=False
                )
                print("✅ 成功创建共享链接！")
            except Exception as final_e:
                print(f"❌ 所有启动方式都失败了: {str(final_e)}")
                print("请检查网络设置或防火墙配置")


    try_launch_gradio()