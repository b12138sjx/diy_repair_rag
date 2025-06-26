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
    background: 
        linear-gradient(rgba(247, 243, 233, 0.85), rgba(237, 224, 211, 0.85)),
        url('https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80') !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
    border-radius: 16px !important;
    min-height: 100vh !important;
}

/* 为整个应用添加半透明背景层 */
.gradio-container::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(247, 243, 233, 0.3);
    z-index: -1;
    pointer-events: none;
}

/* 主标题样式 */
.main-title {
    text-align: center;
    background: linear-gradient(135deg, rgba(139, 115, 85, 0.95) 0%, rgba(109, 90, 71, 0.95) 100%);
    color: #fff5f0 !important;
    padding: 2rem;
    margin: 0 0 2rem 0;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(107, 90, 71, 0.25);
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* 标签页样式 */
.tab-nav {
    background: rgba(250, 248, 245, 0.95);
    border-radius: 16px;
    box-shadow: 0 4px 15px rgba(139, 115, 85, 0.2);
    margin-bottom: 1.5rem;
    border: 1px solid rgba(232, 221, 212, 0.8) !important;
    backdrop-filter: blur(15px);
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
    background: rgba(240, 235, 228, 0.8) !important;
    color: #8b7355 !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, rgba(139, 115, 85, 0.95) 0%, rgba(109, 90, 71, 0.95) 100%) !important;
    color: #fff5f0 !important;
    box-shadow: 0 3px 12px rgba(139, 115, 85, 0.4) !important;
}

/* 卡片容器样式 */
.card-container {
    background: rgba(250, 248, 245, 0.95);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 8px 30px rgba(139, 115, 85, 0.18);
    margin-bottom: 1.5rem;
    border: 1px solid rgba(232, 221, 212, 0.6);
    backdrop-filter: blur(15px);
}

/* 按钮样式 */
.btn-primary {
    background: linear-gradient(135deg, rgba(139, 115, 85, 0.95) 0%, rgba(109, 90, 71, 0.95) 100%) !important;
    border: none !important;
    border-radius: 30px !important;
    padding: 14px 32px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    color: #fff5f0 !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 12px rgba(139, 115, 85, 0.35) !important;
    width: 100% !important;
    margin: 1rem 0 !important;
    backdrop-filter: blur(10px);
}

.btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(139, 115, 85, 0.45) !important;
    background: linear-gradient(135deg, rgba(157, 132, 102, 0.95) 0%, rgba(122, 101, 82, 0.95) 100%) !important;
}

.btn-primary:active {
    transform: translateY(0px) !important;
    box-shadow: 0 4px 12px rgba(139, 115, 85, 0.35) !important;
}

.btn-primary:disabled {
    background: rgba(196, 181, 166, 0.8) !important;
    color: #8b7355 !important;
    cursor: not-allowed !important;
    transform: none !important;
    box-shadow: 0 2px 6px rgba(139, 115, 85, 0.2) !important;
}

/* 输入框样式 */
.input-container {
    # border-radius: 16px !important;
    border: 2px solid rgba(232, 221, 212, 0.8) !important;
    background: rgba(250, 248, 245, 0.95) !important;
    transition: all 0.3s ease !important;
    padding: 1rem !important;
    backdrop-filter: blur(10px);
}


/* 输出框样式 */
.output-container {
    background: rgba(245, 241, 235, 0.95);
    border-radius: 16px;
    border: 1px solid rgba(232, 221, 212, 0.8);
    padding: 1.5rem;
    color: #4a3f36;
    min-height: 200px;
    backdrop-filter: blur(10px);
}

/* 文件上传区域样式 */
.file-upload {
    border: 2px dashed rgba(196, 181, 166, 0.8) !important;
    border-radius: 20px !important;
    padding: 3rem !important;
    text-align: center !important;
    background: rgba(245, 241, 235, 0.9) !important;
    transition: all 0.3s ease !important;
    backdrop-filter: blur(10px);
}

.file-upload:hover {
    border-color: rgba(139, 115, 85, 0.9) !important;
    background: rgba(240, 235, 228, 0.95) !important;
    transform: translateY(-2px) !important;
}

/* 示例按钮样式 */
.examples-container {
    margin-bottom: 0.5rem !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* 示例问题专用样式 */
.gr-examples {
    margin: 0 !important;
    padding: 0 !important;
}

.gr-examples .examples {
    margin: 0 !important;
    padding: 0.5rem 0 !important;
}

.gr-examples .examples > div {
    gap: 0.3rem !important;
}

.gr-examples button {
    background: rgba(240, 235, 228, 0.95) !important;
    border: 1px solid rgba(232, 221, 212, 0.8) !important;
    border-radius: 18px !important;
    padding: 6px 12px !important;
    color: #6d5a47 !important;
    font-size: 0.8rem !important;
    line-height: 1.2 !important;
    transition: all 0.3s ease !important;
    margin: 2px 3px !important;
    min-height: auto !important;
    height: auto !important;
    white-space: nowrap !important;
    text-overflow: ellipsis !important;
    overflow: hidden !important;
    max-width: 200px !important;
    backdrop-filter: blur(5px);
}

.gr-examples button:hover {
    background: rgba(139, 115, 85, 0.95) !important;
    color: #fff5f0 !important;
    border-color: rgba(139, 115, 85, 0.9) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 2px 6px rgba(139, 115, 85, 0.3) !important;
}

.gr-examples .label {
    display: none !important;
}

/* 移除Gradio默认组件样式 */
.gr-group {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* 右侧面板样式 */
.side-panel {
    background: rgba(237, 232, 224, 0.9);
    padding: 1.2rem;
    border-radius: 16px;
    border-left: 4px solid #7d8471;
    margin-top: 0.8rem;
    backdrop-filter: blur(10px);
    height: fit-content;
    box-shadow: 0 4px 15px rgba(139, 115, 85, 0.15);
}

/* 紧凑标题样式 */
.compact-title {
    margin-bottom: 0.5rem !important;
    font-size: 1rem !important;
}

/* 响应式设计优化 */
@media (max-width: 768px) {
    .gradio-container {
        background-attachment: scroll !important;
    }
    
    .main-title {
        font-size: 2rem !important;
        padding: 1.5rem !important;
        border-radius: 16px;
    }
    
    .card-container {
        padding: 1.5rem !important;
        margin: 1rem !important;
        border-radius: 16px;
    }
    
    .tab-nav button {
        padding: 0.8rem 1.5rem !important;
        font-size: 1rem !important;
    }
    
    .btn-primary {
        padding: 12px 24px !important;
    }
}

/* 加载动画 */
.loading {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(232, 221, 212, 0.5);
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
    width: 10px;
}

::-webkit-scrollbar-track {
    background: rgba(245, 241, 235, 0.8);
    border-radius: 8px;
}

::-webkit-scrollbar-thumb {
    background: rgba(196, 181, 166, 0.8);
    border-radius: 8px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(139, 115, 85, 0.8);
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

            with gr.Row(equal_height=True):
                with gr.Column(scale=3, elem_classes=["content-wrapper"]):
                    question_input = gr.Textbox(
                        label="💭 您的问题",
                        show_label=True,
                        max_lines=3,
                        placeholder="请输入您想了解的问题...",
                        elem_classes=["input-container", "component-spacing"]
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
                        <div style="margin-bottom: 0.5rem;">
                            <h4 style="color: #6d5a47; margin-bottom: 0.5rem; text-align: center; font-size: 1rem;">🎯 示例问题</h4>
                        </div>
                    """)
                    
                    gr.Examples(
                        examples=sample_prompts,
                        inputs=question_input
                    )
                    
                    gr.HTML("""
                        <div class="side-panel">
                            <h4 style="color: #4a3f36; margin-bottom: 0.8rem; text-align: center; font-size: 1rem;">✨ 问答技巧</h4>
                            <ul style="color: #6d5a47; line-height: 1.6; margin: 0; padding-left: 1.2rem; font-size: 0.9rem;">
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