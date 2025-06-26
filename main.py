import openai
import os

# 从DEEPSEEK, CHATGLM, QWEN, SILICONFLOW, PISCES中选一个
PLATFORM = "CHATGLM"
API_KEY = "b06d082ae340463eabb99592497f8d93.Hcm3LN1pf3X7agWR"  # 你申请的API Key

if PLATFORM == "DEEPSEEK":
    global_model = "deepseek-chat"
    base_url = "https://api.deepseek.com/v1/"
elif PLATFORM == "CHATGLM":
    global_model = "GLM-4-Flash-250414"
    base_url = "https://open.bigmodel.cn/api/paas/v4/"
elif PLATFORM == "QWEN":
    global_model = "qwen-plus"
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/"
elif PLATFORM == "SILICONFLOW":
    global_model = "deepseek-ai/DeepSeek-V3"
    base_url = "https://api.siliconflow.cn/v1/"
else:
    global_model = "chatgpt-4o-latest"
    base_url = "https://api.pisces.ink/v1/"

client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", API_KEY),
    base_url=base_url,
)

# 用于单轮对话
def get_completion(prompt, model=global_model, temperature=0):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,  # 控制模型输出的随机程度
        )
        return response.choices[0].message.content
    except Exception as e:
        # print(f"API 调用出错: {e}")
        return f"API Error: {e}"

# 用于多轮对话
def get_completion_from_messages(messages, model=global_model, temperature=0):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature  # 控制模型输出的随机程度
            )
            return response.choices[0].message.content
        except Exception as e:
            # print(f"API 调用出错: {e}")
            return f"API Error: {e}"