import os

from langchain.chat_models import QianfanChatEndpoint
from langchain.schema.messages import HumanMessage

# 设置环境变量
os.environ["QIANFAN_AK"] = "wLRppEb0pBay9jZmDyZeG0Zk"
os.environ["QIANFAN_SK"] = "bzqLAMI57mtMttrS54O0ycpBaz8Ggdsf"

# 初始化模型
chatBloom = QianfanChatEndpoint(
   streaming=True,
   model="ERNIE-Bot",
)

def create_poem(question):
    """
    Prompts the model to create a modern poem based on the input sentence.
    """
    poetry_prompt = f"请用我提供句子，用中文生成很长的现代诗，一定要长一点，: '{question}'"
    response = chatBloom([HumanMessage(content=poetry_prompt)])
    return response.content if response and hasattr(response, 'content') else "No response generated."



# # 使用示例
# question = "What is the meaning of life?"
# poem = create_poem(question)
# print(poem)
