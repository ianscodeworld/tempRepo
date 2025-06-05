import requests
import json
from typing import Any, List, Dict, Optional, Iterator

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ChatMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult


# 辅助函数：将LangChain消息列表格式化为单个提示字符串
# 这只是一个简单示例，您可能需要根据公司DeepSeek实例处理聊天记录的方式进行调整。
# 您的API似乎期望一个扁平化的prompt。
def _format_messages_to_prompt_str(messages: List[BaseMessage]) -> str:
    """
    将消息列表转换为单个字符串。
    一个简单的实现是拼接所有消息内容，或者只使用最后一条消息。
    根据您API的 "prompt" 字段的预期格式调整此函数。
    """
    # 示例：简单地将所有消息内容串联起来
    # return "\n".join([msg.content for msg in messages])

    # 或者，根据您的API示例（"prompt": "what is you name?"），可能只需要最近的用户输入
    if not messages:
        return ""
    
    # 一个更通用的格式化方式：
    prompt_parts = []
    for msg in messages:
        role_prefix = ""
        if isinstance(msg, HumanMessage):
            role_prefix = "User"
        elif isinstance(msg, AIMessage):
            role_prefix = "Assistant"
        elif isinstance(msg, SystemMessage):
            role_prefix = "System"
        elif isinstance(msg, ChatMessage): # 一般的聊天消息
            role_prefix = msg.role if msg.role else "Generic"
        
        if role_prefix:
            prompt_parts.append(f"{role_prefix}: {msg.content}")
        else:
            prompt_parts.append(msg.content) # 如果没有特定角色，直接添加内容
            
    return "\n".join(prompt_parts)


class CustomChatDeepSeek(BaseChatModel):
    """
    用于公司内部托管的DeepSeek LLM的自定义ChatModel，
    该LLM具有特定的API端点和请求/响应格式。
    """
    api_url: str
    bearer_token: str
    model_kwargs: Dict[str, Any] # 用于传递如 max_tokens, temperature 等参数

    def __init__(
        self,
        api_url: str,
        bearer_token: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any, # 其他 BaseChatModel 参数
    ):
        super().__init__(**kwargs) # 调用父类的构造函数
        self.api_url = api_url
        self.bearer_token = bearer_token
        self.model_kwargs = model_kwargs or {}

    @property
    def _llm_type(self) -> str:
        """返回LLM的类型字符串。"""
        return "custom_company_deepseek"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None, # LangChain的标准参数，此处未直接使用，但可以根据API能力扩展
        run_manager: Optional[CallbackManagerForLLMRun] = None, # LangChain回调管理器
        **kwargs: Any, # 运行时可能传递的额外参数
    ) -> ChatResult:
        """
        与自定义DeepSeek LLM交互的核心方法。
        """
        if not messages:
            raise ValueError("输入的消息列表不能为空。")

        # 1. 将LangChain消息列表格式化为您API所需的prompt字符串
        #    根据您提供的API格式，它似乎需要一个名为 "prompt" 的字段。
        #    这里的 _format_messages_to_prompt_str 是一个示例，您可能需要调整。
        current_prompt_str = _format_messages_to_prompt_str(messages)

        # 2. 构建请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.bearer_token}",
        }

        # 3. 构建请求体 (Body)
        #    合并默认的 model_kwargs 和任何运行时传递的 kwargs。
        #    您的API示例在 "instances" 对象内部显示 "max_tokens"。
        instance_params = self.model_kwargs.copy()
        instance_params.update(kwargs) # 运行时kwargs可以覆盖默认值

        # 确保 'max_tokens' 存在 (如果您的API要求)
        if "max_tokens" not in instance_params:
            instance_params["max_tokens"] = 4096 # 您示例中提供的默认值

        body = {
            "instances": [
                {
                    "prompt": current_prompt_str,
                    **instance_params # 将其他参数如 max_tokens, temperature 等放在这里
                }
            ]
        }

        # 4. 发送API请求
        try:
            response = requests.post(self.api_url, json=body, headers=headers)
            response.raise_for_status()  # 如果HTTP状态码是4xx或5xx，则抛出异常
        except requests.exceptions.RequestException as e:
            raise ValueError(f"API请求失败: {e}")

        # 5. 解析API响应
        #    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #    !!! 关键: 这是您最需要根据您公司API的实际JSON响应结构进行验证和调整的部分 !!!
        #    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #    您的示例只显示了 `response.text`。您需要查看实际的JSON结构。
        #    通常，类似Vertex AI的端点响应会有一个 "predictions" 字段。
        #    例如: { "predictions": [ { "content": "生成的文本..." } ] }
        #    或者: { "predictions": [ "生成的文本" ] }

        response_data = response.json()
        generated_text = ""

        try:
            # 假设 'predictions' 是一个列表，且我们关心第一个预测结果
            if "predictions" in response_data and isinstance(response_data["predictions"], list) and response_data["predictions"]:
                first_prediction = response_data["predictions"][0]

                if isinstance(first_prediction, str):
                    # 情况1: predictions列表直接包含字符串结果
                    generated_text = first_prediction
                elif isinstance(first_prediction, dict):
                    # 情况2: predictions列表包含字典对象
                    # 您需要找到包含生成文本的键。常见的键有 "content", "text", "output", "completion"。
                    # 示例 (根据 Vertex AI 常见格式):
                    if "content" in first_prediction:
                        generated_text = first_prediction["content"]
                    elif "text" in first_prediction: # 备选键
                        generated_text = first_prediction["text"]
                    # ... 您可以添加更多可能的键
                    else:
                        # 如果找不到，打印出来帮助调试
                        print(f"调试信息：在prediction对象中未找到已知文本键: {first_prediction.keys()}")
                        raise ValueError("无法从API响应的prediction对象中提取生成文本。请检查键名。")
                else:
                    print(f"调试信息：prediction的类型不是str或dict: {type(first_prediction)}")
                    raise ValueError("API响应中的prediction格式无法识别。")
            else:
                print(f"调试信息：API响应中缺少 'predictions' 字段，或其为空/格式不正确: {response_data}")
                raise ValueError("API响应中缺少 'predictions' 字段，或其格式不正确。")
        except Exception as e:
            # 确保在解析失败时能看到原始数据
            print(f"调试信息：解析响应时出错。原始响应数据: {response_data}")
            raise ValueError(f"解析API响应失败: {e}")
        
        # --- 调整结束 ---

        # 6. 创建LangChain的输出对象
        ai_message = AIMessage(content=generated_text)
        chat_generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[chat_generation])

    # 如果您的API支持流式输出，您需要实现 _stream 方法
    # def _stream(
    #     self,
    #     messages: List[BaseMessage],
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[CallbackManagerForLLMRun] = None,
    #     **kwargs: Any,
    # ) -> Iterator[ChatGenerationChunk]:
    #     # ... 实现流式逻辑 ...
    #     raise NotImplementedError("此自定义模型尚未实现流式输出。")

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """获取用于识别此LLM的参数。"""
        return {
            "api_url": self.api_url,
            "model_kwargs": self.model_kwargs,
        }


# --- 示例用法 ---
if __name__ == "__main__":
    # 替换为您的实际API URL和Bearer Token
    COMPANY_API_URL = "https://stork.apps.ocpst3.uat.dba.com/v1/projects/187968754875/locations/asia-southeast1/endpoints/4528567337414033408:predict"
    # !! 重要: 请安全地管理您的Bearer Token，不要硬编码在生产代码中 !!
    # !! 例如，可以从环境变量、配置文件或密钥管理服务中读取 !!
    BEARER_TOKEN = "YOUR_ACTUAL_BEARER_TOKEN" 

    if BEARER_TOKEN == "YOUR_ACTUAL_BEARER_TOKEN":
        print("警告: 请将 'BEARER_TOKEN' 替换为您的真实令牌。")
        # exit() # 在实际测试时取消注释此行或提供真实令牌

    # 初始化自定义聊天模型
    custom_llm = CustomChatDeepSeek(
        api_url=COMPANY_API_URL,
        bearer_token=BEARER_TOKEN,
        model_kwargs={"max_tokens": 100, "temperature": 0.7} # 示例模型参数
    )

    # 创建消息列表
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of France?")
    ]
    
    # (可选) 如果你想测试你的_format_messages_to_prompt_str函数
    # test_prompt = _format_messages_to_prompt_str(messages)
    # print(f"Formatted Prompt for API: \n{test_prompt}\n--------------------")

    try:
        # 调用LLM (这将触发 _generate 方法)
        # response = custom_llm.invoke(messages) # invoke 用于单个输入/输出
        
        # 或者使用 generate (更通用，返回 LLMResult)
        llm_result = custom_llm.generate(messages=[messages]) # generate 接收消息列表的列表
        
        # 从 LLMResult 中获取响应
        # llm_result.generations 是一个列表，每个元素对应一批输入消息
        # 每批输入消息的结果是一个 ChatGeneration 列表 (通常只有一个，除非n>1)
        if llm_result.generations and llm_result.generations[0]:
            ai_response_message = llm_result.generations[0][0].message
            print(f"AI Response: {ai_response_message.content}")
        else:
            print("未能从LLMResult获取响应。")

        # 您也可以直接使用 invoke (更简单)
        # response_message = custom_llm.invoke(messages)
        # print(f"AI Response (using invoke): {response_message.content}")

    except ValueError as ve:
        print(f"配置或请求错误: {ve}")
    except Exception as e:
        print(f"发生意外错误: {e}")

    # 测试一个不同的问题
    messages_2 = [HumanMessage(content="what is your name?")]
    try:
        response_2 = custom_llm.invoke(messages_2)
        print(f"AI Response to 'what is your name?': {response_2.content}")
    except Exception as e:
        print(f"对第二个问题的请求失败: {e}")

