集成公司内部 DeepSeek LLM 到 LangChain
要在 LangChain 中使用您公司部署的、具有特定 URL 和请求格式的 DeepSeek LLM，最好的方法是创建一个自定义的聊天模型类。这个类将继承自 LangChain 的 BaseChatModel，并实现与您公司 API 端点交互的逻辑。

关键步骤：
创建自定义类: 定义一个新类（例如 CustomChatDeepSeek），继承自 langchain_core.language_models.chat_models.BaseChatModel。

初始化方法 (__init__): 构造函数应接收必要的参数，如 API URL、Bearer Token 以及可能的模型参数（如 max_tokens）。

实现 _generate 方法: 这是核心方法。它接收一个消息列表作为输入，需要：

将 LangChain 的消息格式转换为您 API 所需的 prompt 格式。根据您的示例，API 似乎接受一个简单的字符串作为 prompt。

构造符合您公司 API 要求的请求头（Headers）和请求体（Body）。

使用 requests (或其他HTTP客户端) 发送 POST 请求到指定的 URL。

解析 API 返回的 JSON 响应，从中提取出模型生成的文本。这是最需要您根据实际情况调整的部分。

将提取的文本包装成 LangChain 的 AIMessage 和 ChatResult 对象。

实现 _llm_type 属性: 返回一个标识该模型类型的字符串。

（可选）实现 _stream 方法: 如果您的 API 支持流式输出并且您需要此功能。

（可选）_identifying_params 属性: 返回用于识别模型配置的参数字典，有助于缓存和日志记录。

注意事项：
Bearer Token: 请确保安全地管理和提供您的 Bearer Token。在示例中，它作为参数传递给构造函数。

请求体 (Body) 构建: 严格按照您公司 API 的 body 格式构建，特别是 instances 数组中的内容。

响应体 (Response) 解析: 您提供的示例代码中 print(response.text) 只打印了原始响应。您需要检查实际的 JSON 结构，并修改代码以正确提取生成的文本。常见的响应结构可能包含在 predictions 字段中。URL 格式 (/v1/projects/.../endpoints/...:predict) 暗示它可能是 Google Cloud Vertex AI 风格的端点，其响应通常在 predictions 数组中，每个元素对应请求中的一个 instance。

错误处理: 代码应包含适当的错误处理，例如网络请求失败或 API 返回错误状态码。

消息格式化: _format_messages_to_prompt 函数提供了一个将多轮对话消息转换为单个字符串提示的简单示例。您可能需要根据模型的具体要求调整此逻辑。您的API似乎接受一个单一的 prompt 字符串。

接下来是具体的代码实现。