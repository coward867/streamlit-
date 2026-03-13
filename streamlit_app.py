# 导入streamlit，并命名为st
import streamlit as st                                                      #编辑网页的模块

# 设置网页标题
st.title("学习助手")

# 在侧边栏中添加选择框
with st.sidebar:                        #小组件
    # 第一个下拉框选择学科领域
    subject = st.selectbox("选择学科领域", options=["文学", "数学", "计算机"])
    # 第二个下拉框选择讲解风格
    style = st.selectbox("讲解风格", options=["简洁", "详细"])


# 一个聊天输入框，提示用户输入问题
user_input = st.chat_input("你的问题/学习需求")



# 从langchain的chains中导入ConversationChain                                 用于创建对话链
from langchain.chains import ConversationChain
# 从langchain_openai中导入ChatOpenAI类                                       用于创建模型客户端
from langchain_openai import ChatOpenAI
# 从langchain.memory中导入ConversationBufferMemory                           用于创建记忆容器
from langchain.memory import ConversationBufferMemory
# 从langchain.prompts中导入ChatPromptTemplate和MessagesPlaceholder            用于创建提示模板    用于在提示模板中给记忆内容占位
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder



# 判断当前是否已经有会话，如果没有，则初始化消息列表和记忆
if "messages" not in st.session_state:
    # 初始化会话状态，创建一个messages列表来储存聊天记录
    st.session_state["messages"] = [
        {"role": "assistant", "content": "你好，我是你的学习助手！"}
    ]
    # 初始化记忆，指定memory_key为"chat_history"，并设置return_messages为True以返回消息列表
    st.session_state["memory"] = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

# 通过for循环遍历messages列表中的每条消息，依次显示在聊天界面上
for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])


# 获取提示词模板的函数,接收学科领域和讲解风格作为参数,返回一个提示词模板
def get_prompt_template(subject, style):
    # 定义一个字典，将讲解风格映射到具体的描述
    style_dict = {
        "简洁": "仅提供直接答案和最少的必要解释。不要添加额外细节、发散讨论或无关信息。保持回答清晰、简洁，目标是为用户快速提供解决方案。",
        "详细": "第一，针对用户提问给出直接答案和清晰的解释；第二，基于此提供必要的相关知识点的信息，以补充背景或加深理解。",
    }
    # 定义系统提示模板，预留占位符给学科领域和讲解风格的描述
    system_template = "你是{subject}领域的专家，根据用户提问作出回答。\n你需要遵循以下讲解风格：{style}。\n你应当礼貌拒绝与该学科无关的问题。"
    # 创建一个聊天提示模板，包含系统提示、对话记忆的占位和用户输入的占位符
    prompt_template = ChatPromptTemplate(
        [
            ("system", system_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ],
        # 通过部分变量传递学科领域和讲解风格的描述
        partial_variables={"subject": subject, "style": style_dict[style]},
    )
    # 返回创建好的提示词模板
    return prompt_template


# 生成AI回复的函数,接收用户输入、学科领域、讲解风格和记忆作为参数,返回AI生成的回复内容
def generate_response(user_input, subject, style, memory):
    # 创建模型客户端
    client = ChatOpenAI(
        # 使用练习密钥"mock-key-123"
        api_key=st.secrets["OPENAI_API_KEY"],
        # 使用的模型 deepseek-chat
        model="deepseek-chat",
        # 指定Deepseek的API服务地址
        base_url="https://api.deepseek.com",
        # 设置很低的温度值，以使助手生成的回答更可靠
        temperature=0.0,
    )
    # 获取提示词模板，直接调用get_prompt_template来生成
    prompt = get_prompt_template(subject, style)
    # 创建对话链，分别将模型客户端、提示词和记忆传入
    chain = ConversationChain(llm=client, memory=memory, prompt=prompt)
    # 调用对话链的invoke方法，传入用户输入，获取AI生成的回复
    response = chain.invoke({"input": user_input})
    # 将AI生成的回复内容返回，作为函数的输出
    return response["response"]


# 判断是否有用户输入，如果有，则处理输入并生成AI回复
if user_input:
    # 将用户输入显示在聊天界面
    st.chat_message("human").write(user_input)
    # 将用户输入添加到会话状态的消息列表中，以便下一轮对话时能够显示
    st.session_state["messages"].append({"role": "human", "content": user_input})
    # 使用spinner显示AI思考的过程，并调用generate_response函数生成回复
    with st.spinner("AI正在思考中，请稍等..."):
        response = generate_response(
            user_input, subject, style, st.session_state["memory"]
        )
    # 将AI生成的回复显示在聊天界面
    st.chat_message("assistant").write(response)
    # 将AI生成的回复添加到会话状态的消息列表中，以便下一轮对话时能够显示
    st.session_state["messages"].append({"role": "assistant", "content": response})