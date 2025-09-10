
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from mygpt.config import settings
from langchain.prompts import PromptTemplate

# Research Agent가 사용할 도구들을 임포트합니다.
from mygpt.tools.web_search_tool import tavily_web_search
from mygpt.tools.pdf_search_tool import pdf_tool

def create_research_agent() -> AgentExecutor:
    """
    PDF 및 웹 검색 도구를 사용하여 스스로 추론하고 질문에 답변하는
    ReAct 기반의 연구원 에이전트를 생성합니다.
    """
    print("🤖 ReAct 기반 연구원 에이전트를 생성합니다...")

    # 1. LLM 초기화
    # config.py의 설정을 사용하거나, 이 에이전트 전용 LLM을 설정할 수 있습니다.
    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        openai_api_base=settings.VLLM_API_BASE,
        openai_api_key="anything",
        temperature=0
    )

    # 2. 도구 리스트 정의
    # 이 에이전트는 PDF 검색과 웹 검색을 모두 할 수 있습니다.
    tools = [tavily_web_search, pdf_tool]

    # 3. ReAct 프롬프트 가져오기
    # LangChain Hub에 있는 검증된 ReAct 프롬프트를 사용합니다.
    original_prompt =  """

Answer the following questions as best you can. You have access to the following tools:



{tools}



Use the following format:



Question: the input question you must answer

Thought: you should always think about what to do

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action

Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know the final answer

Final Answer: the final answer to the original input question



Begin!



Question: {input}

Thought:{agent_scratchpad}

"""
    prompt = PromptTemplate.from_template(original_prompt)
    # 4. 에이전트 및 실행기 생성
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True # 파싱 에러 발생 시 재시도
    )
    
    print("✅ 연구원 에이전트 생성이 완료되었습니다.")
    return agent_executor