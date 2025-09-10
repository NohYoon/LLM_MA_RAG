# tavily_agent.py

from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

# 설정 파일에서 세팅값 가져오기
from mygpt.config import settings

# --- 도구(Tool) 정의 ---
class TavilySearchToolInput(BaseModel):
    query: str = Field(description="The search query to find information on the web.")

@tool("tavily_web_search", args_schema=TavilySearchToolInput)
def tavily_web_search(query: str) -> str:
    """
    Performs a web search using the Tavily API to find up-to-date information on a given query.
    This tool is ideal for answering questions about recent events, specific topics, or anything that requires current knowledge.
    """
    print(f"\n[Tool Call] 🛠️ tavily_web_search(query='{query}')")
    try:
        tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)
        results = tavily_client.search(query=query, max_results=7, search_depth="advanced")
        context = "\n\n".join([f"URL: {res['url']}\nContent: {res['content']}" for res in results.get('results', [])])
        return context[:4000]
    except Exception as e:
        return f"An error occurred during the search: {e}"

# --- 에이전트 생성 함수 ---
def create_tavily_agent() -> AgentExecutor:
    """
    Tavily 검색 도구를 사용하는 ReAct 에이전트를 생성합니다.
    
    Returns:
        AgentExecutor: 바로 실행 가능한 에이전트 실행기
    """
    print("🤖 Tavily 에이전트를 생성합니다...")

    # 1. LLM 초기화 (config.py 설정 사용)
    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        openai_api_base=settings.VLLM_API_BASE,
        openai_api_key="",  # 사용자의 요청대로 빈 문자열 전달
        temperature=0.1,
        max_tokens=1024,
        model_kwargs={"top_p": 0.95, "top_k": 10}
    )

    # 2. 도구 리스트 정의
    tools = [tavily_web_search]

    # 3. 프롬프트 정의 (하드코딩)
    react_prompt_template = """
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
    prompt = PromptTemplate.from_template(react_prompt_template)

    # 4. 에이전트 및 실행기 생성
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    print("✅ 에이전트 생성이 완료되었습니다.")
    return agent_executor

# main.py
# 2. 생성된 에이전트를 사용하여 질문을 처리합니다.
if __name__ == "__main__":
    tavily_agent_executor = create_tavily_agent()

    # user_input = "2025년 9월 현재, 대한민국의 기준 금리는 몇 퍼센트인가요?"
    user_input = "오늘 대전 날씨 어때?"
    
    response = tavily_agent_executor.invoke({
        "input": user_input
    })
    
    print("\n" + "="*50)
    print("✨ 최종 답변 ✨")
    print("="*50)
    print(response['output'])