from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool
from tavily import TavilyClient
from mygpt.config import settings # 👈 config.py에서 TAVILY_API_KEY를 가져오기 위함

# --- 도구(Tool)의 입력 스키마 정의 ---
class WebSearchToolInput(BaseModel):
    query: str = Field(description="웹에서 정보를 검색하기 위한 검색어")

# --- 사용자 정의 웹 검색 Tool ---
@tool("web_search", args_schema=WebSearchToolInput)
def tavily_web_search(query: str) -> str:
    """
    주어진 쿼리에 대한 최신 정보를 찾기 위해 Tavily API를 사용하여 웹 검색을 수행합니다.
    최신 이벤트, 특정 주제 등 현재 지식이 필요한 질문에 답변하는 데 이상적입니다.
    """
    print(f"\n[Tool Call] 🛠️ Custom Web Search(query='{query}')")
    try:
        # settings.py 파일이 아닌 config.py의 settings 객체를 사용하도록 수정
        # TAVILY_API_KEY는 config.py의 Settings 클래스에 정의되어 있어야 합니다.
        tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)
        
        # 사용자 코드의 장점(search_depth="advanced")을 그대로 활용
        results = tavily_client.search(query=query, max_results=5, search_depth="advanced")
        
        context = "\n\n".join([f"URL: {res['url']}\nContent: {res['content']}" for res in results.get('results', [])])
        
        # 컨텍스트 길이 제한
        return context[:5000]
    except Exception as e:
        return f"웹 검색 중 오류가 발생했습니다: {e}"