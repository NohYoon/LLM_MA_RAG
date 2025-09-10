# agents/agent_definitions.py
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from typing import List
from mygpt.agents.prompts import *
from mygpt.config import config

# --- LLM 정의 ---
llm = ChatOpenAI(
    # 1. 사용할 모델 이름 (vLLM으로 서빙 중인 모델)
    model="openai/gpt-oss-20b", 
    
    # 2. 기존에 설정하신 temperature 값
    temperature=0,
    
    # 3. vLLM 서버 주소 (OpenAI API 형식에 맞춰 '/v1'을 추가합니다)
    base_url="http://localhost:8003/v1",
    
    # 4. 로컬 서버이므로 API 키는 필요 없지만, 형식상 아무 값이나 입력합니다.
    api_key="EMPTY" 
)

# --- 에이전트 출력을 위한 Pydantic 모델 ---
class Plan(BaseModel):
    steps: List[str] = Field(description="List of steps to answer the question")

class StepTask(BaseModel):
    type: str = Field(description="Type of the task, e.g., 'search' or 'aggregate'")
    task: str = Field(description="The detailed task or query for the current step")

class ExtractedNotes(BaseModel):
    notes: List[str] = Field(description="List of extracted notes from the passage")

class QAAnswer(BaseModel):
    analysis: str = Field(description="Analysis of the retrieved information")
    answer: str = Field(description="Concise answer to the question")
    success: bool = Field(description="Whether the answer was successfully found")
    rating: int = Field(description="Confidence rating from 1 to 5")

class PlanSummary(BaseModel):
    output: str = Field(description="Summary of the entire execution process")
    answer: str = Field(description="The final, comprehensive answer to the original question")
    score: int = Field(description="Overall quality score of the final answer")


# --- 에이전트 체인 생성 함수 ---
def create_planner_agent():
    return planner_prompt | llm.with_structured_output(Plan)

def create_step_definer_agent():
    return step_definer_prompt | llm.with_structured_output(StepTask)

def create_extractor_agent():
    return extractor_prompt | llm.with_structured_output(ExtractedNotes)

def create_qa_agent():
    return qa_agent_prompt | llm.with_structured_output(QAAnswer)

def create_final_summary_agent():
    return final_summary_prompt | llm.with_structured_output(PlanSummary)