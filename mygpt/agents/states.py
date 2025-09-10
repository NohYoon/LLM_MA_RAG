from typing import List, TypedDict, Annotated, Deque
import operator

class QAAnswerState(TypedDict):
    analysis: str
    answer: str
    success: str
    rating: int

class PlanState(TypedDict):
    analysis: str
    step: List[str]

class StepTaskState(TypedDict):
    type: str  # "search" or "aggregate"
    task: str

class PlanSummaryState(TypedDict):
    output: str
    answer: str
    score : int

class PlanExecState(TypedDict):
    original_question: str
    plan: List[str]
    current_step: str
    step_question: Annotated[List[StepTaskState], operator.add]
    step_output: Annotated[List[QAAnswerState], operator.add]
    step_docs_ids: Annotated[List[List[str]], operator.add]
    step_notes: Annotated[List[List[str]], operator.add]
    plan_summary: PlanSummaryState
    stop: bool = False

class RagState(TypedDict):
    question: str
    documents: List[str]
    doc_ids: List[str]
    notes: List[str]
    final_raw_answer: QAAnswerState

class GraphState(TypedDict):
    original_question: str
    plan: List[str]
    past_exp: Annotated[List[PlanExecState], operator.add]
    final_answer: str