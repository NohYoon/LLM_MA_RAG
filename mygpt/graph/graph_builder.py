# graph/graph_builder.py
from langgraph.graph import StateGraph, END
from collections import deque
from mygpt.agents.states import GraphState, PlanExecState, RagState
from mygpt.agents.agent_definitions import *
from mygpt.agents.research_agent import create_research_agent 
research_agent_executor = create_research_agent()
# ===================================================================
# == (c) single-task-execute graph: 단일 작업 실행 서브그래프 ==
# ===================================================================
def run_retrieval(state: RagState):
    """
    단순 도구 호출 대신, ReAct 에이전트에게 작업 수행을 위임합니다.
    에이전트는 스스로 PDF와 웹 검색 중 적절한 도구를 선택하고 사용합니다.
    """
    print(f"--- [Sub-Graph c] Retrieval via ReAct Agent ---")
    question = state["question"]
    
    # ReAct 에이전트 실행
    response = research_agent_executor.invoke({"input": question})
    
    # 에이전트의 최종 답변을 문서(documents)로 반환
    return {"documents": [response['output']]}

def run_extractor(state: RagState):
    print("--- [Sub-Graph c] Extract ---")
    extractor = create_extractor_agent()
    notes = extractor.invoke({
        "passage": "\n\n".join(state["documents"]),
        "question": state["question"]
    })
    return {"notes": notes.notes}

def run_qa_generator(state: RagState):
    print("--- [Sub-Graph c] Generate ---")
    qa_agent = create_qa_agent()
    answer = qa_agent.invoke({
        "context": "\n".join(state["notes"]),
        "question": state["question"]
    })
    return {"final_raw_answer": answer.dict()}

# (c) single-task-execute 서브그래프 빌드
single_task_workflow = StateGraph(RagState)
single_task_workflow.add_node("retrieve", run_retrieval)
single_task_workflow.add_node("extract", run_extractor)
single_task_workflow.add_node("generate", run_qa_generator)
single_task_workflow.set_entry_point("retrieve")
single_task_workflow.add_edge("retrieve", "extract")
single_task_workflow.add_edge("extract", "generate")
single_task_workflow.set_finish_point("generate")
single_task_execute_subgraph = single_task_workflow.compile()


# ===================================================================
# == (b) plan-executor-node graph: 계획 실행 루프 서브그래프 ==
# ===================================================================
def run_task_definer(state: PlanExecState):
    """루프의 컨트롤러: 다음 작업이 있으면 정의하고, 없으면 루프 종료 신호를 보냄"""
    print("--- [Sub-Graph b] Task Definer ---")
    if not state["plan"]:
        print("모든 계획이 완료되었습니다. 루프를 종료합니다.")
        return {"current_step": "END_OF_PLAN"}

    current_step = state["plan"].pop(0)
    print(f"Current Step: '{current_step}'")
    
    step_definer = create_step_definer_agent()
    memory = "\n".join([f"- {s['answer']}" for s in state["step_output"]])
    task = step_definer.invoke({
        "plan": state["plan"], "cur_step": current_step, "memory": memory
    })
    return {"current_step": current_step, "plan": state["plan"], "step_question": [task.dict()]}

def run_single_task_execute(state: PlanExecState):
    """(c) 서브그래프를 호출하여 단일 작업을 실행"""
    print("--- [Sub-Graph b] Single Task Execute ---")
    step_task = state["step_question"][-1]
    
    # (c) 그래프를 위한 RagState 초기화
    rag_state: RagState = {"question": step_task["task"], "documents": [], "doc_ids": [], "notes": [], "final_raw_answer": {}}
    
    # (c) 서브그래프 실행
    rag_result = single_task_execute_subgraph.invoke(rag_state)
    
    # 결과를 PlanExecState에 추가
    return {
        "step_notes": [rag_result["notes"]],
        "step_output": [rag_result["final_raw_answer"]]
    }

def should_execute_task(state: PlanExecState):
    """Task Definer의 결과에 따라 루프를 계속할지 종료할지 결정"""
    if state["current_step"] == "END_OF_PLAN":
        return "end"
    else:
        return "execute_task"

# (b) plan-executor 서브그래프 빌드
plan_executor_workflow = StateGraph(PlanExecState)
plan_executor_workflow.add_node("task_definer", run_task_definer)
plan_executor_workflow.add_node("single_task_execute", run_single_task_execute)
plan_executor_workflow.set_entry_point("task_definer")
plan_executor_workflow.add_conditional_edges(
    "task_definer",
    should_execute_task,
    {
        "execute_task": "single_task_execute",
        "end": END
    }
)
plan_executor_workflow.add_edge("single_task_execute", "task_definer") # 루프: 작업이 끝나면 다시 Task Definer로
plan_executor_subgraph = plan_executor_workflow.compile()


# ===================================================================
# == (a) MA-RAG graph: 최상위 그래프 ==
# ===================================================================
def run_planner_node(state: GraphState):
    print("--- [Main Graph a] Planner Node ---")
    planner = create_planner_agent()
    plan = planner.invoke({"Question": state["original_question"]})
    return {"plan": plan.steps}

def run_plan_executor_node(state: GraphState):
    print("--- [Main Graph a] Plan Executor Node ---")
    
    # (b) 그래프를 위한 PlanExecState 초기화
    plan_exec_state: PlanExecState = {
        "original_question": state["original_question"],
        "plan": state["plan"],
        "step_question": [], "step_output": [], "step_docs_ids": [], "step_notes": [], "plan_summary": {}
    }

    # (b) 서브그래프 실행
    final_plan_state = plan_executor_subgraph.invoke(plan_exec_state, {"recursion_limit": 25})
    
    # 최종 답변 생성
    print("--- [Main Graph a] Final Answer Generation ---")
    summarizer = create_final_summary_agent()
    all_answers = "\n".join([f"Step Answer: {s['answer']}" for s in final_plan_state["step_output"]])
    summary = summarizer.invoke({
        "original_question": state["original_question"],
        "all_step_answers": all_answers
    })
    return {"final_answer": summary.answer}

# (a) 최상위 MA-RAG 그래프 빌드
main_workflow = StateGraph(GraphState)
main_workflow.add_node("planner_node", run_planner_node)
main_workflow.add_node("plan_executor_node", run_plan_executor_node)
main_workflow.set_entry_point("planner_node")
main_workflow.add_edge("planner_node", "plan_executor_node")
main_workflow.set_finish_point("plan_executor_node")
ma_rag_graph = main_workflow.compile()