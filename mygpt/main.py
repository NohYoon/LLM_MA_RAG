# main.py
from mygpt.graph.graph_builder import ma_rag_graph
from rich.console import Console
from rich.markdown import Markdown

def main():
    app = ma_rag_graph
    console = Console()
    
    while True:
        query = input("질문을 입력하세요 (종료하려면 'exit' 입력): ")
        if query.lower() == 'exit':
            break

        # 최상위 GraphState로 실행
        inputs = {"original_question": query}
        
        console.print("\n--- [MA-RAG 실행 시작] ---", style="bold green")
        for output in app.stream(inputs, {"recursion_limit": 25}):
            for key, value in output.items():
                console.print(f"--- [Top-Level Node]: {key} ---")
        
        final_state = app.invoke(inputs)
        console.print("\n\n" + "="*50)
        console.print("                  [최종 답변]                  ", style="bold blue")
        console.print("="*50)
        console.print(Markdown(final_state["final_answer"]))
        console.print("\n")

if __name__ == "__main__":
    main()