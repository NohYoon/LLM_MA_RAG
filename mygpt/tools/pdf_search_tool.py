# tools/pdf_search_tool.py

import os
import pickle
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from mygpt.config import settings

# --- Tool 입력 스키마 정의 ---
class PdfSearchToolInput(BaseModel):
    query: str = Field(description="PDF 문서 내에서 찾고 싶은 내용")
    db_name: str = Field(description="검색할 벡터 DB의 이름 (확장자를 제외한 원본 파일명)")

# --- 전역 변수: 성능 최적화를 위한 캐시 ---
# 로드된 DB와 Retriever를 메모리에 저장하여 반복적인 디스크 I/O와 객체 생성을 방지
db_cache = {}

# --- 사용자 정의 PDF 검색 Tool ---
@tool("pdf_document_searcher", args_schema=PdfSearchToolInput)
def pdf_tool(query: str, db_name: str) -> str:
    """
    미리 인덱싱된 특정 PDF 문서의 벡터 DB에서 관련 정보를 검색합니다.
    'db_name'으로 검색할 문서를 정확히 지정해야 합니다.
    """
    print(f"\n[Tool Call] 🛠️  PDF Search(query='{query}', db_name='{db_name}')")
    
    try:
        # 1. 캐시에서 해당 DB의 Retriever가 있는지 확인
        if db_name in db_cache:
            ensemble_retriever = db_cache[db_name]
            print(f"   (캐시에서 '{db_name}' DB Retriever 로드)")
        else:
            # 2. 캐시에 없으면 디스크에서 DB를 로드하여 Retriever 생성
            db_path = os.path.join(settings.VECTOR_DB_BASE_PATH, db_name)
            if not os.path.exists(db_path):
                return f"오류: '{db_name}'에 해당하는 벡터 DB를 찾을 수 없습니다. 먼저 인덱싱을 수행해야 합니다."

            print(f"   ('{db_path}'에서 DB 로딩 및 하이브리드 Retriever 생성 중...)")
            
            # FAISS 벡터 저장소 로드
            embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs=settings.MODEL_KWARGS,
                encode_kwargs=settings.ENCODE_KWARGS
            )
            vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
            faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

            # BM25 Retriever를 위한 chunks.pkl 로드
            with open(os.path.join(db_path, "chunks.pkl"), "rb") as f:
                chunks = pickle.load(f)
            bm25_retriever = BM25Retriever.from_documents(chunks)
            bm25_retriever.k = 5
            
            # 하이브리드 Ensemble Retriever 생성
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
            )
            
            # 생성된 Retriever를 캐시에 저장
            db_cache[db_name] = ensemble_retriever
        
        # 3. 하이브리드 검색 수행
        docs = ensemble_retriever.invoke(query)
        
        if not docs:
            return "관련된 내용을 PDF 문서에서 찾을 수 없습니다."
        
        # 4. 결과 포맷팅
        formatted_results = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', '알 수 없음')
            content_preview = doc.page_content.replace('\n', ' ').strip()
            formatted_results.append(
                f"[{i+1}] 출처: {source}\n"
                f"내용: {content_preview}"
            )
        
        return "\n\n".join(formatted_results)

    except Exception as e:
        return f"PDF 검색 중 오류 발생: {e}"