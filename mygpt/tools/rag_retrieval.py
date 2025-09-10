# rag_retriever.py
import pickle
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import LongContextReorder

class AdvancedQueryPipeline:
    def __init__(self, config):
        print("🔍 1. 고급 쿼리 파이프라인 초기화 중...")
        self.config = config

        # --- 임베딩 및 벡터 저장소 로드 ---
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs=config.MODEL_KWARGS,
            encode_kwargs=config.ENCODE_KWARGS
        )
        vector_store = FAISS.load_local(
            config.VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
        )
        faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

        # --- 하이브리드 검색(Hybrid Search) 설정 ---
        print("🤝 2. 하이브리드 검색(BM25 + FAISS) 설정 중...")
        with open(f"{config.VECTOR_STORE_PATH}/chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 10

        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
        )

        # --- LLM을 이용한 쿼리 변환(Multi-Query) 설정 ---
        print("🔄 3. 쿼리 변환(Multi-Query) 설정 중...")
        llm = ChatOpenAI(model=config.LLM_MODEL_NAME, temperature=0)
        self.multiquery_retriever = MultiQueryRetriever.from_llm(
            retriever=self.hybrid_retriever, llm=llm
        )

        # --- Reranker 및 후처리(Reordering) 설정 ---
        print("✨ 4. Reranker 및 후처리 파이프라인 설정 중...")
        reranker = HuggingFaceCrossEncoder(model_name=config.RERANKER_MODEL)
        reordering = LongContextReorder()
        
        compressor = CrossEncoderReranker(model=reranker, top_n=5)
        # 압축 파이프라인: Reranker 실행 후 -> LongContextReorder 실행
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[compressor, reordering]
        )
        
        # 최종 Retriever: 쿼리 변환 -> 하이브리드 검색 -> Reranking -> Reordering
        self.final_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=self.multiquery_retriever
        )
        print("✅ 고급 쿼리 파이프라인이 준비되었습니다.")

    def query(self, question: str):
        print(f"\n💬 질문: '{question}'")
        final_docs = self.final_retriever.invoke(question)
        print(f"🎯 총 {len(final_docs)}개의 최종 문서 반환.")
        return final_docs