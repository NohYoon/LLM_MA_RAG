# rag_indexer.py
import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from mygpt.tools.ocr_tool import OcrTool

def build_vector_store_from_ocr(file_path: str, config, ocr_tool: OcrTool, output_path: str):
    """
    OcrTool을 사용하여 문서를 마크다운으로 변환하고,
    마크다운 구조를 기반으로 청킹하여 FAISS 벡터 저장소를 생성합니다.

    Args:
        file_path (str): 인덱싱할 PDF 또는 이미지 파일 경로
        config (RAGConfig): 설정 객체
        ocr_tool (OcrTool): 초기화된 OcrTool 인스턴스
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    os.makedirs(output_path, exist_ok=True)

    print(f"🔬 1. OCR Tool로 '{os.path.basename(file_path)}' 파일을 마크다운으로 변환 중...")
    # ocr_tool을 사용하여 전체 문서를 하나의 마크다운 텍스트로 변환
    markdown_text = ocr_tool.get_markdown_from_file(file_path)
    
    # 마크다운 헤더를 기준으로 1차 청킹
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_text)

    print(f"🧱 2. 구조적 청킹(Markdown Splitting) 진행 중...")
    # 1차 청킹된 결과가 너무 클 경우를 대비해 2차 재귀적 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(md_header_splits)

    # 각 청크에 원본 파일명 메타데이터 추가
    for chunk in chunks:
        chunk.metadata["source"] = os.path.basename(file_path)

    print(f"📊 총 {len(chunks)}개의 구조 기반 청크 생성 완료.")
    
    print("🧠 3. 임베딩 모델 로딩 및 벡터 저장소 생성 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs=config.MODEL_KWARGS,
        encode_kwargs=config.ENCODE_KWARGS
    )
    vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(output_path)
    
    with open(os.path.join(output_path, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ 벡터 저장소와 청크 데이터가 '{output_path}' 경로에 성공적으로 저장되었습니다.")
