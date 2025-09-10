# build_db.py

import os
import argparse
from mygpt.config import settings
from mygpt.tools.ocr_tool import OcrTool
from mygpt.tools.rag_indexer import build_vector_store_from_ocr # 우리가 만든 인덱서 함수

def build(PDF_FILE_PATH):
    if not os.path.exists(PDF_FILE_PATH):
        print(f"🔥 오류: '{PDF_FILE_PATH}' 파일을 찾을 수 없습니다.")
        return

    print(f"🚀 '{PDF_FILE_PATH}' 파일의 인덱싱을 시작합니다...")

    # 1. OCR 도구 초기화
    db_name = os.path.splitext(os.path.basename(PDF_FILE_PATH))[0]
    output_path = os.path.join(settings.VECTOR_DB_BASE_PATH, db_name)
    ocr_tool = OcrTool(timeout=1000)

    # 2. 인덱서 실행 (ocr_tool.py -> rag_indexer.py)
    # 이 함수가 실행되면 config.py에 지정된 경로에 FAISS DB가 생성됩니다.
    build_vector_store_from_ocr(PDF_FILE_PATH, settings, ocr_tool, output_path)

    print(f"\n✅ 인덱싱 완료! 이제 'python main.py'를 실행하여 질문할 수 있습니다.")

if __name__ == "__main__":
    # 인덱싱할 PDF 파일 경로 지정
    #PDF_FILE_PATH = "/home/nyseong/DHInvoice/SHA-RA0297-R00.pdf" # 사용자 예시 경로
    parser = argparse.ArgumentParser(description="PDF 파일로부터 FAISS 벡터 데이터베이스를 생성합니다.")
    
    # '--file_path' 라는 이름의 인자를 추가
    parser.add_argument(
        '--file_path', 
        type=str, 
        required=True, # 필수 인자로 지정
        help='인덱싱할 PDF 파일의 전체 경로'
    )
    
    # 명령줄에서 들어온 인자들을 파싱
    args = parser.parse_args()
    build(PDF_FILE_PATH=args.file_path)
