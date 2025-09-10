import requests
import base64
import os
from typing import List

# 설정 파일에서 API 엔드포인트 등을 가져옵니다.
from mygpt.config import settings

class OcrTool:
    """
    파일 경로를 입력받아 이미지 또는 PDF에서 텍스트를 추출하여
    마크다운 형식으로 변환하는 OCR 도구 클래스입니다.
    
    RAG 엔진의 문서 처리 파이프라인에 통합하여 사용할 수 있습니다.
    """
    
    def __init__(self, timeout: int = 120):
        """
        OcrTool 클래스를 초기화합니다.

        Args:
            timeout (int): API 요청 시 타임아웃 시간 (초).
        """
        # 지원하는 이미지 파일 확장자를 정의합니다.
        self.SUPPORTED_IMAGE_EXTENSIONS: List[str] = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
        self.SUPPORTED_PDF_EXTENSION: str = '.pdf'
        
        # API 엔드포인트를 설정 객체에서 가져옵니다.
        self.image_api_url: str = settings.IMAGE_API_BASE
        self.pdf_api_url: str = settings.PDF_API_BASE
        self.timeout: int = timeout

    def _process_image(self, file_path: str) -> str:
        """지정된 이미지 파일을 Base64로 인코딩하여 서버로 전송하고 결과를 반환합니다."""
        print(f"🖼️  Processing image file: {file_path}")
        try:
            with open(file_path, 'rb') as image_file:
                # 이미지를 Base64로 인코딩하고 utf-8 문자열로 변환합니다.
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

            payload = {
                'filename': os.path.basename(file_path),
                'image_base64': encoded_string
            }
            
            response = requests.post(self.image_api_url, json=payload, timeout=self.timeout)
            response.raise_for_status()  # 2xx 상태 코드가 아니면 예외를 발생시킵니다.
            
            return response.json()['markdown_output']

        except FileNotFoundError:
            print(f"🔥 Error: The file was not found at {file_path}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"🔥 API Request Failed: {e}")
            raise

    def _process_pdf(self, file_path: str) -> str:
        """지정된 PDF 파일을 서버로 전송하고 결과를 반환합니다."""
        print(f"📄 Processing PDF file: {file_path}")
        try:
            with open(file_path, 'rb') as f:
                # 'multipart/form-data'로 파일을 전송합니다.
                files = {'file': (os.path.basename(file_path), f, 'application/pdf')}
                response = requests.post(self.pdf_api_url, files=files, timeout=self.timeout)
                response.raise_for_status() # 2xx 상태 코드가 아니면 예외를 발생시킵니다.
                
                return response.json()['markdown_output']

        except FileNotFoundError:
            print(f"🔥 Error: The file was not found at {file_path}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"🔥 API Request Failed: {e}")
            raise

    def get_markdown_from_file(self, file_path: str) -> str:
        """
        파일 경로를 받아 확장자를 확인하고, 적절한 처리 함수를 호출하여
        마크다운 결과를 반환합니다.

        Args:
            file_path (str): 처리할 이미지 또는 PDF 파일의 경로.

        Returns:
            str: 서버로부터 받은 마크다운 텍스트.

        Raises:
            ValueError: 지원하지 않는 파일 형식일 경우 발생합니다.
            FileNotFoundError: 파일이 존재하지 않을 경우 발생합니다.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No such file or directory: '{file_path}'")
            
        # 파일 확장자를 소문자로 가져옵니다.
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        if file_extension in self.SUPPORTED_IMAGE_EXTENSIONS:
            return self._process_image(file_path)
        elif file_extension == self.SUPPORTED_PDF_EXTENSION:
            return self._process_pdf(file_path)
        else:
            supported_formats = self.SUPPORTED_IMAGE_EXTENSIONS + [self.SUPPORTED_PDF_EXTENSION]
            raise ValueError(f"Unsupported file type: '{file_extension}'. Supported types are: {supported_formats}")


# ----------------------------------------------------
# 예제 사용법
# ----------------------------------------------------
if __name__ == '__main__':
    # OcrTool 인스턴스 생성
    ocr_tool = OcrTool()

    # --- 이미지 파일 처리 예제 ---
    # 실제 존재하는 이미지 파일 경로로 수정해야 합니다.
    IMAGE_FILE = 'sample_image.png' 
    # 테스트용 샘플 이미지 파일 생성 (실제로는 이 부분을 주석 처리하고 자신의 파일을 사용하세요)
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (400, 100), color = (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((10,10), "This is a sample image for OCR test.", fill=(0,0,0))
        img.save(IMAGE_FILE)
        print(f"'{IMAGE_FILE}' for testing has been created.")
    except ImportError:
        print("Pillow is not installed. Please create a 'sample_image.png' manually to run this test.")
        
    if os.path.exists(IMAGE_FILE):
        try:
            print("\n" + "="*20)
            markdown_from_image = ocr_tool.get_markdown_from_file(IMAGE_FILE)
            print("✅ Image to Markdown Success!")
            print("--- Result ---")
            print(markdown_from_image)
            print("="*20 + "\n")
        except Exception as e:
            print(f"An error occurred while processing the image: {e}")

    # --- PDF 파일 처리 예제 ---
    # 실제 존재하는 PDF 파일 경로로 수정해야 합니다.
    PDF_FILE = "C:/Users/nyseo/Desktop/9608300 2490 INQ.pdf" # 사용자 예시 경로
    
    if os.path.exists(PDF_FILE):
        try:
            print("="*20)
            markdown_from_pdf = ocr_tool.get_markdown_from_file(PDF_FILE)
            print("✅ PDF to Markdown Success!")
            print("--- Result ---")
            print(markdown_from_pdf)
            print("="*20 + "\n")
        except Exception as e:
            print(f"An error occurred while processing the PDF: {e}")
    else:
        print(f"⚠️ PDF file for testing not found at '{PDF_FILE}'. Skipping PDF test.")

if __name__ == '__main__':
    # OcrTool 인스턴스 생성
    ocr_tool = OcrTool()

    # --- 이미지 파일 처리 예제 ---
    # 실제 존재하는 이미지 파일 경로로 수정해야 합니다.
    IMAGE_FILE = 'sample_image.png' 
    # 테스트용 샘플 이미지 파일 생성 (실제로는 이 부분을 주석 처리하고 자신의 파일을 사용하세요)
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (400, 100), color = (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((10,10), "This is a sample image for OCR test.", fill=(0,0,0))
        img.save(IMAGE_FILE)
        print(f"'{IMAGE_FILE}' for testing has been created.")
    except ImportError:
        print("Pillow is not installed. Please create a 'sample_image.png' manually to run this test.")
        
    if os.path.exists(IMAGE_FILE):
        try:
            print("\n" + "="*20)
            markdown_from_image = ocr_tool.get_markdown_from_file(IMAGE_FILE)
            print("✅ Image to Markdown Success!")
            print("--- Result ---")
            print(markdown_from_image)
            print("="*20 + "\n")
        except Exception as e:
            print(f"An error occurred while processing the image: {e}")

    # --- PDF 파일 처리 예제 ---
    # 실제 존재하는 PDF 파일 경로로 수정해야 합니다.
    PDF_FILE = "C:/Users/nyseo/Desktop/9608300 2490 INQ.pdf" # 사용자 예시 경로
    
    if os.path.exists(PDF_FILE):
        try:
            print("="*20)
            markdown_from_pdf = ocr_tool.get_markdown_from_file(PDF_FILE)
            print("✅ PDF to Markdown Success!")
            print("--- Result ---")
            print(markdown_from_pdf)
            print("="*20 + "\n")
        except Exception as e:
            print(f"An error occurred while processing the PDF: {e}")
    else:
        print(f"⚠️ PDF file for testing not found at '{PDF_FILE}'. Skipping PDF test.")
