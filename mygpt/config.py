# config.py: API 키 및 설정 관리

import os

class Settings:
	def __init__(self):
		self.TAVILY_API_KEY = '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'
		self.LLM_MODEL = "openai/gpt-oss-20b"
		self.VLLM_API_BASE = "http://192.168.219.94:8003/v1"
		self.IMAGE_API_BASE = 'http://192.168.219.94:8002/upload-base64/'
		self.PDF_API_BASE = 'http://192.168.219.94:8002/upload-file/'
		self.EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
		self.MODEL_KWARGS = {"device": "cuda"}
		self.ENCODE_KWARGS = {"normalize_embeddings": True}
		self.RERANKER_MODEL = "mixedbread.ai/mxbai-rerank-xsmall-v1"
		self.VECTOR_STORE_PATH = "./faiss_index_advanced"
		self.VECTOR_DB_BASE_PATH = "./vector_stores" 

settings = Settings()
config = settings  # config.py와 config 객체 호환을 위해 추가