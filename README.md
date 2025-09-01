# Finance_AI_Challenge
금융 및 금융보안 관련 데이터를 바탕으로 주어진 객관식/주관식 문제에 대한 답을 자동으로 생성하는 RAG 모델 개발

# 주제
금융보안 실무에 적합한 개발 방법론 및 AI 모델을 발굴하기 위해 FSKU 평가지표를 기반으로 AI 모델의 성능을 경쟁

# QADataset 생성과정 : 		 		   
1. Load           
      #### 과제에 적합한 PDFReader 선택

      PDF 문서 로딩을 위해 PyPdfReader를 사용하여 문서를 처리합니다. 이때, 불필요한 요소들이 포함된 문서도 존재하므로 이를 처리하는 과정이 중요

2. 문서 정리 및 클린징   
 
      PDF 문서에는 검색에 불필요한 사진, 특수 문자, 공백, 줄바꿈, 부록 등의 불필요한 요소들이 포함될 수 있음

4. Split    
      RecursiveCharacterTextSplitter를 사용하여 문서를 잘게 분할함.
      이 방법은 문서를 일정 길이로 자르고, 각 조각을 개별적으로 다룰 수 있게 해주며, 이후 모델 학습에 최적화된 텍스트를 제공함. 이를 통해 금융 및 금융 보 관련 문서들을 더 잘게 나누어 처리할 수 있음

5. QADataset셋 생성
      생성된 chunk를 모두 합쳐 'skt/A.X-4.0-Light' 오픈 모델을 활용하여 QADataset을 생성

# LLM 파인튜닝


모델 및 토크나이저 설정

model_name = "google/mt5-small"  
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)
