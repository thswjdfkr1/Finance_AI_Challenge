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

3. Split        
     
   RecursiveCharacterTextSplitter를 사용하여 문서를 잘게 분할함.
   이 방법은 문서를 일정 길이로 자르고, 각 조각을 개별적으로 다룰 수 있게 해주며, 이후 모델 학습에 최적화된 텍스트를 제공함. 이를 통해 금융 및 금융 보 관련 문서들을 더 잘게 나누어 처리할 수 있음

4. QADataset셋 생성     
     
   생성된 chunk를 모두 합쳐 'skt/A.X-4.0-Light' 오픈 모델을 활용하여 QADataset을 생성
   
# RAG를 위한 PDF 문서 전처리 과정:   
1. Load           
      #### 과제에 적합한 PDFReader 선택   

      PDF 문서 로딩을 위해 PyPdfReader를 사용하여 문서를 처리합니다. 이때, 불필요한 요소들이 포함된 문서도 존재하므로 이를 처리하는 과정이 중요

2. 문서 정리 및 클린징       
 
      PDF 문서에는 검색에 불필요한 사진, 특수 문자, 공백, 줄바꿈, 부록 등의 불필요한 요소들이 포함될 수 있음

3. Split        
     
   RecursiveCharacterTextSplitter를 사용하여 문서를 잘게 분할함.
   이 방법은 문서를 일정 길이로 자르고, 각 조각을 개별적으로 다룰 수 있게 해주며, 이후 모델 학습에 최적화된 텍스트를 제공함. 이를 통해 금융 및 금융 보 관련 문서들을 더 잘게 나누어 처리할 수 있음

4. Embed    
텍스트 임베딩에는 SentenceTransformer("sentence-transformers/all-mpnet-base-v2") 모델을 사용   
이 모델은 빠르고 효율적으로 텍스트를 벡터 형태로 변환하여 의미 기반 검색에 적합한 표현을 제공함
이 과정을 통해 각 문서가 의미적으로 잘 표현된 벡터로 변환되어, 검색 및 후속 처리에서 높은 성능을 발휘함.   

5. Store
문서를 하이브리드 검색 방식을 사용하기 위한 형식으로 저장      

### BM25 색인 저장

BM25 Retriever를 활용하여 키워드 기반 색인을 생성   
이를 통해 특정 키워드가 포함된 문서를 빠르게 검색할 수 있음   

### FAISS 벡터 저장   

문서를 **SentenceTransformer("sentence-transformers/all-mpnet-base-v2")**를 사용하여 벡터로 변환   
변환된 벡터를 FAISS Retriever에 저장하여 의미적 유사성을 활용한 검색이 가능하도록 합니다. 임계치(Threshold) 기반 문서 필터링   

BM25 스코어 + FAISS 유사도 점수를 결합하여 특정 임계치(Threshold) 이상인 문서만 저장   
이 과정을 통해 불필요한 문서를 걸러내고, 문제와 유의미한 문서만 보관   
이러한 과정을 통해 문서가 검색 시스템에 최적화된 상태로 저장되며, 이후 금융 문제 해결 과정에서 신속하고 정확한 검색이 가능해짐.

# 추론

## LLM 파인튜닝   
1. 모델 및 토크나이저 설정   
'''
      model_name = 'LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct'   
      model = AutoModelForCausalLM.from_pretrained(    
          model_name,    
          device_map={"":0},    
          trust_remote_code=True,    
          )     
 
      tokenizer = AutoTokenizer.from_pretrained(model_name)    
'''
