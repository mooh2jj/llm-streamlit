# Tax PDF RAG Assistant

세무 관련 PDF 문서를 기반으로 질문에 답변하는 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 주요 기능

- 📚 PDF 문서 자동 임베딩 (ChromaDB 사용)
- 🤖 LangChain을 활용한 자연어 질의응답
- 🔍 한국어 임베딩 모델 지원
- 💬 Streamlit 채팅 인터페이스

## 설치 및 실행

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. OpenAI API 키 준비

- OpenAI 계정에서 API 키를 발급받아주세요
- 앱 실행 후 사이드바에서 API 키를 입력합니다

### 3. 앱 실행

```bash
streamlit run test.py
```

## 사용법

1. 앱이 시작되면 자동으로 `tax.pdf` 파일을 임베딩합니다
2. 사이드바에서 OpenAI API 키를 입력합니다
3. 채팅창에서 세무 관련 질문을 입력합니다
4. AI가 PDF 문서 내용을 바탕으로 답변합니다

## 기술 스택

- **Streamlit**: 웹 인터페이스
- **LangChain**: RAG 파이프라인
- **ChromaDB**: 벡터 데이터베이스
- **HuggingFace**: 한국어 임베딩 모델
- **OpenAI GPT**: 언어 모델

## 주의사항

- OpenAI API 사용에 따른 비용이 발생할 수 있습니다
- 첫 실행 시 임베딩 생성으로 시간이 소요될 수 있습니다
- PDF 파일은 프로젝트 루트 디렉토리에 `tax.pdf`로 저장되어야 합니다
