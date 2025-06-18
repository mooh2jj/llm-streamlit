import streamlit as st
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# .env 파일 로드
load_dotenv()

st.set_page_config(page_title="법률가 챗봇", page_icon=":books:", layout="wide")

st.title("📚 법률가 챗봇")
st.caption("법률 관련 질문에 답변합니다")

# OpenAI API 키 로드
openai_api_key = os.getenv("OPENAI_API_KEY")

# 사이드바
with st.sidebar:
    st.header("설정")
    if not openai_api_key:
        st.error("⚠️ .env 파일에 OPENAI_API_KEY를 설정해주세요!")
        st.info("📝 .env 파일 예시:\nOPENAI_API_KEY=your_api_key_here")
    else:
        st.success("✅ OpenAI API 키가 설정되었습니다.")

# RAG 시스템 초기화
@st.cache_resource
def initialize_rag_system():
    """RAG 시스템을 초기화합니다."""
    try:
        # PDF 문서 로딩
        loader = PyPDFLoader("tax.pdf")
        documents = loader.load()
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # 임베딩 모델 설정 (한국어 지원)
        embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sbert-multitask"
        )
        
        # ChromaDB 벡터 저장소 생성
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        return vectorstore
    except Exception as e:
        st.error(f"RAG 시스템 초기화 중 오류 발생: {str(e)}")
        return None

# RAG 질의응답 함수
def get_rag_response(question, vectorstore, api_key):
    """RAG를 사용하여 질문에 답변합니다."""
    try:
        # LLM 설정
        llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name="gpt-4o-mini",
            temperature=0
        )
        
        # 프롬프트 템플릿 설정
        prompt_template = """
        당신은 세무 전문가입니다. 제공된 문서를 바탕으로 정확하고 유용한 답변을 제공해주세요.
        문서에 없는 내용은 추측하지 말고, 모를 경우 솔직히 말씀해주세요.
        
        문서 내용:
        {context}
        
        질문: {question}
        
        답변:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # RetrievalQA 체인 생성
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}), # 검색 결과 개수
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        # 질의응답 실행
        response = qa_chain.run(question)
        return response
        
    except Exception as e:
        return f"답변 생성 중 오류가 발생했습니다: {str(e)}"

# 세션 상태 초기화
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'vectorstore' not in st.session_state:
    with st.spinner("PDF 문서를 임베딩하고 있습니다..."):
        st.session_state.vectorstore = initialize_rag_system()

# 기존 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 사용자 입력 처리
if user_question := st.chat_input(placeholder="세무 관련 질문을 해주세요..."):
    # API 키 확인
    if not openai_api_key:
        st.error("OpenAI API 키를 먼저 설정해주세요! .env 파일을 확인해주세요.")
    elif st.session_state.vectorstore is None:
        st.error("RAG 시스템 초기화에 실패했습니다.")
    else:
        # 사용자 질문 표시
        with st.chat_message("user"):
            st.write(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        # AI 답변 생성 및 표시
        with st.chat_message("assistant"):
            with st.spinner("답변을 생성하고 있습니다..."):
                ai_response = get_rag_response(
                    user_question, 
                    st.session_state.vectorstore, 
                    openai_api_key
                )
            st.write(ai_response)
        st.session_state.messages.append({"role": "assistant", "content": ai_response})

# 디버깅용 (개발 환경에서만)
if st.checkbox("디버그 모드"):
    st.write("세션 상태:", st.session_state.messages)