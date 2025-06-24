import os
# protobuf 호환성을 위한 환경변수 설정 (다른 모든 import보다 먼저!)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("<<<<< sqlite3 patched with pysqlite3 >>>>>")
except ImportError:
    # pysqlite3가 없으면 기본 sqlite3 사용
    print("<<<<< using default sqlite3 >>>>>")

print("<<<<< app.app.py IS BEING LOADED (sqlite3 patched with pysqlite3) >>>>>") # 패치 내용 명시

import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from typing import Dict, List, Any
import time
import uuid
import subprocess

from streamlit_extras.buy_me_a_coffee import button

button(username="ehtjd33e", floating=True, width=221)

# .env 파일 로드
load_dotenv()

st.set_page_config(page_title="법률가 챗봇", page_icon=":books:", layout="wide")

st.title("📚 법률가 챗봇")
st.caption("PDF 문서를 업로드하여 법률 관련 질문에 답변받으세요")

# OpenAI API 키 로드
openai_api_key = os.getenv("OPENAI_API_KEY")

# 스트리밍을 위한 콜백 핸들러
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▋")  # 커서 효과
        time.sleep(0.01)  # 자연스러운 타이핑 효과

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        self.container.markdown(self.text)  # 최종 텍스트 (커서 제거)

# 사이드바
with st.sidebar:
    st.header("📁 파일 업로드")
    
    # PDF 파일 업로드
    uploaded_file = st.file_uploader(
        "PDF 파일을 업로드하세요",
        type=["pdf"],
        help="법률 문서나 관련 자료를 PDF 형태로 업로드하세요"
    )
    
    st.divider()
    
    st.header("⚙️ 설정")
    if not openai_api_key:
        st.error("⚠️ .env 파일에 OPENAI_API_KEY를 설정해주세요!")
        st.info("📝 .env 파일 예시:\nOPENAI_API_KEY=your_api_key_here")
    else:
        st.success("✅ OpenAI API 키가 설정되었습니다.")
    
    if uploaded_file:
        st.success(f"✅ 파일 업로드됨: {uploaded_file.name}")
        st.info(f"📄 파일 크기: {uploaded_file.size / 1024:.1f} KB")

# 안전한 ChromaDB 삭제 함수 (개선된 버전)
def safe_delete_chromadb(max_retries=3, delay=1):
    """ChromaDB 폴더를 안전하게 삭제합니다."""
    import shutil
    import time
    import gc
    
    # 먼저 vectorstore 객체 해제 및 가비지 컬렉션 강제 실행
    if 'vectorstore' in st.session_state:
        st.session_state.vectorstore = None
    gc.collect()  # 가비지 컬렉션 강제 실행
    
    chroma_dir = "./chroma_db"
    if not os.path.exists(chroma_dir):
        return True
    
    # 먼저 이름 변경 시도 (Windows에서 더 안전함)
    temp_name = f"./chroma_db_deleted_{uuid.uuid4().hex[:8]}"
    
    try:
        # 폴더 이름 변경 (일반적으로 삭제보다 빠름)
        os.rename(chroma_dir, temp_name)
        print(f"ChromaDB 디렉토리 이름 변경: {chroma_dir} -> {temp_name}")
        
        # 이름 변경 후 삭제 시도
        for attempt in range(max_retries):
            try:
                time.sleep(delay)
                shutil.rmtree(temp_name)
                print(f"ChromaDB 디렉토리 삭제 성공: {temp_name}")
                return True
            except Exception as e:
                print(f"삭제 시도 {attempt + 1}/{max_retries} 실패: {str(e)}")
                if attempt < max_retries - 1:
                    delay *= 2
        
        # 삭제 실패 시 나중에 정리하도록 남겨둠
        print(f"ChromaDB 디렉토리 삭제 실패, 나중에 정리됨: {temp_name}")
        return True  # 이름 변경은 성공했으므로 True 반환
        
    except Exception as e:
        print(f"ChromaDB 디렉토리 이름 변경 실패: {str(e)}")
        return False

# ChromaDB 정리 함수 (개선된 버전)
def cleanup_chromadb():
    """기존 ChromaDB 폴더들을 정리합니다."""
    import glob
    import shutil
    import time
    
    # 모든 ChromaDB 관련 폴더 찾기 (삭제 예정 폴더 포함)
    chroma_dirs = glob.glob("./chroma_db*")
    
    for dir_path in chroma_dirs:
        try:
            # 약간의 지연 후 삭제
            time.sleep(0.5)
            shutil.rmtree(dir_path)
            print(f"정리됨: {dir_path}")
        except Exception as e:
            print(f"정리 중 오류 ({dir_path}): {str(e)}")
            # 삭제 실패한 폴더는 다음에 다시 시도

# 강제 ChromaDB 정리 함수 (Windows 전용)
def force_cleanup_chromadb():
    """Windows에서 ChromaDB를 강제로 정리합니다."""
    import subprocess
    import glob
    import time
    
    chroma_dirs = glob.glob("./chroma_db*")
    
    for dir_path in chroma_dirs:
        try:
            # Windows의 경우 핸들을 가진 프로세스 확인 및 정리 시도
            if os.name == 'nt':  # Windows
                try:
                    # robocopy를 사용한 빈 폴더로 덮어쓰기 (Windows 전용 트릭)
                    empty_dir = "./temp_empty_dir"
                    os.makedirs(empty_dir, exist_ok=True)
                    
                    # robocopy로 빈 폴더 내용을 대상 폴더에 미러링 (효과적으로 삭제)
                    subprocess.run([
                        "robocopy", empty_dir, dir_path, "/MIR", "/NFL", "/NDL", "/NJH", "/NJS"
                    ], capture_output=True, check=False)
                    
                    # 빈 폴더들 삭제
                    os.rmdir(empty_dir)
                    os.rmdir(dir_path)
                    print(f"강제 정리 성공: {dir_path}")
                    
                except Exception as e:
                    print(f"강제 정리 실패 ({dir_path}): {str(e)}")
            else:
                # Unix 계열 시스템에서는 일반 삭제
                import shutil
                shutil.rmtree(dir_path)
                print(f"정리됨: {dir_path}")
                
        except Exception as e:
            print(f"정리 중 오류 ({dir_path}): {str(e)}")

# 앱 시작 시 ChromaDB 정리 (세션 상태 초기화 전에)
if 'app_initialized' not in st.session_state:
    # 일반 정리 시도
    cleanup_chromadb()
    
    # Windows에서 여전히 남아있는 폴더가 있다면 강제 정리 시도
    if os.name == 'nt':  # Windows
        import glob
        remaining_dirs = glob.glob("./chroma_db*")
        if remaining_dirs:
            print("Windows에서 강제 정리 시도...")
            force_cleanup_chromadb()
    
    st.session_state.app_initialized = True

# RAG 시스템 초기화 (더욱 개선된 버전)
@st.cache_resource
def initialize_rag_system(file_path):
    """RAG 시스템을 초기화합니다."""
    try:
        # PDF 문서 로딩
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # 임베딩 모델 설정 (한국어 지원)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        # 고유한 ChromaDB 디렉토리 사용 (충돌 방지)
        import time
        import uuid
        
        # 타임스탬프와 랜덤 ID로 고유한 디렉토리명 생성
        chroma_dir = f"./chroma_db_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # 기존 chroma_db 폴더가 있다면 이름 변경으로 처리
        old_chroma_dir = "./chroma_db"
        if os.path.exists(old_chroma_dir):
            try:
                # 이름 변경 (삭제보다 안전)
                temp_name = f"./chroma_db_old_{uuid.uuid4().hex[:8]}"
                os.rename(old_chroma_dir, temp_name)
                print(f"기존 ChromaDB 디렉토리 이름 변경: {old_chroma_dir} -> {temp_name}")
            except Exception as e:
                print(f"기존 ChromaDB 디렉토리 이름 변경 실패: {str(e)}")

        # 새로운 ChromaDB 벡터 저장소 생성
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=chroma_dir
        )
        
        return vectorstore
    except Exception as e:
        st.error(f"RAG 시스템 초기화 중 오류 발생: {str(e)}")
        return None

# 업로드된 파일을 임시 파일로 저장
def save_uploaded_file(uploaded_file):
    """업로드된 파일을 임시 파일로 저장하고 경로를 반환합니다."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"파일 저장 중 오류 발생: {str(e)}")
        return None

# RAG 질의응답 함수 (스트리밍 버전)
def get_rag_response_streaming(question, vectorstore, api_key, container):
    """RAG를 사용하여 질문에 답변합니다 (스트리밍 방식)."""
    try:
        # 스트리밍 콜백 핸들러 생성
        callback_handler = StreamlitCallbackHandler(container)
        
        # LLM 설정 (스트리밍 활성화)
        llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name="gpt-4o-mini",
            temperature=0,
            streaming=True,
            callbacks=[callback_handler]
        )
        
        # 프롬프트 템플릿 설정
        prompt_template = """
        당신은 문서 분석 전문가입니다. 제공된 문서를 바탕으로 정확하고 유용한 답변을 제공해주세요.
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
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        # 질의응답 실행
        response = qa_chain.invoke({"query": question})
        return callback_handler.text  # 스트리밍된 전체 텍스트 반환
        
    except Exception as e:
        error_msg = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
        container.error(error_msg)
        return error_msg

# 세션 상태 초기화
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if 'current_file' not in st.session_state:
    st.session_state.current_file = None

# 파일 업로드 처리
if uploaded_file:
    # 새 파일이 업로드되었는지 확인
    if st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.vectorstore = None  # 기존 벡터 스토어 초기화
        st.session_state.messages = []  # 대화 히스토리 초기화
        
        # 파일 저장 및 RAG 시스템 초기화
        with st.spinner("📄 PDF 문서를 분석하고 임베딩하고 있습니다... 잠시만 기다려주세요!"):
            temp_file_path = save_uploaded_file(uploaded_file)
            if temp_file_path:
                st.session_state.vectorstore = initialize_rag_system(temp_file_path)
                # 임시 파일 정리
                os.unlink(temp_file_path)
                
                if st.session_state.vectorstore:
                    st.success("✅ 문서가 성공적으로 분석되었습니다!")
                else:
                    st.error("❌ 문서 분석에 실패했습니다.")
else:
    # 파일이 삭제된 경우 처리
    if st.session_state.current_file is not None:
        # 안전한 ChromaDB 폴더 삭제
        with st.spinner("🗑️ 기존 문서 데이터를 정리하고 있습니다..."):
            success = safe_delete_chromadb()
            if success:
                st.info("🗑️ 기존 문서 데이터가 정리되었습니다.")
            else:
                st.warning("⚠️ 데이터 정리가 지연되었습니다. 다음 앱 재시작 시 자동으로 정리됩니다.")
        
        # 세션 상태 초기화
        st.session_state.current_file = None
        st.session_state.vectorstore = None
        st.session_state.messages = []

# 파일이 업로드되지 않았을 때 안내
if not uploaded_file:
    st.info("📁 먼저 사이드바에서 PDF 파일을 업로드해주세요!")
    st.markdown("""
    ### 📋 사용 방법
    1. **파일 업로드**: 사이드바에서 PDF 파일 업로드
    2. **문서 분석**: 업로드된 문서가 자동으로 분석됩니다
    3. **질문하기**: 문서 내용에 대해 질문해보세요
    
    ### 📄 지원 파일 형식
    - PDF 파일만 지원됩니다
    - 텍스트가 포함된 PDF 파일이어야 합니다
    """)
else:
    # 기존 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 처리
    if user_question := st.chat_input(placeholder="업로드된 문서에 대해 질문해주세요... 💬"):
        # API 키 확인
        if not openai_api_key:
            st.error("OpenAI API 키를 먼저 설정해주세요! .env 파일을 확인해주세요.")
        elif st.session_state.vectorstore is None:
            st.error("문서 분석이 완료되지 않았습니다. 잠시 후 다시 시도해주세요.")
        else:
            # 사용자 질문 표시
            with st.chat_message("user"):
                st.markdown(user_question)
            st.session_state.messages.append({"role": "user", "content": user_question})

            # AI 답변 생성 및 스트리밍 표시
            with st.chat_message("assistant"):
                # 스트리밍을 위한 빈 컨테이너 생성
                message_container = st.empty()
                
                # 스트리밍으로 답변 생성
                ai_response = get_rag_response_streaming(
                    user_question, 
                    st.session_state.vectorstore, 
                    openai_api_key,
                    message_container
                )
                
            st.session_state.messages.append({"role": "assistant", "content": ai_response})

# 하단에 사용 팁 추가
if uploaded_file:
    st.divider()
    with st.expander("💡 사용 팁"):
        st.markdown(f"""
        **현재 분석 중인 파일: {uploaded_file.name}**
        
        **이 챗봇은 어떻게 작동하나요?**
        - 📄 업로드된 PDF 문서의 내용을 기반으로 답변합니다
        - 🔍 관련 정보를 검색하여 정확한 답변을 제공합니다
        - ⚡ 실시간 스트리밍으로 답변이 생성됩니다
        
        **더 나은 답변을 위한 팁:**
        - 구체적이고 명확한 질문을 해주세요
        - 문서의 특정 부분에 대해 질문해보세요
        - 문서에 없는 내용은 솔직히 "모른다"고 답변할 수 있습니다
        """)

    # 디버깅용 (개발 환경에서만)
    if st.checkbox("🔧 디버그 모드"):
        st.json({"메시지 개수": len(st.session_state.messages)})
        with st.expander("전체 대화 내역"):
            st.json(st.session_state.messages)