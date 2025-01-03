# main.py
import os
import uuid
import requests
import convertapi
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import aiohttp
from typing import Optional
import tiktoken
import uvicorn

# Swagger 헤더 설정
SWAGGER_HEADERS = {
    "title": "LINKBRICKS HORIZON-AI PDF TO TEXT API ENGINE",
    "version": "100.100.100",
    "description": "## PDF 텍스트 추출 및 요약 엔진 \n - API Swagger \n - PDF to TEXT \n - Text Summarization with GPT-4o",
    "contact": {
        "name": "Linkbricks Horizon AI",
        "url": "https://www.linkbricks.com",
        "email": "contact@linkbricks.com",
        "license_info": {
            "name": "GNU GPL 3.0",
            "url": "https://www.gnu.org/licenses/gpl-3.0.html",
        },
    },
}

app = FastAPI(**SWAGGER_HEADERS)

# 인증키
REQUIRED_AUTH_KEY = "linkbricks-saxoji-benedict-ji-01034726435!@#$%231%$#@%"

# 파일 저장 디렉토리 설정
UPLOAD_DIR = "uploaded_files"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# API 요청 모델
class PDFExtractRequest(BaseModel):
    convertapi_key: str
    pdf_url: str
    auth_key: str
    api_key: str  # OpenAI API Key

def is_pdf_url(url: str) -> bool:
    """URL이 PDF 파일인지 확인하는 함수"""
    # 특정 PDF 호스팅 사이트들의 URL 패턴 확인
    pdf_patterns = [
        'arxiv.org/pdf/',
        '.pdf',
        'pdf.sciencedirectassets.com',
        'downloads.hindawi.com',
        'download.springer.com',
        'ieeexplore.ieee.org/stamp/stamp.jsp'
    ]
    
    url_lower = url.lower()
    if any(pattern in url_lower for pattern in pdf_patterns):
        return True
    
    # PDF 파일인지 헤더로 확인
    try:
        response = requests.head(url, allow_redirects=True)
        content_type = response.headers.get('content-type', '').lower()
        if 'application/pdf' in content_type:
            return True
            
        # GET 요청으로 실제 파일 확인 (일부 서버는 HEAD 요청에 정확한 content-type을 반환하지 않음)
        if not any(pattern in url_lower for pattern in ['arxiv.org']):  # arxiv는 이미 확인됨
            response = requests.get(url, stream=True)
            content_type = response.headers.get('content-type', '').lower()
            return 'application/pdf' in content_type
            
        return True
    except Exception as e:
        print(f"URL 확인 중 오류 발생: {str(e)}")
        return False

def split_text_into_chunks(text: str, max_tokens: int = 4000) -> list:
    """텍스트를 토큰 수에 맞게 분할하는 함수"""
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode(text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for token in tokens:
        current_chunk.append(token)
        current_length += 1
        
        if current_length >= max_tokens:
            chunks.append(encoding.decode(current_chunk))
            current_chunk = []
            current_length = 0
    
    if current_chunk:
        chunks.append(encoding.decode(current_chunk))
    
    return chunks

async def summarize_text_chunk(api_key: str, text: str) -> str:
    """텍스트 청크를 요약하는 함수"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "당신은 학술 논문과 문서를 요약하는 전문가입니다."},
                    {"role": "user", "content": f"다음 텍스트를 요약해주세요:\n\n{text}"}
                ],
                "max_tokens": 1500
            }
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise HTTPException(status_code=500, detail=f"OpenAI API 오류: {error_text}")
            
            result = await response.json()
            return result['choices'][0]['message']['content']

@app.post("/extract-and-summarize/")
async def extract_and_summarize(request: PDFExtractRequest):
    """PDF 텍스트 추출 및 요약 API 엔드포인트"""
    saved_files = []  # 생성된 파일들의 경로를 저장할 리스트

    try:
        # 인증키 검증
        if request.auth_key != REQUIRED_AUTH_KEY:
            raise HTTPException(status_code=403, detail="인증키가 유효하지 않습니다.")
        
        # PDF URL 검증
        if not is_pdf_url(request.pdf_url):
            raise HTTPException(status_code=400, detail="유효한 PDF URL이 아닙니다.")
        
        # ConvertAPI 설정
        convertapi.api_credentials = request.convertapi_key
        
        # PDF를 텍스트로 변환
        result = convertapi.convert('txt', {
            'File': request.pdf_url
        }, from_format='pdf')
        
        # 결과 파일 저장 및 경로 가져오기
        saved_path = result.save_files(UPLOAD_DIR)[0]
        saved_files.append(saved_path)
        
        # 결과 파일 읽기
        text_content = ""
        with open(saved_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        
        # 텍스트를 청크로 분할
        text_chunks = split_text_into_chunks(text_content)
        
        # 각 청크 요약
        summaries = []
        for chunk in text_chunks:
            summary = await summarize_text_chunk(request.api_key, chunk)
            summaries.append(summary)
        
        # 최종 요약본 생성
        final_summary = "\n\n".join(summaries)
        
        return {
            "text_content": text_content,
            "summary": final_summary
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 생성된 모든 파일 삭제
        for file_path in saved_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"파일 삭제 완료: {file_path}")
            except Exception as e:
                print(f"파일 삭제 중 오류 발생: {file_path} - {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
