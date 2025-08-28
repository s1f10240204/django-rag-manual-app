import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ファイルの先頭に、以下のライブラリを追加でインポートします
import fitz  # PyMuPDF
import base64
from openai import OpenAI

# ... 既存のimport文 ...
# ... 既存の create_vectorstore_from_pdf と ask_question 関数 ...


# ▼▼▼ 以下の新しい関数をファイルの末尾に追加 ▼▼▼

def analyze_image_with_vision(image_bytes: bytes) -> str:
    """
    画像データをOpenAIのVisionモデルに渡し、説明文を生成する関数
    """
    client = OpenAI() # APIキーは.envファイルから自動で読み込まれる
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "これは製品マニュアルに含まれる図やイラストです。この画像が何を示しているか、誰が見てもわかるように詳細に説明してください。専門用語や部品名があればそれも使って説明してください。"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        description = response.choices[0].message.content
        print(f"--- Vision API: Image description generated. ---")
        return description
    except Exception as e:
        print(f"--- Vision API Error: {e} ---")
        return ""


def create_vectorstore_from_vision_pdf(pdf_path: str, vectorstore_dir: str):
    """
    VisionモデルでPDF内の画像を解析し、テキストと統合してベクトルストアを作成する関数。
    """
    print(f"--- Starting Vision-Enhanced PDF Processing for: {pdf_path} ---")
    
    # 1. PyMuPDFでPDFからテキストと画像を抽出
    doc = fitz.open(pdf_path)
    all_content = []
    
    for page_num, page in enumerate(doc):
        # ページのテキストを追加
        all_content.append(f"[ページ {page_num + 1} のテキスト]\n{page.get_text()}")
        
        # ページの画像を取得
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # 2. 画像の説明文をAIが生成
            print(f"--- Analyzing image {img_index + 1} on page {page_num + 1} ---")
            image_description = analyze_image_with_vision(image_bytes)
            
            if image_description:
                all_content.append(f"[ページ {page_num + 1} の図 {img_index + 1} の説明]\n{image_description}")

    doc.close()
    
    # 3. 統合したテキストデータを作成
    full_text_content = "\n\n".join(all_content)
    
    if not full_text_content.strip():
        print("--- Warning: No text content extracted from PDF. ---")
        return False
        
    # 4. ベクトル化 (既存のロジックを再利用)
    try:
        print("--- Splitting combined text into chunks... ---")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        # テキストを直接渡すため、Documentオブジェクトに変換
        from langchain_core.documents import Document
        docs = [Document(page_content=full_text_content)]
        texts = text_splitter.split_documents(docs)
        
        if not texts:
            print("--- Warning: Document could not be split into texts. ---")
            return False
            
        print(f"--- Split into {len(texts)} chunks. Starting embedding... ---")
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(texts, embeddings)
        
        os.makedirs(vectorstore_dir, exist_ok=True)
        db.save_local(vectorstore_dir)
        print(f"--- Vision-Enhanced vector store saved to: {vectorstore_dir} ---")
        return True
    except Exception as e:
        print(f"--- An error occurred during vector store creation: {e} ---")
        return False

def create_vectorstore_from_pdf(pdf_path: str, vectorstore_dir: str):
    """
    Unstructuredを使用して単一のPDFファイルからFAISSベクトルストアを作成する関数。
    成功した場合はTrue、失敗した場合はFalseを返す。
    """
    try:
        print(f"--- Loading PDF from: {pdf_path} ---")
        # ローダーをUnstructuredPDFLoaderに変更
        loader = UnstructuredPDFLoader(pdf_path, mode="elements")
        documents = loader.load()

        if not documents:
            print("--- Warning: No documents were loaded from the PDF. It might be empty or unreadable. ---")
            return False

        print(f"--- Loaded {len(documents)} document elements. ---")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        texts = text_splitter.split_documents(documents)
        
        if not texts:
            print("--- Warning: Document could not be split into texts. ---")
            return False
            
        print(f"--- Split into {len(texts)} chunks. ---")

        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(texts, embeddings)
        
        os.makedirs(vectorstore_dir, exist_ok=True)
        db.save_local(vectorstore_dir)
        print(f"--- Vector store saved to: {vectorstore_dir} ---")
        return True
    except Exception as e:
        print(f"--- An error occurred during vector store creation: {e} ---")
        return False

def ask_question(query: str, vectorstore_path: str) -> str:
    """
    指定されたベクトルストアを使用して、ユーザーの質問に回答を生成する関数。
    """
    if not os.path.exists(vectorstore_path):
        return "ベクトルストアが見つかりません。マニュアルの読み込みからやり直してください。"

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

    retriever = vectorstore.as_retriever(search_kwargs={'k': 4})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    prompt_template = """
    あなたは製品マニュアルの内容に精通したアシスタントです。
    提供された「コンテキスト情報」だけを元にして、ユーザーの「質問」に日本語で回答してください。
    コンテキスト情報に答えが見つからない場合は、正直に「マニュアルには関連する記載がありませんでした。」と回答してください。
    自身の知識やコンテキスト以外の情報を使って回答してはいけません。

    コンテキスト情報:
    {context}

    質問:
    {question}

    回答:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=False
    )

    result = qa_chain.invoke({"query": query})
    return result['result']