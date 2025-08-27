import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

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