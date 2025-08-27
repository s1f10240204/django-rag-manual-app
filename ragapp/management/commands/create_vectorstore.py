import os
from django.core.management.base import BaseCommand
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

# ベクトルストアとマニュアルのパスを定義
VECTORSTORE_PATH = "faiss_index"
MANUALS_PATH = "manuals"

class Command(BaseCommand):
    help = '製品マニュアルを読み込み、ベクトルストアを構築します。'

    def handle(self, *args, **options):
        self.stdout.write("ベクトルストアの構築を開始します...")

        # 1. マニュアルPDFの読み込み
        if not os.path.exists(MANUALS_PATH) or not os.listdir(MANUALS_PATH):
            self.stdout.write(self.style.ERROR(f"'{MANUALS_PATH}' ディレクトリが見つからないか、空です。PDFファイルを配置してください。"))
            return
            
        loader = PyPDFDirectoryLoader(MANUALS_PATH)
        documents = loader.load()
        self.stdout.write(f"{len(documents)}個のドキュメントを読み込みました。")

        # 2. テキストの分割（チャンキング）
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        self.stdout.write(f"ドキュメントを{len(texts)}個のチャンクに分割しました。")

        # 3. テキストのベクトル化とベクトルストアの作成
        # OpenAIのEmbeddingモデルを使用
        embeddings = OpenAIEmbeddings()
        
        # FAISSベクトルストアを作成し、チャンクとEmbeddingを保存
        db = FAISS.from_documents(texts, embeddings)
        
        # 作成したベクトルストアをローカルに保存
        db.save_local(VECTORSTORE_PATH)

        self.stdout.write(self.style.SUCCESS(f"ベクトルストアを '{VECTORSTORE_PATH}' に正常に保存しました。"))