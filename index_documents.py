import os
import fitz  # PyMuPDF
import google.generativeai as genai
import pinecone
from tqdm import tqdm 
import uuid
import time
from dotenv import load_dotenv  # ★ この行を追加

load_dotenv('.env.development.local')

# -----------------------------------------------------------------
# 初期設定 (app.pyと同様の環境変数を読み込む)
# -----------------------------------------------------------------
# .envファイルから環境変数を読み込む (ローカル実行用)
# from dotenv import load_dotenv
# load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# クライアント初期化
genai.configure(api_key=GEMINI_API_KEY)
embedding_model = "models/text-embedding-004"
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

# -----------------------------------------------------------------
# 定数設定
# -----------------------------------------------------------------
DOCUMENTS_DIR = "documents" # 資料が入っているフォルダ名
CHUNK_SIZE = 1000    # テキストを分割する際のチャンクサイズ（文字数）
CHUNK_OVERLAP = 100  # チャンク間でオーバーラップさせる文字数
NAMESPACE = "company-docs" # ★ 会社資料用の名前空間を定義

# -----------------------------------------------------------------
# 関数定義
# -----------------------------------------------------------------
def get_embedding(text: str, task_type="RETRIEVAL_DOCUMENT") -> list[float]:
    """テキストをベクトルに変換する"""
    try:
        # ドキュメントのベクトル化時は task_type を指定するのが推奨
        result = genai.embed_content(
            model=embedding_model,
            content=text,
            task_type=task_type
        )
        return result['embedding']
    except Exception as e:
        print(f"ベクトル化でエラー: {e}")
        return []

def load_and_chunk_documents(directory: str) -> list[dict]:
    """指定されたディレクトリからドキュメントを読み込み、チャンクに分割する"""
    chunks = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if not os.path.isfile(path):
            continue

        text = ""
        if filename.endswith(".pdf"):
            with fitz.open(path) as doc:
                text = "".join(page.get_text() for page in doc)
        elif filename.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            continue # 対応外のファイルはスキップ

        # テキストをチャンクに分割
        for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_text = text[i:i + CHUNK_SIZE]
            chunks.append({
                "text": chunk_text,
                "source": filename
            })
    return chunks

# -----------------------------------------------------------------
# メイン処理
# -----------------------------------------------------------------
def main():
    print("ドキュメントの読み込みとチャンク分割を開始します...")
    chunks = load_and_chunk_documents(DOCUMENTS_DIR)
    if not chunks:
        print("処理対象のドキュメントが見つかりませんでした。")
        return

    print(f"合計 {len(chunks)} 個のチャンクが作成されました。")
    print("ベクトル化とPineconeへの保存を開始します...")

    batch_size = 100 # 一度にアップロードするベクトル数
    vectors_to_upsert = []

    for chunk in tqdm(chunks):
        vector = get_embedding(chunk["text"])
        if not vector:
            continue
        
        vectors_to_upsert.append({
            "id": str(uuid.uuid4()),
            "values": vector,
            "metadata": {
                "source": chunk["source"],
                "text": chunk["text"]
            }
        })

        # バッチサイズに達したらアップロード
        if len(vectors_to_upsert) >= batch_size:
            pinecone_index.upsert(vectors=vectors_to_upsert, namespace=NAMESPACE)
            vectors_to_upsert = []
            print(f"  -> {batch_size}件のベクトルをアップロードしました。")
            time.sleep(1) # APIのレートリミットを避けるため

    # 残りのベクトルをアップロード
    if vectors_to_upsert:
        pinecone_index.upsert(vectors=vectors_to_upsert, namespace=NAMESPACE)
        print(f"  -> 残りの{len(vectors_to_upsert)}件のベクトルをアップロードしました。")

    print("\nすべてのドキュメントのインデックス作成が完了しました！")
    print(f"名前空間 '{NAMESPACE}' にデータが保存されています。")
    print(f"Pinecone Indexの現在のベクトル数: {pinecone_index.describe_index_stats()['total_vector_count']}")

if __name__ == "__main__":
    main()
