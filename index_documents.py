import os
import fitz  # PyMuPDF
import google.generativeai as genai
import pinecone
from tqdm import tqdm
import uuid
import time
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv('.env.development.local')

# -----------------------------------------------------------------
# 初期設定
# -----------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not all([GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME]):
    raise ValueError("必要な環境変数が設定されていません。")

# クライアント初期化
genai.configure(api_key=GEMINI_API_KEY)
embedding_model = "models/text-embedding-004"
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

# -----------------------------------------------------------------
# 定数設定
# -----------------------------------------------------------------
DOCUMENTS_DIR = "documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
NAMESPACE = "company-docs"

# -----------------------------------------------------------------
# 関数定義
# -----------------------------------------------------------------
def get_embedding(text: str, task_type="RETRIEVAL_DOCUMENT") -> list[float]:
    """テキストをベクトルに変換する（文書登録用）"""
    try:
        result = genai.embed_content(
            model=embedding_model,
            content=text,
            task_type=task_type
        )
        return result['embedding']
    except Exception as e:
        print(f"ベクトル化エラー: {e}")
        return []

def load_and_chunk_documents(directory: str) -> list[dict]:
    """ディレクトリからドキュメントを読み込み、チャンクに分割する"""
    chunks = []
    print(f"'{directory}'フォルダ内のドキュメントを読み込みます...")
    if not os.path.exists(directory):
        print(f"エラー: '{directory}' フォルダが見つかりません。")
        return []

    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if not os.path.isfile(path):
            continue

        text = ""
        try:
            if filename.endswith(".pdf"):
                with fitz.open(path) as doc:
                    text = "".join(page.get_text() for page in doc).strip()
            elif filename.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            else:
                continue

            if not text:
                print(f"警告: '{filename}' からテキストを抽出できませんでした。")
                continue

            # テキストをチャンクに分割
            for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk_text = text[i:i + CHUNK_SIZE]
                chunks.append({
                    "text": chunk_text,
                    "source": filename
                })
        except Exception as e:
            print(f"エラー: '{filename}' の処理中に問題が発生しました: {e}")

    return chunks

# -----------------------------------------------------------------
# メイン処理
# -----------------------------------------------------------------
def main():
    # 既存のデータを名前空間から削除
    try:
        print(f"既存の名前空間 '{NAMESPACE}' のデータをクリアします...")
        pinecone_index.delete(delete_all=True, namespace=NAMESPACE)
        print("クリア完了。")
    except Exception as e:
        print(f"名前空間のクリア中にエラーが発生しました（初回実行の場合は問題ありません）: {e}")


    chunks = load_and_chunk_documents(DOCUMENTS_DIR)
    if not chunks:
        print("処理対象のドキュメントが見つかりませんでした。")
        return

    print(f"合計 {len(chunks)} 個のチャンクが作成されました。")
    print("ベクトル化とPineconeへの保存を開始します...")

    batch_size = 100
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch_chunks = chunks[i:i + batch_size]
        vectors_to_upsert = []
        for chunk in batch_chunks:
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

        if vectors_to_upsert:
            pinecone_index.upsert(vectors=vectors_to_upsert, namespace=NAMESPACE)
        time.sleep(1)

    print("\nすべてのドキュメントのインデックス作成が完了しました！")
    print(f"名前空間 '{NAMESPACE}' にデータが保存されています。")
    print(f"Pinecone Indexの現在のベクトル数: {pinecone_index.describe_index_stats()['total_vector_count']}")

if __name__ == "__main__":
    main()