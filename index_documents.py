import os
import fitz  # PyMuPDF
import google.generativeai as genai
import pinecone
from tqdm import tqdm
import uuid
import time
import re # reをインポート
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv('.env.development.local')

# --- 初期設定 ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not all([GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME]):
    raise ValueError("必要な環境変数が設定されていません。")

genai.configure(api_key=GEMINI_API_KEY)
embedding_model = "models/text-embedding-004"
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

# --- 定数設定 ---
DOCUMENTS_DIR = "documents"
CHUNK_SIZE = 800  # 段落ベースなので、少し短めに設定しても良い
CHUNK_OVERLAP = 50 # オーバーラップは少なめでOK
NAMESPACE = "company-docs"

def get_embedding(text: str, task_type="RETRIEVAL_DOCUMENT") -> list[float]:
    try:
        result = genai.embed_content(model=embedding_model, content=text, task_type=task_type)
        return result['embedding']
    except Exception as e:
        print(f"ベクトル化エラー: {e}")
        return []

def load_and_chunk_documents(directory: str) -> list[dict]:
    """意味のある単位（段落や文）で賢く分割する関数"""
    all_chunks = []
    print(f"'{directory}'フォルダ内のドキュメントを読み込みます...")
    if not os.path.exists(directory):
        print(f"エラー: '{directory}' フォルダが見つかりません。")
        return []

    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if not os.path.isfile(path): continue

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

            # 2つ以上の改行を単一の区切り文字に統一
            text = re.sub(r'\n\s*\n', '
', text)
            # 段落で分割
            paragraphs = text.split('
')
            
            for para in paragraphs:
                para = para.strip().replace('\n', ' ') # 段落内の不要な改行はスペースに置換
                if not para: continue
                
                # 段落が長すぎる場合は、さらに句点「。」で分割
                if len(para) > CHUNK_SIZE:
                    sentences = re.split(r'(。|．)', para)
                    current_chunk = ""
                    for i in range(0, len(sentences), 2):
                        sentence_part = "".join(sentences[i:i+2]).strip()
                        if not sentence_part: continue
                        
                        if len(current_chunk) + len(sentence_part) > CHUNK_SIZE:
                            all_chunks.append({"text": current_chunk, "source": filename})
                            current_chunk = sentence_part
                        else:
                            current_chunk += sentence_part
                    if current_chunk:
                        all_chunks.append({"text": current_chunk, "source": filename})
                else:
                    all_chunks.append({"text": para, "source": filename})

        except Exception as e:
            print(f"エラー: '{filename}' の処理中に問題が発生しました: {e}")
            
    return all_chunks

def main():
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
        batch = chunks[i:i + batch_size]
        vectors_to_upsert = []
        for chunk in batch:
            vector = get_embedding(chunk['text'])
            if not vector: continue
            vectors_to_upsert.append({
                "id": str(uuid.uuid4()),
                "values": vector,
                "metadata": {"source": chunk['source'], "text": chunk['text']}
            })
        
        if vectors_to_upsert:
            pinecone_index.upsert(vectors=vectors_to_upsert, namespace=NAMESPACE)
        time.sleep(1)

    print("\nすべてのドキュメントのインデックス作成が完了しました！")
    stats = pinecone_index.describe_index_stats()
    print(f"名前空間 '{NAMESPACE}' に {stats['namespaces'][NAMESPACE]['vector_count']} 件のベクトルが保存されています。")

if __name__ == "__main__":
    main()