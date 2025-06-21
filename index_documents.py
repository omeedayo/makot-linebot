import os
import fitz  # PyMuPDF
import google.generativeai as genai
import pinecone
from tqdm import tqdm
import uuid
import time
import re
from dotenv import load_dotenv

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
CHUNK_SIZE = 1000  # チャンクの最大文字数
NAMESPACE = "company-docs"

def get_embedding(text: str, task_type="RETRIEVAL_DOCUMENT") -> list[float]:
    try:
        result = genai.embed_content(model=embedding_model, content=text, task_type=task_type)
        return result['embedding']
    except Exception as e:
        print(f"ベクトル化エラー: {e}")
        return []

def preprocess_text(text: str) -> str:
    """OCRテキストからノイズを除去し、整形する"""
    # ページ番号や独立した数字の行を削除
    text = re.sub(r'^\s*-\s*\d+\s*-\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # 文中の不要な改行をスペースに置換（ただし、箇条書きの改行は保持したいので工夫が必要）
    # ここでは単純に改行をスペースに置換
    text = re.sub(r'\n', ' ', text)
    # 連続するスペースを一つにまとめる
    text = re.sub(r'\s+', ' ', text)
    return text

def load_and_chunk_documents(directory: str) -> list[dict]:
    """文書を読み込み、章・条・項を考慮してチャンクに分割する"""
    all_chunks = []
    print(f"'{directory}'フォルダ内のドキュメントを読み込みます...")
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if not os.path.isfile(path) or not filename.endswith(".pdf"):
            continue

        print(f"  - ファイル '{filename}' を処理中...")
        full_text = ""
        with fitz.open(path) as doc:
            full_text = "".join(page.get_text("text", sort=True) for page in doc)

        # 章、条、附則などの見出しで分割
        # この正規表現は文書の構造に合わせて調整が必要
        sections = re.split(r'(第\s*[一二三四五六七八九十百]+章|第\s*\d+章|第\s*[一二三四五六七八九十百]+条|第\s*\d+条|附\s*則|別\s*表\s*\d+)', full_text)
        
        current_title = "序文"
        content_buffer = ""

        for section in sections:
            section = section.strip()
            if not section: continue

            # 見出しパターンにマッチした場合
            if re.match(r'(第\s*[一二三四五六七八九十百]+章|第\s*\d+章|第\s*[一二三四五六七八九十百]+条|第\s*\d+条|附\s*則|別\s*表\s*\d+)', section):
                # 前のセクションのバッファがあればチャンクとして処理
                if content_buffer:
                    preprocessed_content = preprocess_text(content_buffer)
                    if len(preprocessed_content) > 20: # 短すぎる内容はスキップ
                        all_chunks.append({
                            "text": f"{current_title} - {preprocessed_content}", # タイトル情報をテキストに含める
                            "source": filename,
                            "title": current_title
                        })
                # 新しいタイトルをセットしてバッファをリセット
                current_title = section
                content_buffer = ""
            else:
                # 見出しでない場合は内容としてバッファに追加
                content_buffer += section
        
        # 最後のセクションのバッファを処理
        if content_buffer:
            preprocessed_content = preprocess_text(content_buffer)
            if len(preprocessed_content) > 20:
                all_chunks.append({
                    "text": f"{current_title} - {preprocessed_content}",
                    "source": filename,
                    "title": current_title
                })

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

    # 確認用にチャンクをファイルに出力
    with open("chunks_output.txt", "w", encoding="utf-8") as f:
        f.write(f"合計チャンク数: {len(chunks)}\n\n")
        for i, chunk in enumerate(chunks):
            f.write(f"--- チャンク {i+1} (Source: {chunk['source']}, Title: {chunk['title']}) ---\n")
            f.write(chunk['text'])
            f.write("\n\n")
    print("\n★ `chunks_output.txt` に分割されたチャンクを出力しました。中身を確認してください。")

    user_confirm = input("インデックス作成を続行しますか？ (y/n): ")
    if user_confirm.lower() != 'y':
        print("処理を中断しました。")
        return

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
                "metadata": {"source": chunk['source'], "title": chunk['title'], "text": chunk['text']}
            })
        
        if vectors_to_upsert:
            pinecone_index.upsert(vectors=vectors_to_upsert, namespace=NAMESPACE)
        time.sleep(1)

    print("\nすべてのドキュメントのインデックス作成が完了しました！")
    stats = pinecone_index.describe_index_stats()
    print(f"名前空間 '{NAMESPACE}' に {stats.get('namespaces', {}).get(NAMESPACE, {}).get('vector_count', 0)} 件のベクトルが保存されています。")

if __name__ == "__main__":
    main()