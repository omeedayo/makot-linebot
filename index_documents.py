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
    """OCRテキストからノイズを除去し、整形する関数"""
    # ヘッダー/フッターによくあるパターンを削除 (例: '－ 12 －')
    text = re.sub(r'－\s*\d+\s*－', '', text)
    # ページ番号のみの行を削除
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # 文中の不要な改行をスペースに置換
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # 連続するスペースを一つにまとめる
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_and_chunk_documents(directory: str) -> list[dict]:
    """文書を読み込み、章・条・項を考慮してチャンクに分割する賢い関数"""
    all_chunks = []
    print(f"'{directory}'フォルダ内のドキュメントを読み込みます...")
    if not os.path.exists(directory):
        print(f"エラー: '{directory}' フォルダが見つかりません。")
        return []

    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if not os.path.isfile(path) or not filename.endswith(".pdf"):
            continue

        print(f"\n--- ファイル '{filename}' を処理中... ---")
        full_text = ""
        try:
            with fitz.open(path) as doc:
                # sort=Trueで読み取り順序を改善
                full_text = "".join(page.get_text("text", sort=True) for page in doc)
        except Exception as e:
            print(f"  PDF読み込みエラー: {e}")
            continue

        # ★★★ 文書構造を意識した分割ロジック ★★★
        # 「第X章」「第Y条」「附則」「別表」などで文書を大きなセクションに分割
        # この正規表現は、文書のパターンに合わせて調整することが重要
        sections = re.split(r'(第\s*[一二三四五六七八九十百]+章|第\s*\d+章|第\s*[一二三四五六七八九十百]+条|第\s*\d+条|附\s*則|別\s*表\s*\d+)', full_text)
        
        current_title = "序文"
        
        # 分割されたセクションを交互に処理 (区切り文字と内容が交互に来る)
        for i in range(1, len(sections), 2):
            title = sections[i].strip()
            content = sections[i+1].strip()
            
            # 前処理でノイズを除去
            cleaned_content = preprocess_text(content)
            
            if len(cleaned_content) < 30: # 短すぎる内容はスキップ
                continue

            # チャンクが長すぎる場合はさらに分割
            if len(cleaned_content) > CHUNK_SIZE:
                for j in range(0, len(cleaned_content), CHUNK_SIZE):
                    sub_chunk = cleaned_content[j:j + CHUNK_SIZE]
                    all_chunks.append({
                        "text": sub_chunk,
                        "source": filename,
                        "title": title
                    })
            else:
                all_chunks.append({
                    "text": cleaned_content,
                    "source": filename,
                    "title": title
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
            # 検索時の関連性を高めるため、タイトル情報もテキストに含めてベクトル化
            text_for_embedding = f"文書: {chunk['source']}, 見出し: {chunk['title']}\n内容: {chunk['text']}"
            vector = get_embedding(text_for_embedding)
            if not vector: continue
            
            # メタデータには元のテキストと情報を保存
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