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
CHUNK_SIZE = 800  # チャンクの最大文字数
NAMESPACE = "company-docs"

def get_embedding(text: str, task_type="RETRIEVAL_DOCUMENT") -> list[float]:
    """テキストをベクトルに変換する"""
    try:
        result = genai.embed_content(model=embedding_model, content=text, task_type=task_type)
        return result['embedding']
    except Exception as e:
        print(f"ベクトル化エラー: {e}")
        return []

def preprocess_text(text: str) -> str:
    """OCRテキストからノイズを除去し、整形する関数"""
    text = re.sub(r'－\s*\d+\s*－', '', text) # ヘッダー/フッター除去
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE) # ページ番号除去
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text) # 文中の不要な改行をスペースに
    text = re.sub(r'\s+', ' ', text) # 連続するスペースを一つに
    return text.strip()

def chunk_and_append(content: str, title: str, chapter: str, filename: str, all_chunks: list):
    """テキストを前処理し、チャンク化してリストに追加するヘルパー関数"""
    cleaned_content = preprocess_text(content)
    if len(cleaned_content) < 30: return

    if len(cleaned_content) > CHUNK_SIZE:
        parts = cleaned_content.split('。')
        current_chunk = ""
        for part in parts:
            if not part: continue
            # チャンクサイズを超える場合は現在のチャンクを確定し、新しいチャンクを開始
            if len(current_chunk) + len(part) + 1 > CHUNK_SIZE:
                if current_chunk:
                    all_chunks.append({"text": current_chunk.strip() + "。", "source": filename, "title": title, "chapter": chapter})
                current_chunk = part.strip()
            else:
                current_chunk += ("" if not current_chunk else " ") + part.strip() + "。"
        # 最後のチャンクを追加
        if current_chunk:
             all_chunks.append({"text": current_chunk, "source": filename, "title": title, "chapter": chapter})
    else:
        all_chunks.append({"text": cleaned_content, "source": filename, "title": title, "chapter": chapter})

def process_section(section_text: str, chapter: str, filename: str, all_chunks: list):
    """章の内容を条文ごとに分割してチャンク化する"""
    # 条文で分割。「第X条 (...)」または「(...)」形式の見出しをキャプチャ
    articles = re.split(r'((?:^第\s*[\d百数十]+条.*?$)|(?:^（.*?）$))', section_text, flags=re.MULTILINE)
    
    # 条文がない部分（章の導入部など）を処理
    if articles and articles[0].strip():
        chunk_and_append(articles[0], chapter, chapter, filename, all_chunks)

    # 条文ごとの処理
    for i in range(1, len(articles), 2):
        article_title = articles[i].strip().replace('\n', ' ')
        article_content = articles[i+1]
        chunk_and_append(article_content, article_title, chapter, filename, all_chunks)

def load_and_chunk_documents(directory: str) -> list[dict]:
    """文書を読み込み、章・条を考慮してチャンクに分割する賢い関数（改善版）"""
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
                full_text = "".join(page.get_text("text", sort=True) for page in doc)
        except Exception as e:
            print(f"  PDF読み込みエラー: {e}")
            continue

        # 章、附則、別表などで文書を大きなセクションに分割
        # PDFの構造に合わせて正規表現を調整
        section_pattern = r'((?:^第\s*[\d一二三四五六七八九十百]+章.*?$)|(?:^附\s*則.*?$)|(?:^別\s*表.*?$)|(?:^Ⅰ\s+総\s*則)|(?:^Ⅱ\s+.*?要件)|(?:^Ⅲ\s+.*?要件)|(?:^Ⅳ\s+.*?基準))'
        sections = re.split(section_pattern, full_text, flags=re.MULTILINE)

        # 最初のセクション（序文など）を処理
        if sections and sections[0].strip():
            process_section(sections[0], "序文", filename, all_chunks)
        
        # 章ごとの処理
        for i in range(1, len(sections), 2):
            chapter_title = sections[i].strip().replace('\n', ' ')
            chapter_text = sections[i+1]
            process_section(chapter_text, chapter_title, filename, all_chunks)
    
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
            f.write(f"--- チャンク {i+1} (Source: {chunk['source']}, Chapter: {chunk['chapter']}, Title: {chunk['title']}) ---\n")
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
            # 検索時の関連性を高めるため、階層的なメタデータもテキストに含めてベクトル化
            text_for_embedding = f"文書: {chunk['source']}, 章: {chunk['chapter']}, 見出し: {chunk['title']}\n内容: {chunk['text']}"
            vector = get_embedding(text_for_embedding)
            if not vector: continue
            
            # メタデータには元のテキストと構造化情報を保存
            vectors_to_upsert.append({
                "id": str(uuid.uuid4()),
                "values": vector,
                "metadata": {
                    "source": chunk['source'], 
                    "chapter": chunk['chapter'], 
                    "title": chunk['title'], 
                    "text": chunk['text']
                }
            })
        
        if vectors_to_upsert:
            pinecone_index.upsert(vectors=vectors_to_upsert, namespace=NAMESPACE)
        time.sleep(1) # APIレート制限対策

    print("\nすべてのドキュメントのインデックス作成が完了しました！")
    stats = pinecone_index.describe_index_stats()
    print(f"名前空間 '{NAMESPACE}' に {stats.get('namespaces', {}).get(NAMESPACE, {}).get('vector_count', 0)} 件のベクトルが保存されています。")

if __name__ == "__main__":
    main()