# ============================================================
# app.py (ステップ3: 会社資料Q&A対応版 - 改善版)
# ============================================================

import os
import random
import re
import base64
import json
import requests
import time
import textwrap
import uuid
from typing import Optional

from flask import Flask, request
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    ImageSendMessage, ImageMessage, StickerMessage
)

# --- AI & Cloud Libraries ---
import google.generativeai as genai
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import redis
import pinecone
from dotenv import load_dotenv

# --- 他のPythonファイルからインポート ---
from character_makot import MAKOT, build_system_prompt, apply_expression_style

# ------------------------------------------------------------
# 初期化処理
# ------------------------------------------------------------
load_dotenv('.env.development.local')
app = Flask(__name__)
# 環境変数
GEMINI_API_KEY            = os.getenv("GEMINI_API_KEY")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET       = os.getenv("LINE_CHANNEL_SECRET")
IMGUR_CLIENT_ID           = os.getenv("IMGUR_CLIENT_ID")
GCP_PROJECT_ID            = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION              = os.getenv("GCP_LOCATION", "us-central1")
GCP_CREDENTIALS_JSON_STR  = os.getenv("GCP_CREDENTIALS_JSON")
REDIS_URL                 = os.getenv("REDIS_URL")
PINECONE_API_KEY          = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME       = os.getenv("PINECONE_INDEX_NAME")

# 各種クライアントの初期化
genai.configure(api_key=GEMINI_API_KEY, transport="rest")
text_model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20") # モデルを更新
embedding_model = "models/text-embedding-004"
line_bot_api    = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
webhook_handler = WebhookHandler(LINE_CHANNEL_SECRET)
if not REDIS_URL: raise ValueError("REDIS_URL 環境変数が設定されていません。")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)
gcp_token_cache = {"token": None, "expires_at": 0}

if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    raise ValueError("Pineconeの環境変数(API_KEY, INDEX_NAME)が設定されていません。")
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)


# ------------------------------------------------------------
# ベクトル化 & RAG関連関数
# ------------------------------------------------------------
def get_embedding(text: str) -> list[float]:
    """テキストをベクトルに変換する（汎用）"""
    try:
        result = genai.embed_content(model=embedding_model, content=text)
        return result['embedding']
    except Exception as e:
        print(f"ベクトル化エラー: {e}")
        return []

def summarize_and_store_memory(user_id: str, history: list[str]):
    """会話を要約し、ベクトル化してPineconeに長期記憶として保存する"""
    recent_talk = "\n".join(history[-4:])
    if len(recent_talk) < 50: return

    summary_prompt = textwrap.dedent(f"""
        あなたはユーザーとの会話の要約担当です。以下の会話から、ユーザーの個人的な情報（名前、好み、最近の出来事、ペット、悩み、計画など）を抽出し、簡潔な箇条書きのメモとして1～2行で要約してください。重要な情報が含まれていない場合は、必ず「特になし」とだけ出力してください。
        ---
        会話:
        {recent_talk}
        ---
        要約:""")
    try:
        summary_response = text_model.generate_content(summary_prompt)
        summary = summary_response.text.strip()

        if summary and "特になし" not in summary:
            vector = get_embedding(summary)
            if not vector: return

            memory_id = str(uuid.uuid4())
            metadata = { "user_id": user_id, "text": summary, "created_at": time.time() }
            pinecone_index.upsert(vectors=[(memory_id, vector, metadata)], namespace="conversation-memory")
            print(f"[{user_id}] の新しい記憶をベクトルDBに保存しました: {summary}")
    except Exception as e:
        print(f"記憶の保存処理でエラー: {e}")

# ------------------------------------------------------------
# Q&Aモードと通常会話モードの処理（改善版）
# ------------------------------------------------------------
QA_SYSTEM_PROMPT = textwrap.dedent("""
    あなたは、後輩女子『まこT』として、提供された参考情報に【基づいてのみ】ユーザーの質問に回答するアシスタントです。
    あなたの役割は、参考情報の内容を分かりやすく、親しみやすい口調で要約して伝えることです。

    【重要ルール】
    - 必ず参考情報に含まれる事実だけを使って回答してください。
    - 参考情報に答えがない場合や、関連性が低い場合は、絶対に推測で答えてはいけません。代わりに「うーん、その情報は見当たらないですね…！ごめんなさい🥺」と正直に回答してください。
    - 回答の最後に出典（source）を `(参考: ファイル名)` の形で付け加えてください。

    【参考情報】
    {context}

    【ユーザーの質問】
    {question}

    以上のルールを厳格に守り、『まこT』として回答してください：
""")

def get_qa_embedding(text: str, task_type="RETRIEVAL_QUERY") -> list[float]:
    """Q&A検索用のテキストをベクトルに変換する"""
    try:
        result = genai.embed_content(model=embedding_model, content=text, task_type=task_type)
        return result['embedding']
    except Exception as e:
        print(f"QAベクトル化エラー: {e}")
        return []

def expand_query(question: str) -> list[str]:
    """LLMを使って質問を複数の表現に拡張する"""
    prompt = textwrap.dedent(f"""
        ユーザーの質問を、ベクトル検索でよりヒットしやすくなるように、異なる視点から3つの類義質問や検索キーワードに書き換えてください。
        元の質問も必ず含めてください。箇条書き（ハイフン区切り）で、説明は不要です。
        
        例1:
        質問: 料金の支払いについて教えて
        書き換え:
        - 料金の支払いについて教えて
        - 料金の算定および支払い方法
        - 支払期日を過ぎた場合の延滞利息
        
        例2:
        質問: FRT要件って何ですか？
        書き換え:
        - FRT要件って何ですか？
        - 事故時運転継続要件の定義
        - FRT要件を満たすための条件
        
        質問: {question}
        書き換え:
    """)
    try:
        response = text_model.generate_content(prompt)
        queries = [line.strip().lstrip('- ') for line in response.text.strip().split('\n') if line.strip()]
        return list(set(queries)) # 重複を削除
    except Exception as e:
        print(f"クエリ拡張エラー: {e}")
        return [question] # 失敗した場合は元の質問だけを返す


def chat_with_makot(user_input: str, user_id: str) -> str:
    QA_TRIGGERS = ["教えて", "説明して", "規定", "ルール", "方法", "って何", "とは", "について", "の条件"]
    is_qa_mode = any(trigger in user_input for trigger in QA_TRIGGERS)

    if is_qa_mode:
        print(f"[{user_id}] Q&Aモードで実行します。")
        try:
            expanded_queries = expand_query(user_input)
            print(f"  [クエリ拡張] 元の質問: '{user_input}' -> 拡張後: {expanded_queries}")

            all_matches = {}
            for query in expanded_queries:
                query_vector = get_qa_embedding(query)
                if not query_vector: continue

                query_response = pinecone_index.query(
                    vector=query_vector,
                    top_k=3,
                    namespace="company-docs",
                    include_metadata=True
                )
                for match in query_response['matches']:
                    if match.id not in all_matches or match.score > all_matches[match.id].score:
                        all_matches[match.id] = match

            sorted_matches = sorted(all_matches.values(), key=lambda x: x.score, reverse=True)

            context_chunks = []
            sources = set()
            
            print("\n--- 統合後の検索結果 ---")
            for match in sorted_matches[:5]:
                # ★★★ ログ出力を強化し、章の情報も表示 ★★★
                print(f"  [検索結果] Score: {match.score:.4f}, Source: {match.metadata['source']}, Chapter: {match.metadata.get('chapter', 'N/A')}, Title: {match.metadata.get('title', 'N/A')}")
                if match.score > 0.55:
                     # ★★★ LLMに与えるコンテキストに「章」の情報も追加 ★★★
                     context_chunks.append(f"【出典: {match.metadata['source']} / 章: {match.metadata.get('chapter', 'N/A')} / 見出し: {match.metadata.get('title', 'N/A')}】\n{match.metadata['text']}")
                     sources.add(match.metadata['source'])

            if not context_chunks:
                return "うーん、その情報は見当たらないですね…！ごめんなさい🥺"

            context_str = "\n---\n".join(context_chunks)
            source_str = f"(参考: {', '.join(sorted(list(sources)))})"

            prompt = QA_SYSTEM_PROMPT.format(context=context_str, question=user_input)
            response = text_model.generate_content(prompt)
            reply = response.text.strip()
            
            if "ごめんなさい" not in reply and "参考:" not in reply:
                reply += f" {source_str}"

            # ★★★ 修正箇所 ★★★
            # Q&Aモードの回答に含まれるMarkdown記法(*, `)も除去する
            reply = re.sub(r'[\*`＊∗]+', '', reply)
            
            return reply

        except Exception as e:
            print(f"Q&A処理エラー: {e}")
            return "ごめんなさい、なんだかシステムが不調みたいです…。もう一度試してみてください！"

    else:
        # --- 通常会話モード (変更なし) ---
        print(f"[{user_id}] 通常会話モードで実行します。")
        history_key = f"chat_history:{user_id}"
        history_json = redis_client.get(history_key)
        history: list[str] = json.loads(history_json) if history_json else []

        long_term_memory = None
        try:
            input_vector = get_embedding(user_input)
            if input_vector:
                query_response = pinecone_index.query(
                    vector=input_vector,
                    top_k=3,
                    namespace="conversation-memory",
                    filter={"user_id": user_id},
                    include_metadata=True
                )
                relevant_memories = [match['metadata']['text'] for match in query_response['matches'] if match['score'] > 0.7]
                if relevant_memories:
                    long_term_memory = "\n".join(f"- {mem}" for mem in relevant_memories)
                    print(f"[{user_id}] の関連記憶を検索: {long_term_memory}")
        except Exception as e:
            print(f"記憶の検索エラー: {e}")

        history.append(f"ユーザー: {user_input}")
        context = "\n".join(history[-12:])
        topic = guess_topic(user_input)
        system_prompt = build_system_prompt(
            context=context, topic=topic, user_id=user_id, long_term_memory=long_term_memory
        )
        
        try:
            response = text_model.generate_content(system_prompt)
            reply = response.text.strip()
        except Exception as e:
            reply = f"エラーが発生しました: {e}"

        reply = post_process(reply, user_input)
        pronoun = decide_pronoun(user_input)
        reply = inject_pronoun(reply, pronoun)
        history.append(f"アシスタント: {reply}")

        redis_client.set(history_key, json.dumps(history[-50:]))
        summarize_and_store_memory(user_id, history)

        return reply

# ------------------------------------------------------------
# ユーティリティ & Webhookハンドラ
# ------------------------------------------------------------
def is_bot_mentioned(text: str) -> bool: return any(nick in text for nick in [MAKOT["name"]] + MAKOT["nicknames"])
def guess_topic(text: str):
    hobby_keys = ["趣味", "休日", "ハマって", "コストコ", "ポケポケ"]; work_keys  = ["仕事", "業務", "残業", "請求書", "統計"]
    if any(k in text for k in hobby_keys): return "hobby"
    if any(k in text for k in work_keys): return "work"
    return None
def decide_pronoun(user_text: str) -> str:
    high_hit = any(k in user_text for k in MAKOT["emotion_triggers"]["high"])
    if not high_hit: return "私"
    return "マコ" if random.random() < 0.10 else "おに"
def inject_pronoun(reply: str, pronoun: str) -> str: return re.sub(r"^(私|おに|マコ)", pronoun, reply, count=1)
UNCERTAIN = ["かも", "かもしれ", "たぶん", "多分", "かな", "と思う", "気がする"]
def post_process(reply: str, user_input: str) -> str:
    high = any(t in user_input for t in MAKOT["emotion_triggers"]["high"])
    low  = any(t in user_input for t in MAKOT["emotion_triggers"]["low"])
    if high: reply = apply_expression_style(reply, mood="high")
    elif low: reply += " 🥺"

    reply = re.sub(r'[\*`＊∗]+', '', reply) # Markdown記法 **, *, ` を除去

    if any(w in reply for w in UNCERTAIN) and random.random() < 0.4:
        reply += " しらんけど"
    
    reply_sentences = re.split(r'([。！？])', reply)
    if len(reply_sentences) > 5:
        processed_reply = ""
        count = 0
        for i in range(0, len(reply_sentences), 2):
            if i+1 < len(reply_sentences):
                processed_reply += reply_sentences[i] + reply_sentences[i+1]
            else:
                processed_reply += reply_sentences[i]
            count += 1
            if count >= 2: break # 2文でカット
        reply = processed_reply
    
    return reply
def get_gcp_token() -> str:
    if gcp_token_cache["token"] and time.time() < gcp_token_cache["expires_at"]: return gcp_token_cache["token"]
    if not GCP_CREDENTIALS_JSON_STR: raise ValueError("GCP_CREDENTIALS_JSON 環境変数が設定されていません。")
    try:
        credentials_info = json.loads(GCP_CREDENTIALS_JSON_STR); creds = service_account.Credentials.from_service_account_info(credentials_info, scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds.refresh(Request());
        if not creds.token: raise ValueError("トークンの取得に失敗しました。")
        gcp_token_cache["token"] = creds.token; gcp_token_cache["expires_at"] = time.time() + 3300
        return creds.token
    except Exception as e: print(f"get_gcp_tokenでエラー: {e}"); raise
def upload_to_imgur(image_bytes: bytes, client_id: str) -> str:
    if not client_id: raise Exception("Imgur Client IDが設定されていません。")
    url = "https://api.imgur.com/3/image"; headers = {"Authorization": f"Client-ID {client_id}"}
    try:
        response = requests.post(url, headers=headers, data={"image": base64.b64encode(image_bytes)}); response.raise_for_status(); data = response.json()
        if data.get("success"): return data["data"]["link"]
        else: raise Exception(f"Imgurへのアップロードに失敗しました: {data.get('data', {}).get('error', 'Unknown error')}")
    except requests.exceptions.RequestException as e: raise Exception(f"Imgur APIへのリクエストに失敗しました: {e}")
def translate_to_english(text: str) -> str:
    if not text: return "a cute girl"
    try:
        prompt = f"Translate the following Japanese into a simple English phrase for an image generation AI. For example, '猫' -> 'a cat', '空を飛ぶ犬' -> 'a dog flying in the sky'. Do not add any extra explanation. Just the translated phrase.\nJapanese: {text}\nEnglish:"
        response = text_model.generate_content(prompt); translated_text = response.text.strip().replace('"', '')
        return translated_text
    except Exception as e: print(f"翻訳でエラーが発生: {e}"); return text
def generate_image_with_rest_api(prompt: str) -> str:
    token = get_gcp_token(); endpoint_url = (f"https://{GCP_LOCATION}-aiplatform.googleapis.com/v1/projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}/publishers/google/models/imagegeneration@006:predict")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"}
    trigger_words = ["画像", "イラスト", "描いて", "絵を"]; clean_prompt = prompt
    for word in trigger_words: clean_prompt = clean_prompt.replace(word, "")
    clean_prompt = clean_prompt.strip(); english_prompt = translate_to_english(clean_prompt); final_prompt = f"anime style illustration, masterpiece, best quality, {english_prompt}"
    data = {"instances": [{"prompt": final_prompt}], "parameters": {"sampleCount": 1, "aspectRatio": "1:1", "negativePrompt": "low quality, bad hands, text, watermark, signature"}}
    response = requests.post(endpoint_url, headers=headers, json=data); response.raise_for_status(); response_data = response.json()
    if "predictions" not in response_data or not response_data["predictions"]:
        error_info = response_data.get("error", {}).get("message", json.dumps(response_data)); raise Exception(f"APIから画像データが返されませんでした。サーバーの応答: {error_info}")
    b64_image = response_data["predictions"][0]["bytesBase64Encoded"]; image_bytes = base64.b64decode(b64_image)
    return upload_to_imgur(image_bytes, IMGUR_CLIENT_ID)

@app.route("/line_webhook", methods=["POST"])
def line_webhook():
    signature = request.headers.get("X-Line-Signature"); body = request.get_data(as_text=True)
    try: webhook_handler.handle(body, signature)
    except InvalidSignatureError: return "Invalid signature", 400
    return "OK", 200

@webhook_handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    src_type = event.source.type; user_text = event.message.text
    if src_type in ["group", "room"] and not is_bot_mentioned(user_text): return
    user_id = event.source.user_id
    
    if any(key in user_text for key in ["画像", "イラスト", "描いて", "絵を"]):
        try:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="おっけーです！ちょっと待っててくださいね…🥰"))
            img_url = generate_image_with_rest_api(user_text)
            msg = ImageSendMessage(original_content_url=img_url, preview_image_url=img_url)
            line_bot_api.push_message(user_id, msg)
        except Exception as e:
            print(f"画像生成でエラーが発生: {e}")
            line_bot_api.push_message(user_id, TextSendMessage(text=f"ごめんなさい、画像生成の調子が悪いみたいです…\n理由: {e}"))
        return
    reply_text = chat_with_makot(user_text, user_id=user_id)
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

@webhook_handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    try:
        message_content = line_bot_api.get_message_content(event.message.id)
        image_bytes = message_content.content
        makot_prompt = "あなたは後輩女子の『まこT』です。ユーザーから送られてきたこの画像を見て、最高のリアクションを1～2文で返してください！食べ物なら「おいしそう！」、動物なら「かわいい！」など、見たままの感情をテンション高めに表現してください。"
        response = text_model.generate_content([makot_prompt, {"mime_type": "image/jpeg", "data": image_bytes}])
        reply_text = response.text.strip()
        reply_text = post_process(reply_text, "テンション上がる")
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
    except Exception as e:
        print(f"画像認識でエラーが発生: {e}")
        if "support image" in str(e).lower():
             reply_text = "ごめんなさい、今ちょっと目が悪くて画像が見れないみたいです…🥺 また今度見せてください！"
        else:
             reply_text = "ごめんなさい、画像がうまく見れなかったです…🥺"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

@webhook_handler.add(MessageEvent, message=StickerMessage)
def handle_sticker_message(event):
    sticker_map = { "11537": {"52002734": "ありがとうございます！うれしいです🥰", "52002748": "おつかれさまです！🙇‍♀️"}, "11538": {"51626494": "ひえっ…！なにかありましたか！？🥺", "51626501": "ふぁーーーーーーーーーーーｗｗｗｗｗｗｗ"} }
    package_id = event.message.package_id; sticker_id = event.message.sticker_id
    reply_text = sticker_map.get(str(package_id), {}).get(str(sticker_id))
    if not reply_text: reply_text = random.choice(["スタンプありがとうございます！🥰", "そのスタンプかわいいですね！", "お、いいスタンプ！私もほしいです！"])
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

@app.route("/")
def home():
    return "まこT LINE Bot is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))