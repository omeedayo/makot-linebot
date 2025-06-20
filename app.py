# ============================================================
# app.py (本格RAG対応版)
# ============================================================

import os
import random
import re
import base64
import json
import requests
import time
import textwrap
import uuid       # ★ 追加: 記憶に一意のIDを付与するため
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
import pinecone   # ★ 追加: ベクトルDBライブラリ

# --- 他のPythonファイルからインポート ---
from character_makot import MAKOT, build_system_prompt, apply_expression_style

# ------------------------------------------------------------
# 初期化処理
# ------------------------------------------------------------
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
PINECONE_API_KEY          = os.getenv("PINECONE_API_KEY")      # ★ 追加
PINECONE_INDEX_NAME       = os.getenv("PINECONE_INDEX_NAME")  # ★ 追加

# 各種クライアントの初期化
genai.configure(api_key=GEMINI_API_KEY, transport="rest")
text_model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
embedding_model = "models/text-embedding-004" # ★ 追加: ベクトル化用モデル
line_bot_api    = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
webhook_handler = WebhookHandler(LINE_CHANNEL_SECRET)
if not REDIS_URL: raise ValueError("REDIS_URL 環境変数が設定されていません。")
redis_client = redis.from_url(REDIS_URL) # 短期記憶(会話履歴)用として引き続き利用
gcp_token_cache = {"token": None, "expires_at": 0}

# ★ 追加: Pineconeクライアントの初期化
if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    raise ValueError("Pineconeの環境変数(API_KEY, INDEX_NAME)が設定されていません。")
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)


# ------------------------------------------------------------
# ★★★ 新設: ベクトル化 & RAG関連関数 ★★★
# ------------------------------------------------------------
def get_embedding(text: str) -> list[float]:
    """テキストをベクトル（AIが意味を理解できる数値配列）に変換する"""
    try:
        result = genai.embed_content(model=embedding_model, content=text)
        return result['embedding']
    except Exception as e:
        print(f"ベクトル化でエラー: {e}")
        return []

def summarize_and_store_memory(user_id: str, history: list[str]):
    """会話を要約し、ベクトル化してPineconeに長期記憶として保存する"""
    recent_talk = "\n".join(history[-2:])
    if len(recent_talk) < 20: return

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
            # 1. 要約文をベクトル化
            vector = get_embedding(summary)
            if not vector: return

            # 2. 一意のIDと、検索用のメタデータを作成
            memory_id = str(uuid.uuid4())
            metadata = { "user_id": user_id, "text": summary, "created_at": time.time() }
            
            # 3. Pineconeにベクトルとメタデータを保存
            pinecone_index.upsert(vectors=[(memory_id, vector, metadata)])
            print(f"[{user_id}] の新しい記憶をベクトルDBに保存しました: {summary}")

    except Exception as e:
        print(f"記憶の保存処理でエラー: {e}")

# ------------------------------------------------------------
# ★★★ メインロジックをRAG対応に刷新 ★★★
# ------------------------------------------------------------
def chat_with_makot(user_input: str, user_id: str) -> str:
    # 1. 短期記憶(会話履歴)をRedisから取得
    history_key = f"chat_history:{user_id}"
    history_json = redis_client.get(history_key)
    history: list[str] = json.loads(history_json) if history_json else []

    # 2. ★ RAG検索: ユーザーの今の発言に最も関連する長期記憶をPineconeから検索
    long_term_memory = None
    try:
        input_vector = get_embedding(user_input)
        if input_vector:
            # 同じユーザーの記憶の中から、意味が近いものを最大3つ検索
            query_response = pinecone_index.query(
                vector=input_vector,
                top_k=3,
                filter={"user_id": user_id},
                include_metadata=True
            )
            # 見つかった記憶のテキスト部分を取り出す
            relevant_memories = [match['metadata']['text'] for match in query_response['matches']]
            if relevant_memories:
                long_term_memory = "\n".join(f"- {mem}" for mem in relevant_memories)
                print(f"[{user_id}] の関連記憶を検索: {long_term_memory}")

    except Exception as e:
        print(f"記憶の検索でエラー: {e}")

    # 3. 会話履歴とプロンプトの生成
    history.append(f"ユーザー: {user_input}")
    context = "\n".join(history[-12:])
    topic = guess_topic(user_input)
    system_prompt = build_system_prompt(
        context=context,
        topic=topic,
        user_id=user_id,
        long_term_memory=long_term_memory # ★ 検索した関連記憶だけをカンペとして渡す
    )
    
    # 4. AIによる応答生成
    try:
        response = text_model.generate_content(system_prompt)
        reply = response.text.strip()
    except Exception as e:
        reply = f"エラーが発生しました: {e}"

    # 5. 応答の加工と保存
    reply = post_process(reply, user_input)
    pronoun = decide_pronoun(user_input)
    reply = inject_pronoun(reply, pronoun)
    history.append(f"アシスタント: {reply}")

    # 6. 短期記憶をRedisに保存
    redis_client.set(history_key, json.dumps(history[-50:]))

    # 7. ★ 新しい長期記憶の保存処理を呼び出す
    summarize_and_store_memory(user_id, history)

    return reply

# ------------------------------------------------------------
# (ここから下のコードは一切変更ありません)
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
    high = any(t in user_input for t in MAKOT["emotion_triggers"]["high"]); low  = any(t in user_input for t in MAKOT["emotion_triggers"]["low"])
    if high: reply = apply_expression_style(reply, mood="high")
    elif low: reply += " 🥺"
    if any(w in reply for w in UNCERTAIN) and random.random() < 0.4: reply += " しらんけど"
    reply_sentences = re.split(r'(。|！|？)', reply)
    if len(reply_sentences) > 4: reply = "".join(reply_sentences[:4])
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
    src_id = (event.source.user_id if src_type == "user" else event.source.group_id if src_type == "group" else event.source.room_id if src_type == "room" else "unknown")
    if any(key in user_text for key in ["画像", "イラスト", "描いて", "絵を"]):
        try:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="おっけーです！ちょっと待っててくださいね…🥰"))
            img_url = generate_image_with_rest_api(user_text)
            msg = ImageSendMessage(original_content_url=img_url, preview_image_url=img_url)
            line_bot_api.push_message(src_id, msg)
        except Exception as e:
            print(f"画像生成でエラーが発生: {e}")
            line_bot_api.push_message(src_id, TextSendMessage(text=f"ごめんなさい、画像生成の調子が悪い・・・のはおめぇのせいだよ\n理由: {e}"))
        return
    reply_text = chat_with_makot(user_text, user_id=src_id)
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
    reply_text = sticker_map.get(package_id, {}).get(sticker_id)
    if not reply_text: reply_text = random.choice(["スタンプありがとうございます！🥰", "そのスタンプかわいいですね！", "お、いいスタンプ！私もほしいです！"])
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

@app.route("/")
def home():
    return "まこT LINE Bot is running!"
