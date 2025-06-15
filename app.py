# ============================================================
# app.py (最終修正版)
# Gemini Flash (text)  +  Vertex AI Imagen (REST API)  +  Imgur upload
# ============================================================

import os
import random
import re
import base64
import json
import requests

from flask import Flask, request
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent,
    TextMessage,
    TextSendMessage,
    ImageSendMessage,
)

# --- AI & Cloud Libraries ---
import google.generativeai as genai
# ★変更: google-authライブラリで認証を行う
from google.oauth2 import service_account
from google.auth.transport.requests import Request

# (character_makot.py は変更なし)
from character_makot import MAKOT, build_system_prompt, apply_expression_style

# ------------------------------------------------------------
# Flask & LINE Bot setup
# ------------------------------------------------------------
app = Flask(__name__)

# --- 環境変数 ---
GEMINI_API_KEY            = os.getenv("GEMINI_API_KEY")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET       = os.getenv("LINE_CHANNEL_SECRET")
IMGUR_CLIENT_ID           = os.getenv("IMGUR_CLIENT_ID")
GCP_PROJECT_ID            = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION              = os.getenv("GCP_LOCATION", "us-central1")
GCP_CREDENTIALS_JSON_STR  = os.getenv("GCP_CREDENTIALS_JSON")

# --- Gemini client (text) ---
genai.configure(api_key=GEMINI_API_KEY, transport="rest")

# --- LINE SDK ---
line_bot_api    = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
webhook_handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ------------------------------------------------------------
# In‑memory simple chat history (per user / group)
# (このセクションは変更なし)
# ------------------------------------------------------------
chat_histories: dict[str, list[str]] = {}

# ------------------------------------------------------------
# Helpers & Post‑process
# (このセクションは変更なし)
# ------------------------------------------------------------
NICKNAMES = [MAKOT["name"]] + MAKOT["nicknames"]
def is_bot_mentioned(text: str) -> bool:
    return any(nick in text for nick in NICKNAMES)
def guess_topic(text: str) -> str | None:
    hobby_keys = ["趣味", "休日", "ハマって", "コストコ", "ポケポケ"]
    work_keys  = ["仕事", "業務", "残業", "請求書", "統計"]
    if any(k in text for k in hobby_keys): return "hobby"
    if any(k in text for k in work_keys): return "work"
    return None
def decide_pronoun(user_text: str) -> str:
    high_hit = any(k in user_text for k in MAKOT["emotion_triggers"]["high"])
    if not high_hit: return "私"
    return "マコ" if random.random() < 0.10 else "おに"
def inject_pronoun(reply: str, pronoun: str) -> str:
    return re.sub(r"^(私|おに|マコ)", pronoun, reply, count=1)
UNCERTAIN = ["かも", "かもしれ", "たぶん", "多分", "かな", "と思う", "気がする"]
def post_process(reply: str, user_input: str) -> str:
    high = any(t in user_input for t in MAKOT["emotion_triggers"]["high"])
    low  = any(t in user_input for t in MAKOT["emotion_triggers"]["low"])
    if high: reply = apply_expression_style(reply, mood="high")
    elif low: reply += " 🥺"
    if any(w in reply for w in UNCERTAIN) and random.random() < 0.4:
        reply += " しらんけど"
    return reply

# ------------------------------------------------------------
# ★★★ ここから大幅に変更 ★★★
# Vertex AI Imagen (REST API) ➜ Imgur upload
# ------------------------------------------------------------

def get_gcp_token() -> str:
    """サービスアカウントキーからGCP API用のアクセストークンを確実に取得する"""
    if not GCP_CREDENTIALS_JSON_STR:
        raise ValueError("GCP_CREDENTIALS_JSON 環境変数が設定されていません。")
    
    try:
        credentials_info = json.loads(GCP_CREDENTIALS_JSON_STR)
        creds = service_account.Credentials.from_service_account_info(
            credentials_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        
        # 常にトークンをリフレッシュして最新の状態を確保する
        creds.refresh(Request())
        
        if not creds.token:
            raise ValueError("トークンの取得に失敗しました。")
        
        return creds.token
    except Exception as e:
        # 認証プロセスでエラーが起きた場合に詳細をログに出力
        print(f"get_gcp_tokenでエラー: {e}")
        raise

def upload_to_imgur(image_bytes: bytes, client_id: str) -> str:
    """画像をImgurにアップロードして公開URLを返す (この関数は変更なし)"""
    if not client_id: raise Exception("Imgur Client IDが設定されていません。")
    url = "https://api.imgur.com/3/image"
    headers = {"Authorization": f"Client-ID {client_id}"}
    try:
        response = requests.post(
            url, headers=headers, data={"image": base64.b64encode(image_bytes)}
        )
        response.raise_for_status()
        data = response.json()
        if data.get("success"): return data["data"]["link"]
        else: raise Exception(f"Imgurへのアップロードに失敗しました: {data.get('data', {}).get('error', 'Unknown error')}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Imgur APIへのリクエストに失敗しました: {e}")

def generate_image_with_rest_api(prompt: str) -> str:
    """Vertex AIのREST APIを直接呼び出して画像を生成し、URLを返す"""
    token = get_gcp_token()
    
    # Vertex AI Imagen APIのエンドポイント
    endpoint_url = (
        f"https://{GCP_LOCATION}-aiplatform.googleapis.com/v1/projects/{GCP_PROJECT_ID}"
        f"/locations/{GCP_LOCATION}/publishers/google/models/imagegeneration@006:predict"
    )
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }
    
    # APIに送るデータ本体
    data = {
        "instances": [{"prompt": f"高品質なアニメイラスト, {prompt[:100]}"}],
        "parameters": {
            "sampleCount": 1,
            "aspectRatio": "1:1",
            "negativePrompt": "low quality, bad hands, text, watermark"
        }
    }
    
    # APIにリクエストを送信
    response = requests.post(endpoint_url, headers=headers, json=data)
    response.raise_for_status() # HTTPエラーがあれば例外を発生
    
    # レスポンスからBase64エンコードされた画像データを取得
    response_data = response.json()
    if "predictions" not in response_data or not response_data["predictions"]:
        raise Exception("APIから画像の予測データが返されませんでした。")
        
    b64_image = response_data["predictions"][0]["bytesBase64Encoded"]
    
    # Base64をデコードして画像のバイトデータに戻す
    image_bytes = base64.b64decode(b64_image)
    
    # Imgurにアップロードして公開URLを取得
    imgur_url = upload_to_imgur(image_bytes, IMGUR_CLIENT_ID)
    
    return imgur_url

# ------------------------------------------------------------
# Main chat logic (このセクションは変更なし)
# ------------------------------------------------------------
def chat_with_makot(user_input: str, user_id: str) -> str:
    history = chat_histories.get(user_id, [])
    history.append(f"ユーザー: {user_input}")
    context = "\n".join(history[-2:])
    topic = guess_topic(user_input)
    system_prompt = build_system_prompt(context, topic=topic)
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        resp  = model.generate_content(system_prompt)
        reply = resp.text.strip()
    except Exception as e:
        reply = f"エラーが発生しました: {e}"
    reply = post_process(reply, user_input)
    pronoun = decide_pronoun(user_input)
    reply = inject_pronoun(reply, pronoun)
    history.append(reply)
    chat_histories[user_id] = history
    return reply

# ------------------------------------------------------------
# Flask endpoints
# ------------------------------------------------------------
@app.route("/line_webhook", methods=["POST"])
def line_webhook():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)
    try:
        webhook_handler.handle(body, signature)
    except InvalidSignatureError:
        return "Invalid signature", 400
    return "OK", 200

@webhook_handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    src_type = event.source.type
    user_text = event.message.text
    if src_type in ["group", "room"] and not is_bot_mentioned(user_text): return
    src_id = (
        event.source.user_id if src_type == "user" else
        event.source.group_id if src_type == "group" else
        event.source.room_id  if src_type == "room" else "unknown"
    )
    if any(key in user_text for key in ["画像", "イラスト", "描いて", "絵を"]):
        try:
            # ★変更: 新しい関数を呼び出す
            img_url = generate_image_with_rest_api(user_text) 
            msg = ImageSendMessage(original_content_url=img_url, preview_image_url=img_url)
            line_bot_api.reply_message(event.reply_token, msg)
        except Exception as e:
            # エラーメッセージをより詳細に表示
            print(f"画像生成でエラーが発生: {e}") 
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"ごめん、画像生成でエラーでちゃった🥺\n理由: {e}"))
        return
    reply_text = chat_with_makot(user_text, user_id=src_id)
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

@app.route("/")
def home():
    return "makoT LINE Bot is running!"
