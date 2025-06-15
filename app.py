# ============================================================
# app.py (修正版)
# Gemini Flash (text)  +  Vertex AI Imagen (image)  +  Imgur upload
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
import vertexai # ★変更: Vertex AI SDK をインポート
from google.oauth2 import service_account # ★変更: サービスアカウント認証用
from vertexai.vision_models import ImageGenerationModel, Image # ★変更: Vertex AIの画像生成モデルをインポート

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
GCP_PROJECT_ID            = os.getenv("GCP_PROJECT_ID") # ★変更: 追加
GCP_LOCATION              = os.getenv("GCP_LOCATION", "us-central1") # ★変更: 追加
GCP_CREDENTIALS_JSON_STR  = os.getenv("GCP_CREDENTIALS_JSON") # ★変更: 追加

# --- Gemini client (text) ---
genai.configure(api_key=GEMINI_API_KEY, transport="rest")

# --- Vertex AI client (image generation) ---
# ★変更: Vercel環境用の認証設定
try:
    if GCP_CREDENTIALS_JSON_STR:
        credentials_info = json.loads(GCP_CREDENTIALS_JSON_STR)
        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION, credentials=credentials)
    else:
        # ローカル開発用 (gcloud auth application-default login で認証)
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
except Exception as e:
    print(f"Vertex AIの初期化に失敗しました: {e}")


# --- LINE SDK ---
line_bot_api    = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
webhook_handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ------------------------------------------------------------
# In‑memory simple chat history (per user / group)
# ------------------------------------------------------------
chat_histories: dict[str, list[str]] = {}

# ------------------------------------------------------------
# Helpers: mention / topic / pronoun
# (このセクションは変更なし)
# ------------------------------------------------------------
NICKNAMES = [MAKOT["name"]] + MAKOT["nicknames"]

def is_bot_mentioned(text: str) -> bool:
    return any(nick in text for nick in NICKNAMES)

def guess_topic(text: str) -> str | None:
    hobby_keys = ["趣味", "休日", "ハマって", "コストコ", "ポケポケ"]
    work_keys  = ["仕事", "業務", "残業", "請求書", "統計"]
    if any(k in text for k in hobby_keys):
        return "hobby"
    if any(k in text for k in work_keys):
        return "work"
    return None

def decide_pronoun(user_text: str) -> str:
    high_hit = any(k in user_text for k in MAKOT["emotion_triggers"]["high"])
    if not high_hit:
        return "私"
    return "マコ" if random.random() < 0.10 else "おに"

def inject_pronoun(reply: str, pronoun: str) -> str:
    return re.sub(r"^(私|おに|マコ)", pronoun, reply, count=1)

# ------------------------------------------------------------
# Post‑process (emoji / しらんけど etc.)
# (このセクションは変更なし)
# ------------------------------------------------------------
UNCERTAIN = ["かも", "かもしれ", "たぶん", "多分", "かな", "と思う", "気がする"]

def post_process(reply: str, user_input: str) -> str:
    high = any(t in user_input for t in MAKOT["emotion_triggers"]["high"])
    low  = any(t in user_input for t in MAKOT["emotion_triggers"]["low"])
    if high:
        reply = apply_expression_style(reply, mood="high")
    elif low:
        reply += " 🥺"
    if any(w in reply for w in UNCERTAIN) and random.random() < 0.4:
        reply += " しらんけど"
    return reply

# ------------------------------------------------------------
# ★変更: Vertex AI Imagen ➜ Imgur upload
# ------------------------------------------------------------

def upload_to_imgur(image_bytes: bytes, client_id: str) -> str:
    """画像をImgurにアップロードして公開URLを返す"""
    if not client_id:
        raise Exception("Imgur Client IDが設定されていません。")
    url = "https://api.imgur.com/3/image"
    headers = {"Authorization": f"Client-ID {client_id}"}
    try:
        response = requests.post(
            url,
            headers=headers,
            data={"image": base64.b64encode(image_bytes)}
        )
        response.raise_for_status()
        data = response.json()
        if data.get("success"):
            return data["data"]["link"]
        else:
            error_msg = data.get('data', {}).get('error', 'Unknown error')
            raise Exception(f"Imgurへのアップロードに失敗しました: {error_msg}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Imgur APIへのリクエストに失敗しました: {e}")

def generate_vertex_image(prompt: str) -> str:
    """Vertex AIのImagenで画像を生成し、ImgurにアップロードしてURLを返す"""
    # 最新のモデルを指定 (例: imagen@006)
    # 利用可能なモデルはドキュメントで確認できます
    model = ImageGenerationModel.from_pretrained("imagegeneration@006")
    
    # プロンプトが長すぎる場合があるため、簡略化する
    # ここでは単純に最初の100文字にしていますが、より賢い要約も可能です
    generation_prompt = f"高品質なアニメイラスト, {prompt[:100]}"

    response = model.generate_images(
        prompt=generation_prompt,
        number_of_images=1,
        aspect_ratio="1:1",  # 1:1, 9:16, 16:9 などが指定可能
        # ネガティブプロンプトで品質を上げることも可能
        negative_prompt="low quality, bad hands, text, watermark"
    )

    if not response.images:
        raise Exception("モデルから画像が生成されませんでした。")
    
    # レスポンスから画像のバイトデータを取得
    image_bytes = response.images[0]._image_bytes

    # Imgurにアップロードして公開URLを取得
    imgur_url = upload_to_imgur(image_bytes, IMGUR_CLIENT_ID)
    
    return imgur_url

# ------------------------------------------------------------
# Main chat logic
# (このセクションは変更なし)
# ------------------------------------------------------------
def chat_with_makot(user_input: str, user_id: str) -> str:
    # ... (変更なし) ...
    history = chat_histories.get(user_id, [])
    history.append(f"ユーザー: {user_input}")
    context = "\n".join(history[-2:])

    topic = guess_topic(user_input)
    system_prompt = build_system_prompt(context, topic=topic)

    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest") # モデル名を最新版に更新
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

    if src_type in ["group", "room"] and not is_bot_mentioned(user_text):
        return

    src_id = (
        event.source.user_id if src_type == "user" else
        event.source.group_id if src_type == "group" else
        event.source.room_id  if src_type == "room" else "unknown"
    )

    if any(key in user_text for key in ["画像", "イラスト", "描いて", "絵を"]):
        try:
            # ★変更: 呼び出す関数名を変更
            img_url = generate_vertex_image(user_text) 
            msg = ImageSendMessage(original_content_url=img_url, preview_image_url=img_url)
            line_bot_api.reply_message(event.reply_token, msg)
        except Exception as e:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"画像生成に失敗しました: {e}"))
        return

    reply_text = chat_with_makot(user_text, user_id=src_id)
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))


@app.route("/")
def home():
    return "makoT LINE Bot is running!"
