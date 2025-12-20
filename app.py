# ============================================================
# app.py (Gemini画像生成ルートB対応・全文)
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
import datetime
import jpholiday

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
from googleapiclient.discovery import build
import redis
import pinecone
from dotenv import load_dotenv

from character_makot import MAKOT, build_system_prompt, apply_expression_style

# ------------------------------------------------------------
# 初期化
# ------------------------------------------------------------
load_dotenv('.env.development.local')
app = Flask(__name__)

# --- 環境変数 ---
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
TEXT_MODEL_NAME           = os.getenv("TEXT_MODEL_NAME", "gemini-3-flash-preview")
IMAGE_MODEL_NAME          = os.getenv("IMAGE_MODEL_NAME", "nano-banana-pro-preview")
VERTEX_EMBEDDING_MODEL    = os.getenv("VERTEX_EMBEDDING_MODEL", "text-multilingual-embedding-002")
RAG_SCORE_THRESHOLD       = float(os.getenv("RAG_SCORE_THRESHOLD", 0.55))
CRON_SECRET               = os.getenv("CRON_SECRET")
GOOGLE_SEARCH_API_KEY     = os.getenv("GOOGLE_SEARCH_API_KEY")
SEARCH_ENGINE_ID          = os.getenv("SEARCH_ENGINE_ID")

# --- クライアント初期化 ---
genai.configure(api_key=GEMINI_API_KEY, transport="rest")
text_model = genai.GenerativeModel(TEXT_MODEL_NAME)

line_bot_api    = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
webhook_handler = WebhookHandler(LINE_CHANNEL_SECRET)

redis_client = redis.from_url(REDIS_URL, decode_responses=True)
gcp_token_cache = {"token": None, "expires_at": 0}

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

# ------------------------------------------------------------
# Gemini API 画像生成（ルートB）
# ------------------------------------------------------------
def generate_image_with_gemini(prompt: str) -> str:
    trigger_words = ["画像", "イラスト", "描いて", "絵を"]
    clean_prompt = re.sub("|".join(trigger_words), "", prompt).strip()
    if not clean_prompt:
        clean_prompt = "cute anime style illustration"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{IMAGE_MODEL_NAME}:generateContent"
    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [{
            "parts": [{"text": clean_prompt}]
        }],
        "generationConfig": {
            "responseModalities": ["Image"],
            "imageConfig": {
                "aspectRatio": "1:1",
                "imageSize": "1K"
            }
        }
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()

    parts = data["candidates"][0]["content"]["parts"]
    b64 = None
    for p in parts:
        if "inlineData" in p:
            b64 = p["inlineData"]["data"]
            break

    if not b64:
        raise Exception(f"画像が返ってきませんでした: {data}")

    image_bytes = base64.b64decode(b64)
    return upload_to_imgur(image_bytes, IMGUR_CLIENT_ID)

# ------------------------------------------------------------
# Imgur
# ------------------------------------------------------------
def upload_to_imgur(image_bytes: bytes, client_id: str) -> str:
    url = "https://api.imgur.com/3/image"
    headers = {"Authorization": f"Client-ID {client_id}"}
    response = requests.post(
        url,
        headers=headers,
        data={"image": base64.b64encode(image_bytes)}
    )
    response.raise_for_status()
    return response.json()["data"]["link"]

# ------------------------------------------------------------
# Webhook
# ------------------------------------------------------------
@app.route("/line_webhook", methods=["POST"])
def line_webhook():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)
    webhook_handler.handle(body, signature)
    return "OK", 200

# ------------------------------------------------------------
# メッセージ処理
# ------------------------------------------------------------
@webhook_handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_id = event.source.user_id
    redis_client.sadd("users", user_id)
    user_text = event.message.text

    if any(k in user_text for k in ["画像", "イラスト", "描いて", "絵を"]):
        try:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="おっけーです！ちょっと待っててくださいね…🥰")
            )
            img_url = generate_image_with_gemini(user_text)
            msg = ImageSendMessage(
                original_content_url=img_url,
                preview_image_url=img_url
            )
            line_bot_api.push_message(user_id, msg)
        except Exception as e:
            line_bot_api.push_message(
                user_id,
                TextSendMessage(text=f"ごめんなさい、画像生成の調子が悪いみたいです…\n理由: {e}")
            )
        return

    reply_text = chat_with_makot(user_text, user_id)
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

# ------------------------------------------------------------
@app.route("/")
def home():
    return "まこT LINE Bot is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
