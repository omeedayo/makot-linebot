# ============================================================
# app.py  (v13 – image‑enabled)
# Gemini 2.5 Flash (text)  +  Gemini Image 2.0  +  Imgur upload
# ============================================================

import os
import random
import re
import base64
from io import BytesIO

# 追加 import

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
import google.generativeai as genai

from character_makot import MAKOT, build_system_prompt, apply_expression_style

# ------------------------------------------------------------
# Flask & LINE Bot setup
# ------------------------------------------------------------
app = Flask(__name__)

GEMINI_API_KEY          = os.getenv("GEMINI_API_KEY")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET       = os.getenv("LINE_CHANNEL_SECRET")
IMGUR_CLIENT_ID           = os.getenv("IMGUR_CLIENT_ID")

# --- Gemini client (text + image) ---
genai.configure(api_key=GEMINI_API_KEY, transport="rest")

# --- LINE SDK ---
line_bot_api    = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
webhook_handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ------------------------------------------------------------
# In‑memory simple chat history (per user / group)
# ------------------------------------------------------------
chat_histories: dict[str, list[str]] = {}

# ------------------------------------------------------------
# Helpers: mention / topic / pronoun
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
        return "私"  # normal
    return "マコ" if random.random() < 0.10 else "おに"


def inject_pronoun(reply: str, pronoun: str) -> str:
    return re.sub(r"^(私|おに|マコ)", pronoun, reply, count=1)

# ------------------------------------------------------------
# Post‑process (emoji / しらんけど etc.)
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
# Gemini Image ➜ Imgur upload
# ------------------------------------------------------------


# Gemini Image ➜ Imgur upload
def generate_gemini_image(prompt: str) -> str:
    # Image モデルを呼び出し
    img_model = genai.ImageGenerationModel("image-generation-001")
    # 1枚だけ生成
    result = img_model.generate_images(prompt=prompt, number_of_images=1)
    # SDK が返す URI をそのまま返す
    return result.data[0].uri


# ------------------------------------------------------------
# Main chat logic
# ------------------------------------------------------------

def chat_with_makot(user_input: str, user_id: str) -> str:
    history = chat_histories.get(user_id, [])
    history.append(f"ユーザー: {user_input}")
    context = "\n".join(history[-2:])

    topic = guess_topic(user_input)
    system_prompt = build_system_prompt(context, topic=topic)

    try:
        model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
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

    # グループ / ルームではメンション時のみ応答
    if src_type in ["group", "room"] and not is_bot_mentioned(user_text):
        return

    # ユニーク ID
    src_id = (
        event.source.user_id if src_type == "user" else
        event.source.group_id if src_type == "group" else
        event.source.room_id  if src_type == "room" else "unknown"
    )

    # 画像リクエスト判定
    if any(key in user_text for key in ["画像", "イラスト", "描いて", "絵を"]):
        try:
            img_url = generate_gemini_image(user_text)
            msg = ImageSendMessage(original_content_url=img_url, preview_image_url=img_url)
            line_bot_api.reply_message(event.reply_token, msg)
        except Exception as e:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"画像生成に失敗しました: {e}"))
        return

    # 通常テキスト応答
    reply_text = chat_with_makot(user_text, user_id=src_id)
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))


@app.route("/")
def home():
    return "makoT LINE Bot is running!"

# ------------------------------------------------------------
# END app.py
# ------------------------------------------------------------
