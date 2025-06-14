import os
import random
import textwrap

from flask import Flask, request
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import google.generativeai as genai

from character_makot import MAKOT, build_system_prompt  # ← 追加 import

app = Flask(__name__)

# ---------- Gemini API ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY, transport="rest")

# ---------- LINE Bot ----------
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET       = os.getenv("LINE_CHANNEL_SECRET")
line_bot_api              = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
webhook_handler           = WebhookHandler(LINE_CHANNEL_SECRET)

# ---------- 簡易メモリ ----------
chat_histories = {}

# ---------- トピック判定ヘルパ ----------

def guess_topic(text: str):
    """ごく簡単なキーワードマッチで hobby / work を返す"""
    hobby_keys = ["趣味", "休日", "ハマって", "コストコ", "ポケポケ"]
    work_keys  = ["仕事", "業務", "残業", "請求書", "統計"]
    if any(k in text for k in hobby_keys):
        return "hobby"
    if any(k in text for k in work_keys):
        return "work"
    return None

# ---------- チャットメイン ----------

def chat_with_makot(user_input: str, user_id: str) -> str:
    history = chat_histories.get(user_id, [])
    history.append(f"ユーザー: {user_input}")
    context = "\n".join(history[-2:])  # 直近2ターン

    topic = guess_topic(user_input)               # ★ 追加
    system_prompt = build_system_prompt(context, topic=topic)

    try:
        model    = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
        response = model.generate_content(system_prompt)
        reply    = response.text.strip()
    except Exception as e:
        reply = f"エラーが発生しました: {e}"

    history.append(reply)
    chat_histories[user_id] = history
    return reply

# ---------- Flask エンドポイント ----------

@app.route("/line_webhook", methods=["POST"])
def line_webhook():
    signature = request.headers.get("X-Line-Signature")
    body      = request.get_data(as_text=True)
    try:
        webhook_handler.handle(body, signature)
    except InvalidSignatureError:
        return "Invalid signature", 400
    return "OK", 200


@webhook_handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    src_type  = event.source.type
    user_text = event.message.text

    # グループ/ルームではメンション時のみ応答
    if src_type in ["group", "room"]:
        bot_name = os.getenv("BOT_MENTION_NAME", "まこT")
        if bot_name not in user_text:
            return

    # ID 決定
    src_id = (
        event.source.user_id if src_type == "user" else
        event.source.group_id if src_type == "group" else
        event.source.room_id  if src_type == "room" else "unknown"
    )

    reply_text = chat_with_makot(user_text, user_id=src_id)

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )


@app.route("/")
def home():
    return "まこT LINE Bot is running!"

# ------------------------------------------------------------
# END app.py
# ------------------------------------------------------------
