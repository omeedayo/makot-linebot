# ============================================================
# app.py (最終診断用・超シンプル版)
# ============================================================
import os
from flask import Flask, request
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, JoinEvent, LeaveEvent,
    MemberJoinedEvent, MemberLeftEvent, StickerMessage, ImageMessage,
    VideoMessage, AudioMessage
)

# --- 環境変数 ---
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

# --- Flask & LINE Bot setup ---
app = Flask(__name__)
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
webhook_handler = WebhookHandler(LINE_CHANNEL_SECRET)

# --- Webhook endpoint ---
@app.route("/line_webhook", methods=["POST"])
def line_webhook():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)
    try:
        webhook_handler.handle(body, signature)
    except InvalidSignatureError:
        return "Invalid signature", 400
    return "OK", 200

# --- メインのメッセージ処理 ---
@webhook_handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    # このBotは、どんなメッセージが来ても「テスト成功」と返すだけです。
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="グループ参加テスト成功！生きてます！")
    )

# --- グループ参加やスタンプなど、他のイベントはすべて無視 ---
@webhook_handler.add([JoinEvent, LeaveEvent, MemberJoinedEvent, MemberLeftEvent])
def handle_group_events(event):
    pass # 何もしない

@webhook_handler.add(MessageEvent, message=[StickerMessage, ImageMessage, VideoMessage, AudioMessage])
def handle_other_message(event):
    pass # 何もしない

# --- ルートURL ---
@app.route("/")
def home():
    return "シンプルBotが正常に起動中！"
