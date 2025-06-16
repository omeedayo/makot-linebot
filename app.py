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
# 環境変数が設定されていない場合はエラーを出す
if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET:
    raise ValueError("LINEの環境変数が設定されていません。")
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

# --- メインのメッセージ処理（オウム返し） ---
@webhook_handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    # このBotは、どんなメッセージが来てもオウム返しするだけです。
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=event.message.text)
    )

# --- グループ参加やスタンプなど、他のイベントはすべて無視 ---
@webhook_handler.add(JoinEvent)
def handle_join(event):
    pass

@webhook_handler.add(LeaveEvent)
def handle_leave(event):
    pass

@webhook_handler.add(MemberJoinedEvent)
def handle_member_joined(event):
    pass

@webhook_handler.add(MemberLeftEvent)
def handle_member_left(event):
    pass

@webhook_handler.add(MessageEvent, message=[StickerMessage, ImageMessage, VideoMessage, AudioMessage])
def handle_other_message(event):
    pass

# --- ルートURL ---
@app.route("/")
def home():
    return "シンプルBotが正常に起動中！"
