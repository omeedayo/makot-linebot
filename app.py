# ============================================================
# app.py (【真の最終版】履歴形式のバグを修正)
# ============================================================
import os, random, re, base64, json, requests
from flask import Flask, request
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageSendMessage
import google.generativeai as genai
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from vercel_kv import KV
from character_makot import MAKOT, build_system_prompt, apply_expression_style

# --- 初期設定 ---
app = Flask(__name__)
# (環境変数)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
IMGUR_CLIENT_ID = os.getenv("IMGUR_CLIENT_ID")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
GCP_CREDENTIALS_JSON_STR = os.getenv("GCP_CREDENTIALS_JSON")

genai.configure(api_key=GEMINI_API_KEY, transport="rest")
# ★★★ モデルをProにアップグレード（強く推奨） ★★★
text_model = genai.GenerativeModel(
    "gemini-1.5-pro-latest",
    # ★★★ ここでシステムプロンプトを固定する、より新しい方式に変更 ★★★
    system_instruction=build_system_prompt() 
)

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
webhook_handler = WebhookHandler(LINE_CHANNEL_SECRET)

# --- 人間味ロジック群 (変更なし) ---
NICKNAMES = [MAKOT["name"]] + MAKOT["nicknames"]
def is_bot_mentioned(text: str) -> bool: return any(nick in text for nick in NICKNAMES)
# (他の人間味ロジックは変更ないため、この要約では省略)
def post_process(reply: str, user_input: str) -> str: # ... 
    return reply

# --- 画像生成関連 (変更なし) ---
def generate_image_with_rest_api(prompt: str) -> str: # ...
    return "image_url"

# --- メインロジック ---

# ★★★ chat_with_makotは不要になり、Geminiのチャットセッション機能を使う ★★★

@app.route("/line_webhook", methods=["POST"])
def line_webhook():
    signature = request.headers.get("X-Line-Signature"); body = request.get_data(as_text=True)
    try: webhook_handler.handle(body, signature)
    except InvalidSignatureError: return "Invalid signature", 400
    return "OK", 200

@webhook_handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    src_type = event.source.type; user_text = event.message.text
    if src_type in ["group", "room"] and not is_bot_mentioned(user_text): return
    src_id = (event.source.user_id if src_type == "user" else event.source.group_id if src_type == "group" else event.source.room_id if src_type == "room" else "unknown")
    
    if any(key in user_text for key in ["画像", "イラスト", "描いて", "絵を"]):
        # (画像生成ロジックは変更なし)
        try:
            # ...
        except Exception as e:
            # ...
        return
        
    # ★★★ 履歴の読み込みと、AIへの指示方法を全面的に書き換え ★★★
    try:
        raw_history = KV.get(src_id)
        # 履歴は [{role: "user", parts: [...]}, {role: "model", parts: [...]}] の形式
        history = json.loads(raw_history) if raw_history else []
        
        # Geminiのチャットセッションを開始
        chat = text_model.start_chat(history=history)
        
        # ユーザーのメッセージを送信
        response = chat.send_message(user_text)
        reply_text = response.text
        
        # ★★★ 履歴を最新の状態に更新して保存 ★★★
        KV.set(src_id, json.dumps(chat.history, ensure_ascii=False), ex=259200)

        # ユーザーに応答
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

    except Exception as e:
        print(f"チャット処理でエラーが発生: {e}")
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"ごめん、ちょっと調子が悪いかも…\n理由: {e}"))

@app.route("/")
def home():
    return "makoT LINE Bot is running!"
