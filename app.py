import os
import random
import textwrap

from flask import Flask, request
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import google.generativeai as genai

from character_makot import MAKOT  # v5.4 定義

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

# ---------- システムプロンプト生成 ----------
def build_system_prompt(context: str) -> str:
    """
    1) persona
    2) behavior_rules
    3) catch_phrases からランダム1行
    をまとめた要約版プロンプト
    """
    persona = MAKOT["persona"]
    rules   = "\n".join(f"・{r}" for r in MAKOT["behavior_rules"])
    spice   = random.choice(MAKOT["catch_phrases"])

    prompt = textwrap.dedent(f"""
        {persona}

        ■振る舞いルール
        {rules}

        ■参考フレーズ（任意で使用）
        {spice}

        ■会話履歴
        {context}

        ユーザーの問いかけに 1～2 文で自然に返答してください：
    """).strip()
    return prompt

# ---------- チャット処理 ----------
def chat_with_makot(user_input: str, user_id: str) -> str:
    # 履歴を保持し直近2行を context に
    history = chat_histories.get(user_id, [])
    history.append(f"ユーザー: {user_input}")
    context = "\n".join(history[-2:])

    system_prompt = build_system_prompt(context)

    try:
        model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")  # トークン枠広いモデル推奨
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_input}
        ]
        response = model.generate_content(
            messages,
            generation_config={"temperature": 0.7, "max_output_tokens": 512}
        )
        reply = response.text.strip()
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

    # グループ/ルームではメンション必須
    if src_type in ["group", "room"]:
        bot_name = os.getenv("BOT_MENTION_NAME", "まこT")
        if bot_name not in user_text:
            return

    src_id = (
        event.source.user_id  if src_type == "user"  else
        event.source.group_id if src_type == "group" else
        event.source.room_id  if src_type == "room"  else "unknown"
    )

    reply_text = chat_with_makot(user_text, user_id=src_id)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )

@app.route("/")
def home():
    return "まこT LINE Bot is running!"
