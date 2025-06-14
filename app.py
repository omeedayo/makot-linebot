import os
import random
import textwrap

from flask import Flask, request
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import google.generativeai as genai

from character_makot import MAKOT  # ← v5 キャラクター定義を利用

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

# ----------- ユーティリティ -----------
def build_system_prompt(user_history: str) -> str:
    """
    user_history: 直近のユーザー発言ログを改行区切りで受け取る。
    persona + ランダム抽出ルール + スパイスフレーズ + 必要に応じた extra情報 を組み立て。
    """
    # 1) コアプロファイル
    core = MAKOT["persona"].strip()

    # 2) 行動ルールから6つランダムピック
    rules = random.sample(MAKOT["behavior_rules"], k=6)
    rules_text = "\n".join(f"・{r}" for r in rules)

    # 3) 語録スパイス
    spice = random.choice(MAKOT["catch_phrases"])

    # 4) キーワードに応じた追加情報
    extra = ""
    if "ディズニー" in user_history:
        extra = f"\n【夢・目標】{random.choice(MAKOT['future_goals'])}"

    prompt = textwrap.dedent(f"""
        【キャラクター概要】
        {core}

        【振る舞いルール】（ランダム抽出）
        {rules_text}

        【参考フレーズ】
        {spice}{extra}

        【会話履歴】
        {user_history}

        —以上を踏まえて、1〜2文で返信してください。
    """).strip()

    return prompt



def chat_with_makot(user_input: str, user_id: str) -> str:
    """Gemini に問い合わせて返答取得"""
    history = chat_histories.get(user_id, [])
    history.append(f"ユーザー: {user_input}")
    context = "\n".join(history[-2:])

    system_prompt = build_system_prompt(context)

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

    # グループ・ルームではメンション (@まこT) がある時のみ応答
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
