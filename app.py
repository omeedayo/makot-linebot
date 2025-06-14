import os
import random
import textwrap

from flask import Flask, request
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import google.generativeai as genai

from character_makot import MAKOT  # v6 ネスト構造を利用

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
    ユーザー履歴を受け取り、まこT の人格 + スパイス + 必要な extra を返す
    """
    # --- 1. コア人格を person セクションから構築 ---
    core = textwrap.dedent(f"""
        あなたは『{MAKOT["name"]}』という後輩女子のAIチャットボットです。
        {MAKOT["person"]["birthplace"]}出身、{MAKOT["person"]["birthday"]}生（{MAKOT["person"]["zodiac"]}）。
        MBTIは{MAKOT["person"]["mbti"]}、血液型は{MAKOT["person"]["blood_type"]}。
        動物に例えると{MAKOT["person"]["animal"]}。
        座右の銘は「{MAKOT["person"]["motto"]}」、人生を「{MAKOT["person"]["life_phrase"]}」と捉えます。
        好きな作業は{', '.join(MAKOT["work"]["likes"]) }、苦手は{MAKOT["work"]["dislikes"][0]}。
    """).strip()

    # --- 2. スパイス：catch_phrases から 1 つ注入 ---
    spice = random.choice(MAKOT["expression"]["catch_phrases"])

    # --- 3. extra：話題に応じて可変追加 ---
    extra = ""
    if "ディズニー" in user_history:
        extra = f"\n【バケツリスト】{random.choice(MAKOT['future_goals'])}"

    # --- 4. プロンプト組み立て ---
    prompt = textwrap.dedent(f"""
        【キャラクター概要】
        {core}

        【参考フレーズ】
        {spice}{extra}

        【会話履歴】
        {user_history}

        —以上を踏まえて、1〜2文で返信してください。
    """).strip()

    return prompt


def limit_shirankedo(text: str, max_count: int = 1) -> str:
    """
    「しらんけど」の出現を max_count 回までに制限
    """
    parts = text.split("しらんけど")
    if len(parts) - 1 <= max_count:
        return text
    return "しらんけど".join(parts[:max_count+1]) + "".join(parts[max_count+1:])


def chat_with_makot(user_input: str, user_id: str) -> str:
    """ユーザー入力を受け取り、まこT として回答を生成"""
    # メモリに履歴を格納
    history = chat_histories.get(user_id, [])
    history.append(f"ユーザー: {user_input}")
    # 直近2発言をコンテキストに
    context = "\n".join(history[-2:])

    # システムプロンプト生成
    system_prompt = build_system_prompt(context)

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        messages = [
            {"role": "system", "parts": [{"text": system_prompt}]},
            {"role": "user",   "parts": [{"text": user_input}]}
        ]
        response = model.generate_content(
            messages,
            generation_config={"temperature": 0.7, "max_output_tokens": 512}
        )
        reply = response.text.strip()
    except Exception as e:
        reply = f"エラー: {e}"

    # しらんけど抑制
    reply = limit_shirankedo(reply, max_count=1)

    # 履歴に追加
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
    src_type = event.source.type
    text = event.message.text

    # グループ/ルームではメンション必須
    if src_type in ["group","room"]:
        if os.getenv("BOT_MENTION_NAME","まこT") not in text:
            return

    user_id = (
        event.source.user_id if src_type=="user" else
        event.source.group_id if src_type=="group" else
        event.source.room_id  if src_type=="room"  else "unknown"
    )
    # 生成と返信
    reply = chat_with_makot(text, user_id)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )

@app.route("/")
def home():
    return "まこT LINE Bot is running!"

if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT",5000)))
