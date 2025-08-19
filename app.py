# ============================================================
# app.py – あだおかLINE Bot (Redis + Pinecone + 画像対応)
# ============================================================

import os
import re
import base64
import requests
import jpholiday
import redis
import google.generativeai as genai
from flask import Flask, request
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageSendMessage
)
from pinecone import Pinecone

# ------------------------------------------------------------
# Flask & LINE Bot setup
# ------------------------------------------------------------
app = Flask(__name__)
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET       = os.getenv("LINE_CHANNEL_SECRET")
line_bot_api              = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
webhook_handler           = WebhookHandler(LINE_CHANNEL_SECRET)

# ------------------------------------------------------------
# Gemini setup
# ------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
text_model = genai.GenerativeModel("gemini-1.5-pro")
image_model = genai.GenerativeModel("imagen-3.0")

# ------------------------------------------------------------
# Redis setup
# ------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    raise ValueError("REDIS_URL が未設定です")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# ------------------------------------------------------------
# Pinecone setup
# ------------------------------------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("adoka-memory")
RAG_SCORE_THRESHOLD = 0.75

# ------------------------------------------------------------
# 会話履歴 + 長期記憶
# ------------------------------------------------------------
def get_user_history(user_id: str):
    return redis_client.lrange(f"history:{user_id}", 0, -1)

def append_user_history(user_id: str, role: str, text: str):
    redis_client.rpush(f"history:{user_id}", f"{role}:{text}")
    redis_client.ltrim(f"history:{user_id}", -20, -1)

def summarize_and_store_memory(user_id: str, history: list):
    """直近の会話を要約して Pinecone に保存"""
    if not history: return
    summary_prompt = f"次の会話から重要な情報を簡潔に1文で要約:\n{history[-5:]}"
    try:
        response = text_model.generate_content(summary_prompt)
        summary = (response.text or "").strip()
        if not summary or "特になし" in summary: return
        pinecone_index.upsert(
            vectors=[{
                "id": f"{user_id}-{len(history)}",
                "values": get_qa_embedding(summary),
                "metadata": {"user": user_id, "text": summary}
            }],
            namespace="user-memories"
        )
    except Exception as e:
        print(f"[Memory保存エラー] {e}")

# ------------------------------------------------------------
# Embedding生成
# ------------------------------------------------------------
def get_qa_embedding(text: str):
    try:
        resp = genai.embed_content(
            model="models/embedding-001",
            content=text
        )
        return resp["embedding"]
    except Exception as e:
        print(f"[Embeddingエラー] {e}")
        return None

# ------------------------------------------------------------
# Q&A処理 (Pinecone検索)
# ------------------------------------------------------------
QA_SYSTEM_PROMPT = """以下の情報を参考に、ユーザーの質問に答えてください。
- 出典を明示すること
- 不明な場合は「分からない」と答えること

【コンテキスト】
{context}

【質問】
{question}
"""

def _handle_qa_request(user_input: str, user_id: str) -> str:
    print(f"[{user_id}] Q&Aモードで実行")
    try:
        queries = [user_input]
        all_matches = {}
        for q in queries:
            vec = get_qa_embedding(q)
            if not vec: continue
            res = pinecone_index.query(
                vector=vec, top_k=3,
                namespace="company-docs", include_metadata=True
            )
            for m in res['matches']:
                mid = m['id']
                if mid not in all_matches or m['score'] > all_matches[mid]['score']:
                    all_matches[mid] = m

        sorted_matches = sorted(all_matches.values(), key=lambda x: x['score'], reverse=True)
        context_chunks, sources = [], set()
        for m in sorted_matches[:5]:
            meta = m['metadata']
            print(f"[検索結果] {m['score']:.3f} {meta['source']}")
            if m['score'] > RAG_SCORE_THRESHOLD:
                context_chunks.append(f"【出典: {meta['source']}】\n{meta['text']}")
                sources.add(meta['source'])

        if not context_chunks:
            return "うーん、その情報は見当たりませんでした…"

        context_str = "\n---\n".join(context_chunks)
        source_str = f"(参考: {', '.join(sorted(list(sources)))})"
        prompt = QA_SYSTEM_PROMPT.format(context=context_str, question=user_input)
        resp = text_model.generate_content(prompt)
        reply = resp.text.strip()
        if "参考:" not in reply: reply += " " + source_str
        return reply

    except Exception as e:
        print(f"[Q&A処理エラー] {e}")
        return "すみません、Q&A処理でエラーが発生しました。"

# ------------------------------------------------------------
# 通常チャット処理
# ------------------------------------------------------------
def _handle_normal_chat(user_input: str, user_id: str) -> str:
    history = get_user_history(user_id)
    context = "\n".join(history[-5:]) if history else ""
    prompt = f"ユーザー: {user_input}\n履歴:\n{context}\n短く自然に返答してください。"
    try:
        resp = text_model.generate_content(prompt)
        reply = resp.text.strip()
    except Exception as e:
        reply = f"[エラー] {e}"
    append_user_history(user_id, "user", user_input)
    append_user_history(user_id, "bot", reply)
    summarize_and_store_memory(user_id, history)
    return reply

# ------------------------------------------------------------
# 画像生成
# ------------------------------------------------------------
def upload_to_imgur(image_bytes: bytes, client_id: str) -> str:
    if not client_id: raise Exception("IMGUR_CLIENT_ID 未設定")
    url = "https://api.imgur.com/3/image"
    headers = {"Authorization": f"Client-ID {client_id}"}
    resp = requests.post(url, headers=headers,
                         data={"image": base64.b64encode(image_bytes).decode("utf-8")})
    resp.raise_for_status()
    data = resp.json()
    if data.get("success"): return data["data"]["link"]
    raise Exception(f"Imgur失敗: {data}")

def generate_image(prompt: str) -> str:
    try:
        resp = image_model.generate_content(prompt)
        b64 = resp.candidates[0].content.parts[0].inline_data.data
        img_bytes = base64.b64decode(b64)
        return upload_to_imgur(img_bytes, os.getenv("IMGUR_CLIENT_ID"))
    except Exception as e:
        print(f"[画像生成エラー] {e}")
        return None

# ------------------------------------------------------------
# LINE Webhook
# ------------------------------------------------------------
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
    user_id = event.source.user_id if event.source.type == "user" else "group"
    text = event.message.text

    if text.startswith("画像:"):
        url = generate_image(text.replace("画像:", "").strip())
        if url:
            line_bot_api.reply_message(
                event.reply_token, ImageSendMessage(original_content_url=url, preview_image_url=url)
            )
        else:
            line_bot_api.reply_message(event.reply_token, TextSendMessage("画像生成に失敗しました"))
        return

    if "？" in text or "教えて" in text:
        reply = _handle_qa_request(text, user_id)
    else:
        reply = _handle_normal_chat(text, user_id)

    line_bot_api.reply_message(event.reply_token, TextSendMessage(reply))

@app.route("/")
def home():
    return "Adoka LINE Bot is running!"
