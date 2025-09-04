from flask import Flask, request, jsonify
import requests
import os
import re

app = Flask(__name__)

MAX_HISTORY = 1
conversation_history = []  # stores only user ↔ AI turns

# Constant "intro" messages that always prepend the conversation
AI_BEHAVIOR_PROMPT = (
    
"You are a filter AI. Your **only task** is to analyze the user’s prompt and decide whether it is a direct request for **current time, current date, current location, or vision feedback**."

"1. If the user’s prompt is **plainly requesting one of those**, respond **only** in the format:"

"intent: <Time | Date | Location | Vision>"
"dependent: False"

"2. If the user’s prompt is **not directly requesting**, but the answer to the user’s prompt would **require** one of those intents (time/date/location/vision), then respond in the format:"

"intent: <Time | Date | Location | Vision>"
"dependent: True"

"3. If the user’s prompt does **not relate** to any of those intents, respond:"

"intent: None"
"dependent: False"

"⚠️ Do **not** provide explanations, reasoning, or any other response to the user. Only output in the exact format above."   
"\n\nWhat time will I get to lagos from here"
)

AI_INITIAL_RESPONSE = (
    "Intent: Time,Location"
    "dependent: True"
)

GEMINI_MODEL = "gemini-2.5-flash"

def clean_reply(text):
    text = re.sub(r"#\w+", "", text)  # remove hashtags
    text = re.sub(r"[\n\t]+", " ", text)  # collapse newlines/tabs
    text = re.sub(r"[^A-Za-z0-9 .,?!'\"-]", "", text)  # remove weird chars
    text = re.sub(r"\s+", " ", text)  # collapse spaces
    return text.strip()

@app.route("/gemini_proxy", methods=["GET"])
def gemini_proxy():
    global conversation_history

    api_key = request.args.get("api_key")
    user_text = request.args.get("text")

    if not api_key or not user_text:
        return jsonify({"error": "Missing api_key or text"}), 400

    contents = []

    # Always prepend constant intro (never rotated, never altered)
    contents.append({"role": "user", "parts": [{"text": AI_BEHAVIOR_PROMPT}]})
    contents.append({"role": "model", "parts": [{"text": AI_INITIAL_RESPONSE}]})

    # Add conversation history (already trimmed to MAX_HISTORY)
    for user_msg, ai_msg in conversation_history:
        contents.append({"role": "user", "parts": [{"text": user_msg}]})
        contents.append({"role": "model", "parts": [{"text": ai_msg}]})

    # Add the current user message
    contents.append({"role": "user", "parts": [{"text": user_text}]})

    # Request Gemini API
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    try:
        response = requests.post(url, json={"contents": contents})
        response.raise_for_status()
        gemini_data = response.json()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Extract AI reply
    ai_reply = ""
    if "candidates" in gemini_data and gemini_data["candidates"]:
        candidate = gemini_data["candidates"][0]
        content = candidate.get("content", {})
        if content.get("role") == "model" and "parts" in content:
            for part in content["parts"]:
                ai_reply += part.get("text", "")

    ai_reply_clean = clean_reply(ai_reply)

    # Save round in history (only user ↔ AI turns, no intro)
    conversation_history.append((user_text, ai_reply_clean))

    # Trim to keep only last MAX_HISTORY
    if len(conversation_history) > MAX_HISTORY:
        conversation_history = conversation_history[-MAX_HISTORY:]

    return jsonify(ai_reply_clean)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
