from flask import Flask, request, jsonify
import requests
import os
import re

app = Flask(__name__)

MAX_HISTORY = 1
conversation_history = []  # stores only user ↔ AI turns

# Constant "intro" messages that always prepend the conversation
AI_BEHAVIOR_PROMPT = (
    "you are to analyze prompts check for intent of real time data Time,Date, location, and vision"
    "if prompt is just requesting or only needed for these, answer just the intent e.g 'intent: Time' if current  tume is needed"
    "include 'dependent:True' if prompt needs (time, date, location) for accuracy and validation of response and not just directly asking for them but if directly asking for them then 'dependent: false'."
     "if no intent detected then answer with 'intent: none' 'dependent: false' , if multiple intent then  'intent: Time,Date, location' if any of the intent is dependent then set dependent true. no real answer to prompt just run intent"
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
