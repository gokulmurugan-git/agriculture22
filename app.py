import os
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from google import genai

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.1-flash-lite"

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured in the .env file.")

client = genai.Client(api_key=GEMINI_API_KEY)

with open(BASE_DIR / "chatbot_config", "r", encoding="utf-8") as file:
    SYSTEM_PROMPT = file.read().strip()


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"reply": "Please enter an agriculture-related question."}), 400

    try:
        prompt = f"{SYSTEM_PROMPT}\n\nUSER QUESTION:\n{message}"

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        reply = (response.text or "").strip()

        if not reply:
            reply = "I could not generate a response. Please ask an agriculture-related question."

        return jsonify({"reply": reply})

    except Exception as error:
        print("Gemini API error:", error)
        return jsonify({
            "reply": "Sorry, I could not process your question right now. Please try again."
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
