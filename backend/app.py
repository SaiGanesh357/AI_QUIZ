from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
import requests
from bs4 import BeautifulSoup
import os  

app = Flask(__name__)
CORS(app)

def get_content(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(res.text, "html.parser")
    return soup.get_text()

@app.route("/")
def home():
    return "Server running"

@app.route("/quiz-generator", methods=["POST"])
def QuizGenerator():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        path = data.get("link")
        api_key = data.get("api")

        if not path:
            return jsonify({"error": "Invalid link"}), 400
        if not api_key:
            return jsonify({"error": "Missing API key"}), 400

        content = get_content(path)[:12000]

        model = init_chat_model(
            "groq:llama-3.3-70b-versatile",
            api_key=api_key
        )

        messages = [
            SystemMessage(content="""You are a study assistant that generates quiz questions from provided content.

Generate exactly 10 multiple choice questions covering the entire content provided.

STRICT RULES:
1. The correct answer MUST be randomly distributed across option1, option2, option3, and option4.
   - Do NOT always put the correct answer as option1.
   - Across 10 questions, answers should appear as option1, option2, option3, and option4 roughly equally.
2. The "answer" field must contain the key (e.g. "option3") that holds the correct answer.
3. The correct answer text in the "answer" field must exactly match the text in the corresponding option field.
4. Difficulty should be medium to hard.
5. Wrong options (distractors) should be plausible and closely related to the topic — not obviously wrong.
6. Return ONLY a valid JSON array. No markdown, no backticks, no explanation.

Output format:
[
  {
    "question_number": "1",
    "question": "...",
    "option1": "...",
    "option2": "...",
    "option3": "...",
    "option4": "...",
    "answer": "option3",
    "explanation": "..."
  }
]

Before finalizing, verify that the answers are spread across all four options and not clustered on option1."""),
            HumanMessage(content=content)
        ]

        response = model.invoke(messages)
        return jsonify({"response": response.content})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # ✅ Now os is imported
    app.run(host="0.0.0.0", port=port)
