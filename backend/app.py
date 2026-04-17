from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)

# Safe content loader (instead of WebBaseLoader)
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

        message = [
            SystemMessage(content=SystemMessage(content="""You are a study assistant. Only give content according to the subject.
Your task is to generate 10 quiz questions that cover the entire content.
NOTE: The answer should always refer to option1, option2, option3, or option4.
NOTE: The given answer should match one of the options exactly.
The difficulty level should be medium to hard and jumble the options.
Return the response strictly as a JSON array with no markdown, no backticks, and no extra explanation outside the array:

[
  {
    "question_number": "1",
    "question": "...",
    "option1": "...",
    "option2": "...",
    "option3": "...",
    "option4": "...",
    "answer": "option2",
    "explanation": "..."
  }
        ]"""),
            HumanMessage(content=content)
        ]

        response = model.invoke(message)

        return jsonify({"response": response.content})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
