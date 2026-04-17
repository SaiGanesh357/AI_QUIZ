from flask import Flask,request,jsonify
from flask_cors import CORS
from langchain_community.document_loaders import WebBaseLoader
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
app=Flask(__name__)
CORS(app)

@app.route("/quiz-generator",methods=["POST"])
def QuizGenerator():
    data = request.get_json()
    path = data.get("link")
    api_key = data.get("api")
    loader = WebBaseLoader(path)
    data = loader.load()
    content = data[0].page_content[:11999]
    model = init_chat_model(
    "groq:llama-3.3-70b-versatile",
    api_key = api_key
    )
    message = [
    SystemMessage(content="""You are a study assistant.only give the content according to subject
                   Your task is to generate 10 quiz questions that cover the entire content.
                  NOTE: the answer Should be always refers option1 ,option2,option3,option4 etc....
                  NOTE: the given answer should match to the on of the option exactly

    Return the response strictly in this JSON list format: The exam should be hard
    [
    {
    "question_number": 1,
    "question": "...",
    "option1": "...",
    "option2": "...",
    "option3": "...",
    "option4": "...",
    "answer": "...",
    "explanation": "..."
    }
                  
    ]
    """),
    HumanMessage(content=content)
    ]
    try:
       response = model.invoke(message)
    except:
        return jsonify({"error":"Unable to fetch"})
    return jsonify({
        "response":response.content
    })
app.run(debug=True)