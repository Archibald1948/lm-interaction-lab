"""BERT(SQuAD) 질문 응답 모델을 Flask 웹 페이지로 띄운다.

QA_CONTEXT=epl PORT=5001 python qa_server.py  ->  http://127.0.0.1:5001
macOS는 5000번 포트를 AirPlay가 쓰므로 다른 포트를 쓴다.
"""
import os

from flask import Flask, request, jsonify
from transformers import pipeline

# Flask 앱 생성
app = Flask(__name__)

# Question Answering 파이프라인 로드
qa_pipeline = pipeline("question-answering", model="google-bert/bert-large-uncased-whole-word-masking-finetuned-squad")

# 예제 컨텍스트 (질문에 대한 답변이 추출될 텍스트)
CONTEXT = """
Artificial Intelligence (AI) is a field of computer science that aims to create systems capable of performing tasks
that normally require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.
Recent advances in AI have enabled applications such as autonomous vehicles, personalized recommendations, and conversational agents.
"""

EPL_CONTEXT = """
The English Premier League (EPL) is widely regarded as one of the most competitive and exciting football leagues in the world. Established in 1992, the league consists of 20 teams that compete annually for the championship title. Over the years, clubs like Manchester United, Liverpool, Chelsea, and Manchester City have become household names due to their success in domestic and international competitions. The league's global popularity is bolstered by its diverse pool of players, with athletes from more than 100 countries participating. Broadcasting deals have played a significant role in the EPL's financial growth, with the league securing multi-billion-dollar agreements with networks worldwide, allowing fans from every corner of the globe to watch matches live. The league operates on a promotion and relegation system, where the bottom three teams are relegated to the English Football League Championship, and the top three teams from the Championship are promoted to the EPL. Beyond football, the EPL has a profound cultural and economic impact, generating billions of pounds annually and supporting thousands of jobs across the UK. Despite its success, the league faces challenges such as financial disparity among clubs, fixture congestion, and debates over the implementation of technology like VAR (Video Assistant Referee). Nevertheless, the EPL remains a cornerstone of global football culture.

"""

# QA_CONTEXT=epl 로 실행하면 EPL 글을 지문으로 쓴다
if os.environ.get("QA_CONTEXT") == "epl":
    CONTEXT = EPL_CONTEXT


# HTML 템플릿 (간단한 웹페이지 생성)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>질문 응답 시스템</title>
</head>
<body>
    <h1>질문 응답 시스템</h1>
    <div>
        <p><b>컨텍스트:</b> {{ context }}</p>
    </div>
    <div id="qa-box">
        <form id="qaForm" action="/answer" method="POST">
            <input type="text" name="user_question" id="user_question" placeholder="질문을 입력하세요" required>
            <button type="submit">답변 요청</button>
        </form>
        <div id="answer-box"></div>
    </div>
    <script>
        const form = document.getElementById("qaForm");
        const answerBox = document.getElementById("answer-box");

        form.onsubmit = async function (e) {
            e.preventDefault();
            const userQuestion = document.getElementById("user_question").value;
            const response = await fetch("/answer", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ user_question: userQuestion }),
            });
            const result = await response.json();
            const answerHTML = `<p><b>질문:</b> ${userQuestion}</p><p><b>답변:</b> ${result.answer}</p>`;
            answerBox.innerHTML = answerHTML;
            document.getElementById("user_question").value = "";
        };
    </script>
</body>
</html>
"""

# 루트 페이지: HTML 반환
@app.route("/")
def home():
    return HTML_TEMPLATE.replace("{{ context }}", CONTEXT)

# 질문 응답 처리
@app.route("/answer", methods=["POST"])
def answer():
    # 사용자 질문 받기
    user_question = request.json.get("user_question", "")

    # Question Answering 모델 호출
    result = qa_pipeline(question=user_question, context=CONTEXT)

    # 응답 반환
    return jsonify({"answer": result['answer']})

# Flask 앱 실행
if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=int(os.environ.get("PORT", 5000)))
