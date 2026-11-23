"""웹 화면에는 보이지 않는 답변 신뢰도(score)를 질문별로 출력한다.

python qa_scores.py
"""
import qa_server

QUESTIONS = [
    "When was the English Premier League (EPL) established?",
    "Which teams are mentioned as being successful in both domestic and international competitions?",
    "What system does the EPL use to manage promotion and relegation?",
    "What are some of the challenges currently faced by the EPL?",
    "How has the EPL achieved global popularity?",
    "Who won the EPL in 2020?",          # 지문에 없는 질문
    "EPL은 몇 년에 만들어졌어?",          # 한국어 질문
]

for q in QUESTIONS:
    r = qa_server.qa_pipeline(question=q, context=qa_server.EPL_CONTEXT)
    print(f"{r['score']:.3f} | {q} -> {r['answer']}")
