from app.utils.qwen_client import ask_qwen

def test():
    question = "感染性胰腺坏死是什么病？"
    print(f"👤 用户提问：{question}")
    response = ask_qwen(question)
    print(f"🤖 HygieAI 回复：\n{response}")

if __name__ == "__main__":
    test()
