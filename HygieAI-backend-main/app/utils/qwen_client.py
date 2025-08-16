# app/utils/qwen_client.py
from dotenv import load_dotenv
load_dotenv()

import os, dashscope, logging
dashscope.api_key = os.getenv("QWEN_API_KEY", "").strip()

logger = logging.getLogger(__name__)

# ========= 1. 对话回答 =========
def ask_qwen(question: str) -> str:
    """
    调通义千问 Qwen-Turbo，返回纯文本答案
    """
    if not dashscope.api_key:
        return "⚠️ 服务器未配置 Qwen API Key。"

    try:
        rsp = dashscope.Generation.call(
            model="qwen-turbo",
            messages=[
                {"role": "system",
                 "content": "你是一名消化内科医生助手，专长感染性胰腺坏死等胰腺疾病。"
                            "请用专业、准确、简短的中文回答用户提出的医学相关问题。"},
                {"role": "user", "content": question},
            ],
            result_format="message",
            temperature=0.7,
            max_tokens=512,
        )
        return rsp["output"]["choices"][0]["message"]["content"]
    except Exception as e:
        logger.exception("Call Qwen failed")
        return f"⚠️ 通义千问调用失败：{e}"

# ========= 2. 意图分类 =========
_INTENT_SYS_PROMPT = (
    "你是一名意图分类模型，只需回答四个 label 之一："
    "`introduction`, `prediction`, `chat`, `other`。\n"
    "定义：\n"
    "- introduction：用户想了解你的功能或怎么用你。\n"
    "- prediction  ：用户想输入临床特征，让你调用本地模型预测风险。\n"
    "- chat        ：一般医学/胰腺病学知识问答。\n"
    "- other       ：任何与医学无关或你应拒答的内容。\n"
    "只输出 label 本身，不要额外解释。"
)

def detect_intent(question: str) -> str:
    """
    把自然语言问题分类为四种 intent 之一。
    出错时返回 'other' 作为保底。
    """
    if not dashscope.api_key:
        return "other"

    try:
        rsp = dashscope.Generation.call(
            model="qwen-turbo",
            messages=[
                {"role": "system", "content": _INTENT_SYS_PROMPT},
                {"role": "user",   "content": question},
            ],
            result_format="message",
            temperature=0.0,        # 关闭随机性，保证可复现
            max_tokens=5,
        )
        label = rsp["output"]["choices"][0]["message"]["content"].strip().lower()
        if label in {"introduction", "prediction", "chat", "other"}:
            return label
        return "other"             # LLM 偶尔跑偏时兜底
    except Exception as e:
        logger.warning(f"Intent detect failed: {e}")
        return "other"

def summarize_prediction(prob_dict: dict) -> str:
    """
    给定 Bleeding / Infection / Outcome 三个概率值，调用 Qwen 生成语义化的中文解读。
    """
    if not dashscope.api_key:
        return "⚠️ 服务器未配置 Qwen API Key。"

    summary_prompt = (
        "你是一名消化内科医生助手，现有一模型针对可能的一场胰腺手术预测了三个风险概率。请根据这三个风险，生成一段专业的中文解读，"
        "内容面向患者或非专业医护人员，避免生硬术语，建议分点描述，并可给出初步建议。\n\n"
        "以下是模型预测结果：\n"
        f"- 腹腔出血风险 (Bleeding): {prob_dict.get('Bleeding', 'N/A'):.4f}\n"
        f"- 腹腔感染风险 (Infection): {prob_dict.get('Infection', 'N/A'):.4f}\n"
        f"- 胰腺手术失败或者不良结局风险 (Outcome): {prob_dict.get('Outcome', 'N/A'):.4f}\n\n"
        "请据此进行解读（特别注意，Outcome值大于0.2的时候就有一定风险）："
    )

    try:
        rsp = dashscope.Generation.call(
            model="qwen-turbo",
            messages=[
                {"role": "system", "content": "你是一名临床医生助手，擅长患者沟通与风险评估。"},
                {"role": "user", "content": summary_prompt}
            ],
            result_format="message",
            temperature=0.7,
            max_tokens=512,
        )
        return rsp["output"]["choices"][0]["message"]["content"]
    except Exception as e:
        logger.exception("Call Qwen for prediction summary failed")
        return f"⚠️ 风险解读失败：{e}"
