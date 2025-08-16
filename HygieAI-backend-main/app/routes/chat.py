# app/routes/chat.py
from flask import Blueprint, request, jsonify
from app.utils.qwen_client import summarize_prediction
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

chat_bp = Blueprint('chat', __name__, url_prefix='/api')

# ======= 预加载模型（模块级别一次性加载）=======
MODEL_DIR = Path("data/model")
MODELS = {
    "Bleeding":  MODEL_DIR / "ensemble-learning/ipn_bleeding_ensemble.pkl",
    "Infection": MODEL_DIR / "ensemble-learning/ipn_infection_ensemble.pkl",
    "Outcome":   MODEL_DIR / "ensemble-learning/ipn_blend_bundle.pkl"
}
loaded_models = {}
for name, path in MODELS.items():
    if path.exists():
        loaded_models[name] = joblib.load(path)
    else:
        raise FileNotFoundError(f"❌ 模型文件未找到: {path}")

def feat_names(model):
    if isinstance(model, dict):
        return model["weighted_lgb"].named_steps["prep"].feature_names_in_
    else:
        return model.named_steps["prep"].feature_names_in_

def introduction_reply(data):
    """
    回复机器人功能介绍
    """
    return jsonify({
        "intent": "introduction",
        "message": (
            "您好，我是 HygieAI，是一名智能胰腺医生助手。"
            "我可以：\n"
            "1️⃣ 回答相关的医学问题\n"
            "2️⃣ 接收临床特征，调用模型预测腹腔出血/感染与手术失败风险\n"
            "欢迎随时向我提问哦~我一直都在👌"
        )
    })

def other_reply(data):
    """
    回复与医学无关内容
    """
    return jsonify({
        "intent": "other",
        "message": "⚠️ 很抱歉，可能有神秘力量阻止我理解您的问题。"
    })

def chat_reply(data):
    """
    通义千问问答
    """
    question = data.get("question", "")
    from app.utils.qwen_client import ask_qwen
    answer = ask_qwen(question)

    return jsonify({
        "intent": "chat",
        "message": answer
    })

def prediction_reply(data: dict):
    """
    返回预测类请求的引导语，提示用户输入结构化信息。
    """
    return jsonify({
        "intent": "prediction",
        "message": "🧠 我可以帮助您预测相关风险...请在表单中填写相关信息",
        "require_form": True
    })

@chat_bp.route('/intent', methods=['POST'])
def intent_router():
    """
    主接口：识别 intent，并分发给对应接口处理
    """
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"message": "❗ 请提供字段 question"}), 400

    from app.utils.qwen_client import detect_intent

    intent = detect_intent(question)
    data["intent"] = intent   # 可传递下去用
    print(intent)

    # 动态路由分发
    if intent == "introduction":
        return introduction_reply(data)
    elif intent == "prediction":
        return prediction_reply(data)
    elif intent == "chat":
        return chat_reply(data)
    else:
        return other_reply(data)

@chat_bp.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}

    icd_codes = data.get("icd_codes", [])   # e.g. ["D0001", "D0002", ...]
    features = data.get("features", {})     # 结构化特征，如 {"年龄": 45, "体重": 66, ...}

    if not isinstance(icd_codes, list) or not isinstance(features, dict):
        return jsonify({"error": "格式错误，请传入 icd_codes 列表 和 features 字典"}), 400

    all_cols = sorted({c for m in loaded_models.values() for c in feat_names(m)})
    icd_cols = [c for c in all_cols if c.startswith("icd_")]
    other_cols = [c for c in all_cols if not c.startswith("icd_")]

    # 构建一行输入数据
    row = {}
    for col in other_cols:
        val = features.get(col, None)
        row[col] = np.nan if val is None else float(val)

    for ic in icd_cols:
        row[ic] = 1 if ic.split("icd_")[1] in icd_codes else 0

    df = pd.DataFrame([row], columns=all_cols)

    # 逐模型推理
    result = {}
    for name, model in loaded_models.items():
        feats = feat_names(model)
        if name != "Outcome":
            prob = model.predict_proba(df[feats])[:, 1][0]
        else:
            alpha = model["alpha"]
            thresh = model.get("threshold", 0.5)
            lgbm = model["weighted_lgb"]
            voting = model["undersample_voting"]
            prob_l = lgbm.predict_proba(df[feats])[:, 1][0]
            prob_v = voting.predict_proba(df[feats])[:, 1][0]
            prob = alpha * prob_l + (1 - alpha) * prob_v
        result[name] = round(prob, 4)
    
    message = summarize_prediction(result)
    return jsonify({
        "result": result,
        "message": message
    })