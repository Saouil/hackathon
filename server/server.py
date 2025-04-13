# server.py
import json
import math
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
import langdetect
from RAG import retrieve_fuzzy_rules_context

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
with open("C:/Users/Andrew/Hackaton/fuzzy_inference_system/fuzzy_membership_functions.json") as f:
    membership_data = json.load(f)
with open("C:/Users/Andrew/Hackaton/fuzzy_inference_system/tsk_rules_layer3.json", encoding="utf-8") as f:
    tsk_rules = json.load(f)["rules"]

# === FastAPI App ===
app = FastAPI()
llm = Llama(
    model_path="C:/Users/Andrew/Hackaton/DeepSeek-R1-Distill-Qwen-7B-Q2_K.gguf",
    chat_format="chatml",
    temperature=0.1,
    top_p=0.85,
    repeat_penalty=1.15,
    max_tokens=512,
    n_gpu_layers=16,
    n_ctx=2048
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Fuzzy Rain Evaluation ===
def gaussmf(x, c, sigma):
    return math.exp(-((x - c) ** 2) / (2 * sigma ** 2))

def normalize(value, normalization):
    return (value - normalization["min"]) / (normalization["max"] - normalization["min"])

def evaluate_rain_only_fuzzy_risk(prcp_value):
    feature = "prcp"
    normalization = membership_data["features"][feature]["normalization"]
    norm = normalize(prcp_value, normalization)
    memberships = []
    for mf in membership_data["features"][feature]["membership_functions"]:
        val = gaussmf(norm, mf["center"], mf["sigma"])
        memberships.append(val)
    idx = memberships.index(max(memberships))
    fuzzy_label = ["Low", "Medium", "High"][idx]

    for rule in tsk_rules:
        if rule["IF"].get("prcp") == fuzzy_label:
            eq = rule["THEN"]
            result = eq["intercept"] + eq["coefficients"]["prcp"] * prcp_value
            return round(result, 2), fuzzy_label

    return None, fuzzy_label

# === Chatbot Endpoint ===
@app.post("/chat")
async def chat(req: Request):
    data = await req.json()
    prompt = data["message"]
    lang = langdetect.detect(prompt)

    prcp = 300.0  # Simulated input
    result, fuzzy_label = evaluate_rain_only_fuzzy_risk(prcp)
    rag_context = retrieve_fuzzy_rules_context(prompt, top_k=3)

    if lang.startswith("zh"):
        prompt_final = f"""
你是一名马来西亚天气科学家，专注于下雨预测并使用模糊推理系统。
当前降雨值为: {prcp}，模糊分类为: {fuzzy_label}
预测降雨风险评分为: {result}
相关模糊规则如下:
{rag_context}
用户: {prompt}
专家:"""
    else:
        prompt_final = f"""
You are a Malaysian meteorologist focused on rainfall prediction using a fuzzy inference system.
Current rainfall (prcp) = {prcp}, fuzzy classified as: {fuzzy_label}
Predicted rain risk score: {result}
Related fuzzy rules:
{rag_context}
User: {prompt}
Expert:"""

    output = llm.create_chat_completion(messages=[{"role": "user", "content": prompt_final}])
    return {"response": output["choices"][0]["message"]["content"]}
