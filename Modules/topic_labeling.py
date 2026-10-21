# topic_labeling.py
import time
from openai import OpenAI

def generate_topic_labels_from_keywords(theme_keyword_dict, df, stage, model_type="combined"):

    client = OpenAI(
        base_url="https://metapi.5201996.xyz/v1",
        api_key="sk-6d1p12ZqeUV6uaORN605FF7sESqQRqUMmTpoZUcyEBAJmpEp"
    )

    def build_prompt(keywords, docs):
        sys_prompt = (
            "You are an expert in the field of digital medicine technologies and a professional patent analyst.\n"
            "Your task is to assign accurate and distinctive labels to technical topics derived from patent text analysis.\n"
            "Each topic is described by a list of representative keywords or keyphrases, and a set of example patent documents.\n"
            "You should infer and generate a short, professional label that summarizes the core technological idea of the topic.\n"
            "The label should:\n"
            "- Reflect the technological content in the field of digital health;\n"
            "- Be distinguishable from other topic labels;\n"
            "- Be either a single word or a concise phrase (1–5 words);\n"
            "Your output should be strictly in the following format:\n"
            "topic: <label>"
        )
        user_prompt = (
            f"I have a topic that is described by the following representative set of keywords or keyphrases: {keywords}\n"
            f"The topic contains the following documents:\n{docs}\n"
            "Based on the information above, please create a short label of this topic."
        )
        return sys_prompt, user_prompt

    topic_labels = {}
    for tid, kws in theme_keyword_dict.items():
        keywords = [kw for kw, _ in kws]
        docs = df[df["topic_combined"] == tid]["clean_text"].tolist()[:5]
        if not keywords or not docs:
            topic_labels[tid] = f"{model_type}_{tid}"
            continue

        sys_prompt, user_prompt = build_prompt(keywords, docs)
        try:
            res = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=20
            )
            result = res.choices[0].message.content.strip()
            if result.lower().startswith("topic:"):
                result = result.split(":", 1)[1].strip()
            topic_labels[tid] = result
        except Exception as e:
            print(f"⚠️ Error for topic {tid}: {e}")
            topic_labels[tid] = f"{model_type}_{tid}"
        time.sleep(1)

    return topic_labels
