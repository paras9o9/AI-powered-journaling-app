from transformers import pipeline

LABEL2ID = {"NEU": 0, "HUMOR": 1, "MH": 2, "SI": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def load_model(model_path, tokenizer_name=None):
    clf = pipeline(
        "text-classification",
        model=model_path,
        top_k=None,
        device=-1
    )
    return None, clf

def predict_entry(text, tokenizer, model, max_length=256):
    results = model(text, truncation=True, max_length=max_length)[0]
    prob_dict = {r["label"]: round(r["score"], 4) for r in results}
    pred_label = max(prob_dict, key=prob_dict.get)
    return pred_label, prob_dict
