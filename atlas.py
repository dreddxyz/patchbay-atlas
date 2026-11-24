"""Compare a lexical baseline with a portable embedding model on a studio manual."""
import argparse
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from qdrant_client import QdrantClient, models
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM

ENC = ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
GEN = ("google/flan-t5-small", "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab")
OUT = Path("build")
SITE = Path("site")


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        mean = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return torch.nn.functional.normalize(mean, dim=1)


def load_tokenizer(pair):
    return AutoTokenizer.from_pretrained(pair[0], revision=pair[1])


def tokens(tokenizer, texts):
    return tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)


def load_runtime():
    return ort.InferenceSession(str(OUT/"encoder.onnx"), providers=["CPUExecutionProvider"])


def embed(session, tokenizer, texts):
    batch = tokens(tokenizer, texts)
    return session.run(None, {k:batch[k].numpy() for k in ("input_ids","attention_mask")})[0]


def top(db, vector):
    rows = db.search("manual", vector.tolist(), limit=3)
    return [{"id":r.payload["id"], "score":round(float(r.score),4)} for r in rows]


def metrics(results, method):
    ranks = [next((i for i,item in enumerate(r[method],1) if item["id"] == r["expected"]),0) for r in results]
    return {"top1":sum(v==1 for v in ranks)/len(ranks), "recall3":sum(v>0 for v in ranks)/len(ranks),
            "mrr3":sum(1/v if v else 0 for v in ranks)/len(ranks)}


def generate_answer(query, card, tokenizer, model):
    prompt = f"Read the passage. Answer the question in one sentence using the passage. Passage: {card['text']} Question: {query}"
    with torch.inference_mode():
        result = model.generate(**tokenizer(prompt,return_tensors="pt"), max_new_tokens=50, do_sample=False)
    return tokenizer.decode(result[0], skip_special_tokens=True)


def build():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    cards = json.loads(Path("cards.json").read_text())
    evaluation = json.loads(Path("evaluation.json").read_text())
    ids = [c["id"] for c in cards]
    assert len(set(ids)) == len(ids) and all(e["expected"] in ids for e in evaluation)
    texts = [c["title"]+". "+c["text"] for c in cards]
    queries = [e["query"] for e in evaluation]
    tokenizer = load_tokenizer(ENC)
    model = Encoder(AutoModel.from_pretrained(ENC[0],revision=ENC[1])).eval()
    batch = tokens(tokenizer, texts[:2])
    names = ["input_ids","attention_mask"]
    torch.onnx.export(model, tuple(batch[k] for k in names), str(OUT/"encoder.onnx"),
                      input_names=names, output_names=["embedding"], opset_version=17,
                      dynamic_axes={**{k:{0:"batch",1:"sequence"} for k in names},"embedding":{0:"batch"}})
    onnx.checker.check_model(str(OUT/"encoder.onnx"))
    session = load_runtime()
    vectors = embed(session, tokenizer, texts)
    probe = tokens(tokenizer,texts+queries)
    with torch.inference_mode():
        reference = model(*(probe[k] for k in names)).numpy()
    actual = embed(session, tokenizer, texts+queries)
    parity = float(np.max(np.abs(reference-actual)))
    np.testing.assert_allclose(actual, reference, atol=2e-5, rtol=1e-4)
    db = QdrantClient(path=str(OUT/"qdrant"))
    if db.collection_exists("manual"):
        db.delete_collection("manual")
    db.create_collection("manual", vectors_config=models.VectorParams(size=vectors.shape[1],distance=models.Distance.COSINE))
    db.upsert("manual", points=[models.PointStruct(id=i,vector=v.tolist(),payload={"id":ids[i]}) for i,v in enumerate(vectors)])
    vectorizer = TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)
    lex = vectorizer.fit_transform(texts)
    scores = (vectorizer.transform(queries) @ lex.T).toarray()
    results=[]
    for e,vec,row in zip(evaluation,actual[len(texts):],scores):
        baseline = [{"id":ids[i],"score":round(float(row[i]),4)} for i in np.argsort(-row, kind="stable")[:3]]
        results.append({**e, "tfidf":baseline, "dense":top(db,vec)})
    db.close()
    gen_tok = load_tokenizer(GEN)
    gen = AutoModelForSeq2SeqLM.from_pretrained(GEN[0],revision=GEN[1]).eval()
    by_id = {c["id"]:c for c in cards}
    for r in results:
        source = by_id[r["dense"][0]["id"]]
        r["draft"] = generate_answer(r["query"],source,gen_tok,gen)
    data = {"cards":by_id,"results":results,"metrics":{m:metrics(results,m) for m in ["tfidf","dense"]},
            "max_abs_error":parity,"encoder":ENC,"generator":GEN}
    (SITE/"results.json").write_text(json.dumps(data,indent=2),encoding="utf-8")
    print(json.dumps(data["metrics"]), "ONNX max error", parity)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--query",help="Search the built local index with your own English question")
    args=p.parse_args()
    if args.query:
        if not (OUT/"encoder.onnx").exists():
            p.error("Build first with python atlas.py")
        db = QdrantClient(path=str(OUT/"qdrant"))
        print(json.dumps(top(db, embed(load_runtime(),load_tokenizer(ENC),[args.query])[0]),indent=2))
        db.close()
    else:
        build()


if __name__ == "__main__":
    main()
