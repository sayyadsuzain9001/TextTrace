"""Prepare a candidate adaptive model from reviewed feedback.

This script intentionally requires a minimum number of reviewed examples per class.
It trains a candidate artifact only; it never overwrites the production ensemble.
Use only feedback whose labels have been verified by an authorised reviewer.
"""
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import joblib

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/"feedback"/"reviewed_feedback.jsonl"
OUT=ROOT/"models"/"candidate_adaptive_lr.pkl"
MIN_PER_CLASS=30

def main():
    if not SOURCE.exists():
        raise SystemExit("No feedback file found. Submit reviewed feedback through the app/API first.")
    rows=[]
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        try: r=json.loads(line)
        except json.JSONDecodeError: continue
        label=r.get("verified_label")
        if label in {"Human-written","AI-generated"} and r.get("text","").strip():
            rows.append({"text":r["text"],"label":1 if label=="AI-generated" else 0})
    df=pd.DataFrame(rows).drop_duplicates(subset=["text","label"])
    counts=df.label.value_counts().to_dict()
    if counts.get(0,0)<MIN_PER_CLASS or counts.get(1,0)<MIN_PER_CLASS:
        raise SystemExit(f"Need at least {MIN_PER_CLASS} verified examples per class. Current counts: {counts}")
    X_train,X_test,y_train,y_test=train_test_split(df.text,df.label,test_size=.2,random_state=42,stratify=df.label)
    model=Pipeline([("tfidf",TfidfVectorizer(ngram_range=(1,2),max_features=100000,sublinear_tf=True)),("classifier",LogisticRegression(max_iter=2000,class_weight="balanced",random_state=42))])
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    report=classification_report(y_test,pred,output_dict=True,zero_division=0)
    OUT.parent.mkdir(exist_ok=True)
    joblib.dump(model,OUT)
    metrics={"candidate":str(OUT),"examples":len(df),"class_counts":counts,"f1_macro":f1_score(y_test,pred,average="macro"),"classification_report":report,"approved_for_production":False}
    (ROOT/"models"/"candidate_adaptive_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(json.dumps(metrics,indent=2))
    print("\nCandidate saved separately. Review metrics and test against the official held-out test set before any deployment.")
if __name__=="__main__": main()
