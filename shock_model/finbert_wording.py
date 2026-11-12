"""Optional text sentiment; never used as independent forecasting evidence."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--text',required=True); p.add_argument('--out',default='artifacts/finbert_wording.json'); a=p.parse_args()
    from transformers import pipeline
    classify=pipeline('text-classification',model='ProsusAI/finbert',tokenizer='ProsusAI/finbert',device=-1)
    text=Path(a.text).read_text(encoding='utf-8')
    result=classify(text,top_k=None,truncation=True,max_length=512)
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(text=text,scores=result,scope='Wording only; not public sentiment, an independent forecast, or confirmation. Text is truncated to 512 tokens.'),indent=2))
    print(out)
if __name__=='__main__': main()
