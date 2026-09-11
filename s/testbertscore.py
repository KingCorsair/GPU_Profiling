from bert_score import BERTScorer
import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForSequenceClassification
import json

bertscore = BERTScorer(model_type='roberta-large-mnli')

# preds = ["master kenobi", "hello there", "how are you doing"]
# target = ["general kenobi", "hi", "whats up brother"]

# output = bertscore.score(preds, target)
# print(output)

def compute_entailment_score(preds, targets, model_name="roberta-large-mnli"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    scores = []

    for pred, target in zip(preds, targets):
        inputs = tokenizer(target, pred, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            output = model(**inputs)
        
        logits = output.logits

        probs = torch.softmax(logits, dim=-1)

        scores.append(probs)
    
    return scores

# print(compute_entailment_score(preds, target))

def eval(answers_file):
    afile = []
    with open(answers_file, 'r') as f:
        preds = []
        ground_truths = []

        for aline in f:
            line = json.loads((aline))
            pred = line["answers"]
            gt = line["text"]

            if isinstance(pred, list):
                pred = "".join(pred)
            if isinstance(gt, list):
                gt = "".join(gt)
            
            preds.append(pred)
            ground_truths.append(gt)
            
        entailment_scores = compute_entailment_score(preds, ground_truths)
        precision, recall, f1 = bertscore.score(preds, ground_truths)

    return f1.tolist(), entailment_scores

if __name__ == "__main__":
    berts, entail = eval("testfile.jsonl")
    print("BERT Scores", berts)
    print("Entailment Scores", entail)
