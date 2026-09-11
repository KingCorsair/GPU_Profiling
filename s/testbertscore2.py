import argparse
import json
import time
import torch
from bert_score import BERTScorer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertModel

MODEL_NAME = "roberta-large-mnli"

# Model loading (two RoBERTa-large checkpoints) is one-time, fixed overhead --
# time it separately from scoring so it doesn't get lumped into per-run
# scoring cost, same split model_vqa_heterogeneous.py makes for the LLaVA
# checkpoint (rule 3: report load time separately).
_load_start = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

bertscore = BERTScorer(model_type="roberta-large", device=device)

if device == "cuda":
    torch.cuda.synchronize()
MODEL_LOAD_S = time.perf_counter() - _load_start


def compute_nli_scores(preds, targets, strategy="net_entailment", batch_size=16):
    """Here we compute a single NLI score for a pred-target batch. We want this output to be a scalar
    
    Currently, I'll have 4 strategies.
      - 'net_entailment': We penalize contradictions more here
            P(Entailment) - P(Contradiction), range [-1, 1]
      - 'net_normalized': Same as net_entailment, but output between 0 and 1
            Normalized net entailment, range [0, 1]
      - 'weighted': Cleaner probability score
            P(E) + 0.5 * P(N), range [0, 1]
      - 'relative': Precision-esque metric -> Care less about neutral and focus on right vs wrong more as a balance
            P(E) / (P(E) + P(C)), range [0, 1]
    """
    scores = []
    
    for i in range(0, len(preds), batch_size):
        batch_preds = preds[i:i + batch_size]
        batch_targets = targets[i:i + batch_size]

        inputs = tokenizer(
            batch_targets, 
            batch_preds, 
            return_tensors="pt", 
            truncation=True, 
            padding=True
        ).to(device)

        with torch.no_grad():
            output = model(**inputs)
        
        probs = torch.softmax(output.logits, dim=-1)
        
        p_contradiction = probs[:, 0]
        p_neutral = probs[:, 1]
        p_entailment = probs[:, 2]

        if strategy == "net_entailment":
            batch_scores = p_entailment - p_contradiction
        elif strategy == "net_normalized":
            batch_scores = ((p_entailment - p_contradiction) + 1.0) / 2.0
        elif strategy == "weighted":
            batch_scores = p_entailment + (0.5 * p_neutral)
        elif strategy == "relative":
            batch_scores = p_entailment / (p_entailment + p_contradiction + 1e-8)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        scores.extend(batch_scores.cpu().tolist())
    
    return scores


def compute_composite_score(bert_f1_scores, nli_norm_scores):
    """Harmonic mean of BERTScore F1 and normalized NLI score"""
    composite_scores = []
    for b_f1, nli in zip(bert_f1_scores, nli_norm_scores):
        b_f1_clamped = max(0.0, b_f1)
        denom = b_f1_clamped + nli
        if denom == 0:
            composite_scores.append(0.0)
        else:
            harmonic_mean = 2 * (b_f1_clamped * nli) / denom
            composite_scores.append(harmonic_mean)
    return composite_scores

def calculate_accuracy(scores, threshold=0.75):
    if not scores:
        return 0.0
    return sum(1 for score in scores if score > threshold) / len(scores)

def _checkpoint(start):
    """perf_counter delta, synchronized so it captures actual GPU work
    finishing rather than just kernel-launch (queueing) time -- see rule 1."""
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start


def eval(answers_file, strategy="net_normalized"):
    preds, ground_truths = [], []

    t_read = time.perf_counter()
    with open(answers_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            gt = data["answers"]
            pred = data["text"]

            preds.append(" ".join(pred) if isinstance(pred, list) else pred)
            ground_truths.append(" ".join(gt) if isinstance(gt, list) else gt)
    read_input_s = _checkpoint(t_read)

    # 1. Compute single-score NLI
    t_nli = time.perf_counter()
    nli_scores = compute_nli_scores(preds, ground_truths, strategy=strategy)
    nli_primary_s = _checkpoint(t_nli)

    # 2. Compute BERTScore
    t_bert = time.perf_counter()
    _, _, f1_tensor = bertscore.score(preds, ground_truths)
    bert_f1 = f1_tensor.tolist()
    bertscore_s = _checkpoint(t_bert)

    # 3. Compute unified composite score
    # (Uses normalized NLI score in range [0, 1] for harmonic mean)
    t_nli_norm = time.perf_counter()
    nli_norm = compute_nli_scores(preds, ground_truths, strategy="net_normalized")
    nli_normalized_s = _checkpoint(t_nli_norm)
    composite_scores = compute_composite_score(bert_f1, nli_norm)

    bert_f1_acc = calculate_accuracy(bert_f1)
    nli_score_acc = calculate_accuracy(nli_scores)
    comp_score_acc = calculate_accuracy(composite_scores)

    timing = {
        "num_examples": len(preds),
        "model_load_s": MODEL_LOAD_S,
        "read_input_s": read_input_s,
        "nli_primary_strategy_s": nli_primary_s,
        "bertscore_s": bertscore_s,
        "nli_normalized_s": nli_normalized_s,
        "scoring_total_s": read_input_s + nli_primary_s + bertscore_s + nli_normalized_s,
    }

    return {
        "bert_accuracy": bert_f1_acc,
        "entailment_accuracy": nli_score_acc,
        "composite_score_accuracy": comp_score_acc,
        "bert_f1": bert_f1,
        "nli_score": nli_scores,
        "composite_score": composite_scores,
        "timing": timing,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--answers-file", default="testfile.jsonl",
        help="jsonl with 'answers' (ground truth) and 'text' (prediction) "
             "fields per line -- matches model_vqa_heterogeneous.py's output.",
    )
    parser.add_argument("--strategy", default="net_normalized",
                         choices=["net_entailment", "net_normalized", "weighted", "relative"])
    parser.add_argument("-o", "--output", default="evaluation_results_normalized.json")
    parser.add_argument("--timing-file", default=None,
                         help="Optional separate file for just the timing dict.")
    args = parser.parse_args()

    output_data = eval(args.answers_file, strategy=args.strategy)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    print(json.dumps(output_data["timing"], indent=2))
    if args.timing_file:
        with open(args.timing_file, "w", encoding="utf-8") as f:
            json.dump(output_data["timing"], f, indent=2)