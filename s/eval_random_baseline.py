"""Evaluate answers normally and against same-category random answers.

Expected JSONL schema (one object per line):
    {"text": "model answer", "answers": ["reference 1", ...],
     "category": "OCR", "question": "optional question text"}

Example:
    python evaluate_with_random_baseline.py testfile.jsonl \
        --output evaluation_with_baseline.json --trials 100
"""

import argparse
import json
import random
import statistics
from collections import defaultdict

import torch
from bert_score import BERTScorer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "roberta-large-mnli"


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = {"text", "answers", "category"} - record.keys()
            if missing:
                raise ValueError(f"Line {line_number} is missing: {sorted(missing)}")

            text = record["text"]
            record["text"] = " ".join(text) if isinstance(text, list) else str(text)
            answers = record["answers"]
            record["answers"] = answers if isinstance(answers, list) else [answers]
            if not record["answers"]:
                raise ValueError(f"Line {line_number} has no reference answers")
            records.append(record)
    return records


def compute_nli_scores(tokenizer, model, device, predictions, references, strategy, batch_size):
    """Score whether each reference (premise) supports its prediction (hypothesis)."""
    scores = []
    for start in range(0, len(predictions), batch_size):
        preds = predictions[start : start + batch_size]
        refs = references[start : start + batch_size]
        inputs = tokenizer(refs, preds, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(model(**inputs).logits, dim=-1)

        contradiction, neutral, entailment = probabilities[:, 0], probabilities[:, 1], probabilities[:, 2]
        if strategy == "net_entailment":
            values = entailment - contradiction
        elif strategy == "net_normalized":
            values = (entailment - contradiction + 1.0) / 2.0
        elif strategy == "weighted":
            values = entailment + 0.5 * neutral
        elif strategy == "relative":
            values = entailment / (entailment + contradiction + 1e-8)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        scores.extend(values.cpu().tolist())
    return scores


def harmonic_mean(left, right):
    left, right = max(0.0, left), max(0.0, right)
    return 0.0 if left + right == 0 else 2 * left * right / (left + right)


def score_pairs(predictions, references, tokenizer, model, bertscore, device, strategy, batch_size):
    """Return one BERTScore, NLI score, and composite score per supplied pair."""
    nli = compute_nli_scores(tokenizer, model, device, predictions, references, strategy, batch_size)
    nli_normalized = nli if strategy == "net_normalized" else compute_nli_scores(
        tokenizer, model, device, predictions, references, "net_normalized", batch_size
    )
    _, _, bert_f1 = bertscore.score(predictions, references, batch_size=batch_size)
    bert_f1 = bert_f1.tolist()
    composite = [harmonic_mean(bert, nli_norm) for bert, nli_norm in zip(bert_f1, nli_normalized)]
    return {"bert_f1": bert_f1, "nli_score": nli, "composite_score": composite}


def summarize(scores, threshold):
    return {
        "mean_bert_f1": statistics.fmean(scores["bert_f1"]),
        "mean_nli_score": statistics.fmean(scores["nli_score"]),
        "mean_composite_score": statistics.fmean(scores["composite_score"]),
        "bert_fraction_above_threshold": sum(x > threshold for x in scores["bert_f1"]) / len(scores["bert_f1"]),
        "nli_fraction_above_threshold": sum(x > threshold for x in scores["nli_score"]) / len(scores["nli_score"]),
        "composite_fraction_above_threshold": sum(x > threshold for x in scores["composite_score"]) / len(scores["composite_score"]),
    }


def normal_evaluation(records, tokenizer, model, bertscore, device, strategy, batch_size, threshold):
    """Compare every answer with all its own references and retain the best complete pairing."""
    preds, refs, owners = [], [], []
    for index, record in enumerate(records):
        for reference in record["answers"]:
            preds.append(record["text"])
            refs.append(str(reference))
            owners.append(index)

    flat = score_pairs(preds, refs, tokenizer, model, bertscore, device, strategy, batch_size)
    best = [None] * len(records)
    for pair_index, owner in enumerate(owners):
        candidate = {key: values[pair_index] for key, values in flat.items()}
        if best[owner] is None or candidate["composite_score"] > best[owner]["composite_score"]:
            best[owner] = candidate

    scores = {key: [item[key] for item in best] for key in flat}
    return summarize(scores, threshold)


def random_category_baseline(records, trials, seed, tokenizer, model, bertscore, device, strategy, batch_size, threshold):
    """Randomly compare each item with a *different* item's reference in its category."""
    by_category = defaultdict(list)
    for index, record in enumerate(records):
        by_category[record["category"]].append(index)

    eligible = [i for i, record in enumerate(records) if len(by_category[record["category"]]) > 1]
    skipped = len(records) - len(eligible)
    if not eligible:
        raise ValueError("Every category has only one record; no random within-category baseline is possible.")

    rng = random.Random(seed)
    trial_summaries = []
    for _ in range(trials):
        predictions, references = [], []
        for index in eligible:
            record = records[index]
            candidates = [other for other in by_category[record["category"]] if other != index]
            other_record = records[rng.choice(candidates)]
            predictions.append(record["text"])
            references.append(str(rng.choice(other_record["answers"])))
        scores = score_pairs(predictions, references, tokenizer, model, bertscore, device, strategy, batch_size)
        trial_summaries.append(summarize(scores, threshold))

    aggregate = {}
    for key in trial_summaries[0]:
        values = [trial[key] for trial in trial_summaries]
        aggregate[key] = {
            "mean_over_trials": statistics.fmean(values),
            "std_over_trials": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return {"eligible_examples": len(eligible), "skipped_singleton_category_examples": skipped, "trials": trials, "summary": aggregate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to the JSONL evaluation data")
    parser.add_argument("--output", default="evaluation_with_baseline.json")
    parser.add_argument("--trials", type=int, default=100, help="Random baseline repetitions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--strategy", default="net_normalized", choices=["net_entailment", "net_normalized", "weighted", "relative"])
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least 1")

    records = load_records(args.input)
    if not records:
        raise ValueError("The input file contained no records")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device).eval()
    bertscore = BERTScorer(model_type="roberta-large", device=device)

    result = {
        "configuration": {"input": args.input, "nli_strategy": args.strategy, "threshold": args.threshold, "random_seed": args.seed},
        "normal_evaluation": normal_evaluation(records, tokenizer, model, bertscore, device, args.strategy, args.batch_size, args.threshold),
        "same_category_random_baseline": random_category_baseline(records, args.trials, args.seed, tokenizer, model, bertscore, device, args.strategy, args.batch_size, args.threshold),
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
