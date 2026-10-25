# Accuracy checks needed from Sribhav

From Amay, 2026-10-25. Two of my speed changes alter some model answers. I can show *that*
answers changed; whether they got *worse* is a scoring question, and that's yours. Neither
check blocks my work, but both must be done before these changes appear in the final writeup.

Everything referenced is on branch `amay/wp3-wp5-serving`.

---

## Request 1: FP16 vision tower (WP2), per-category accuracy

**What changed.** Commit `caf19fe` loads the CLIP vision tower in FP16 instead of FP32. It is
one line, `clip_encoder.py:30`. The parent commit, `e295fd0`, is the FP32 baseline. Nothing
else differs between the two.

**Why it matters.** The tower got 2.4× faster (34.0 → 14.4 ms). But at `visual_token_num=128`
some answers changed, and the changes cluster in the categories CLAUDE.md says break first.

**What I already checked** (`results/wp2_correctness/`):

| Check | vtn=576 | vtn=128 |
|---|---|---|
| Exact text, FP32 vs FP16, 90 dev images | 90/90 identical | 80/90 identical |
| TextVQA, 1,000 questions, official evaluator | 57.62% → 57.65% | 56.50% → 56.75% (95% CI −0.74 to +1.25 pp) |

The 10 changed answers at 128, by category:

| Category | Changed |
|---|---|
| OCR | 5 of 15 |
| counting | 2 of 15 (both flip yes ↔ no) |
| object presence | 2 of 15 (wording only) |
| spatial reasoning | 1 of 15 (wording only) |
| coarse description, fine attributes | 0 |

The full before/after text for each changed answer is at the bottom of
`results/wp2_correctness/wp2_layer1_1.txt`.

**What I need from you:**

1. Per-category accuracy for FP32 (`e295fd0`) vs FP16 (`caf19fe`) at vtn=128, plus 576 as a
   sanity check. Use your validated scorer.
2. Run it on your **dev split**, not the locked test set. This result decides whether I keep or
   revert the change, and making that decision from test results would contaminate the test
   set (rule 11). STATUS.md currently says "locked eval set"; that's my mistake, and I'll fix it.
3. Call out **OCR and counting** separately. If OCR drops beyond your noise level, I'll revert
   the one-line change.
4. Give the answer in the form "OCR X% → Y%, ±Z", with whatever paired test or interval you
   normally use. If the difference is inside noise, say that; it's a valid result.

---

## Request 2: batched serving (WP5), does batching cost accuracy?

**What changed.** My new server (`vis_pruner_copy/llava/serve/batch_server.py`) can run up to
8 requests through the model together, instead of one at a time. Throughput at a fixed
64-token output, measured offline (`results/wp5_validation/`):

| Batch size | vtn=576 | vtn=128 |
|---|---|---|
| 1 | 0.48 req/s | 0.52 req/s |
| 8 | 2.05 req/s (4.3×) | 3.01 req/s (5.8×) |
| 16 | 2.69 req/s (5.6×) | 4.52 req/s (8.7×) |

**Why answers can change.** In FP16, a matrix multiply over 8 rows doesn't round exactly like
one over 1 row. Greedy decoding occasionally flips a word, after which the rest of the answer
diverges. This isn't requests leaking into each other; I tested for that:

- Each request gives the identical answer whatever other request it is paired with, in either
  order (10/10 test requests, both configs).
- 8 copies of the same image in one batch all keep the same visual tokens.

**What I already checked**, batched answers vs one-at-a-time answers on the 90 dev images:

| | vtn=576 | vtn=128 |
|---|---|---|
| Queue only (batch size 1) | 90/90 identical | 90/90 identical |
| Batch of 2 | 90/90 | 90/90 |
| Batch of 4 | 90/90 | 90/90 |
| Batch of 8 | 89/90 | 90/90 |

So 1 of 540 batched answers differs. It is `textvqa_34610` (OCR) at vtn=576, batch of 8,
with the same visual tokens kept: the single-request answer stops after one sentence, and
the batched answer continues for two more. Rounding tipped the decision to emit
end-of-sequence, and the first sentence is identical.

An earlier run (`wp5_val_1`) found a bigger problem at 128: only 79–80/90 identical, because
batching changed which visual tokens VisPruner kept. That's fixed in commit `6a471cb`
(selection now runs per image). `wp5_val_2`, the table above, is the check on the fixed code.

**What I need from you:**

1. Score the batched answers and the single-request answers with your scorer, per category.
   Both sets are in `results/wp5_validation/wp5_val_2.json`:
   - `configs.<vtn>.reference_answers` is the single-request answer for each question id;
   - `configs.<vtn>.batch.<B>.diffs` lists every batched answer that differs (one, see above).
   If you also want the full batched answer sets for scoring, tell me and I'll export them.
2. Tell me whether the differing answers are better, worse or neutral, and whether they
   concentrate in any category. With one difference in 540 I expect "neutral", but I'd rather you say it than me.
3. Dev split only, same reason as above.

---

## Summary

| # | What | Commits / files | Decision it feeds |
|---|---|---|---|
| 1 | FP32 vs FP16 tower, per category, vtn=128 (+576) | `e295fd0` vs `caf19fe`; `results/wp2_correctness/` | Keep or revert FP16 |
| 2 | Batched vs single answers, per category | `results/wp5_validation/wp5_val_2.json` | Whether S2 throughput can be reported without an accuracy caveat |

Ping me if any of the files are unclear or you want the answers in a different format.
