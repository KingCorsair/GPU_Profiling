Here's where things stand and what's left:

Done:

Eval set built: 180 images across 6 categories (30 each), sourced from TextVQA/MME/VizWiz/GQA, deduped globally by image hash.
Dev/test split written (seeded shuffle, 50/50 per category) — worth a last sanity check that dev.json/test.json actually landed with balanced per-category counts, then treat test.json as locked: commit it and don't regenerate.
ScienceQA correctly scoped out — it's serving the speed/prefill axis and already has its own eval script.
Next:

The ScienceQA hand-check (still open, and worth doing before step 3, not after) — take 20 outputs from the existing ScienceQA script, compare exact match against your own judgment by hand. Whatever exact match gets wrong there is a preview of what'll go wrong on the new set, which now has three different answer shapes (free-text from TextVQA/VizWiz, binary yes/no from MME/GQA), not just ScienceQA's clean multiple choice.

Inference runner for the heterogeneous set — loop over dev.json only (test stays untouched), run the model, save raw generations tagged with category/source_dataset/question_type. This is boilerplate — happy to scaffold it with you.

Scoring logic for the heterogeneous set — yours to design, and it's the real work: given what the hand-check surfaces, decide how binary questions get scored differently from free-text ones, and whether free-text needs more than exact match (synonym handling, substring match, etc.).

Random baseline — once scoring works, run a random-answer baseline through it per rule 17. If random scores near real answers on any category, that category (or the scorer) is broken.

Validate the scorer itself — once you've got a working scoring approach, run it alongside a second scoring method on a sample and check agreement (rule 13), same idea as the ScienceQA hand-check but now against your own new scorer.

Inference runner — model_vqa_heterogeneous.py is scaffolded: loops over dev.json only, runs the model with VisPruner's visual_token_num/important_ratio knobs, writes raw generations to jsonl tagged with category/source_dataset/question_type/answers/visual_token_num. Not yet smoke-tested (no GPU in this sandbox) — worth running on the pod before trusting the output shape. You've since edited the file yourself (comment on line 19 now flags the scorer as still needed).

Open flag from the dev.json sanity check: category counts are balanced (15/15/15/15/15/15), but source_dataset is skewed — MME:69, TEXT_VQA:15, GQA:6, VizWiz:0. Looks like MME's scene→coarse_description mapping fills that category before the VizWiz loop ever runs. Eval-set composition is your call, not something I've touched.

Scorer design (yours, per project rules) — landed on:

Not multiple-choice prompting for this set (unlike ScienceQA), so per-dataset official scorers (convert_gqa_for_eval.py etc.) don't apply — you need one scorer keyed by question_type (binary vs free-text), not one dispatching by source_dataset.
Ruled out sentiment analysis (wrong tool — measures emotional polarity, not factual agreement).
Landed on BERTScore as the primary candidate for free-text, with a known blind spot: it's similarity-based, so it doesn't reliably catch negation ("yes there's a dog" vs "no there's no dog" can score deceptively high) — a real risk for your binary category (MME/GQA).
Discussed entailment/NLI (a separately fine-tuned model, e.g. roberta-large-mnli, not the same checkpoint as BERTScore) as the structural fix for negation, since it's explicitly trained to output contradiction/entailment/neutral.
Current lean: keep BERTScore and entailment as two independent scorers rather than fusing into a weighted average right away — a blend collapses the diagnostic signal you need for the rule-13 hand-check (where they disagree tells you something; averaged, it doesn't), and the weight itself would be an unvalidated free parameter.
Not yet started: the actual scorer implementation (yours to write — I'll review, not author), the random baseline (rule 17), and the second scoring-method agreement check (rule 13) once a working scorer exists.

_______
AUG 14
_______

Here's the sequence I'd follow, in order — each one unblocks the next:

1. Decouple the backbones (revisit, before anything else)
Swap BERTScorer(model_type="roberta-large-mnli") → BERTScorer(model_type="roberta-large") (or whatever bert-score's standard default is), keep roberta-large-mnli for the entailment classifier only. This is a one-line change but it changes what every downstream number means, so do it before you generate any results you'll look at seriously.

2. Build the hand-labeled validation sample (rule 13)
Pull a sample of pred/target pairs — deliberately over-weighted toward your binary category (MME/GQA), since that's where negation is most likely to bite. Label each yourself as correct/incorrect. This is the part that's actually yours to design (not a scorer to hand off) — how many pairs, how you sample across categories, is a judgment call worth thinking through rather than defaulting to "grab 50 at random."

3. Look at score distributions before picking any threshold
For BERTScore F1 and each entailment strategy separately, plot or just eyeball where your hand-labeled correct vs incorrect pairs land. Only pick a threshold once you can see it separates the two groups — not a round number like 0.75 chosen in advance. Different strategies (net_entailment vs weighted vs relative) will likely need different thresholds since their distributions aren't shaped the same.

4. Check where BERTScore and entailment disagree
This is the actual payoff of keeping them separate. Pull the pairs where one says correct and the other says incorrect, read them. That tells you whether entailment is catching the negation cases BERTScore misses (the hypothesis), or whether the disagreement is noise. This will also retroactively validate whether step 1's fix mattered — if disagreement was suspiciously rare before, it should widen after decoupling the backbones.

5. Random baseline (rule 17)
Not started yet per your earlier note — score random pred/target pairings (shuffle targets against unrelated preds) through the same pipeline. If random scores land close to real scores for either metric, that metric isn't discriminating and no threshold will save it.

6. Only then: decide on the composite
Once BERTScore and entailment are each individually validated (thresholds set, disagreement understood), revisit whether harmonic-meaning them adds anything over reporting them separately, or whether the disagreement itself is more useful to keep visible in your final eval output than a fused number.

7. Per-category breakdown (rule 14)
Once the scorer is trustworthy, don't just report one accuracy number — break it out by category (OCR, counting, spatial, binary, free-text) so a collapse in one category isn't hidden by the average.

Not yet relevant but keep in mind: none of this should touch your test set (rule 11) — steps 2–6 should all run against dev. Test set stays locked until the scorer itself is validated and finalized.

I'd start with #1 and #2 today — #1 is quick, and #2 is the long pole since it's genuinely your judgment call on what "correct" means for free-form answers.

---
AUG 20
----

Confirm the swap's scope before rerunning anything. Check whether answers/text are shaped the same way for MME and GQA as they are for TextVQA — a list of crowd answers vs. a single string. If GQA/MME already had single-string fields in the right positions, applying this same fix there could break what was already correct. Grep a few lines per source_dataset in testfile.jsonl to check before you touch the loader further.

Decide how to handle the list-of-answers case, since that's now your actual ground truth, not garbage to discard. Standard VQA eval doesn't join 10 crowd answers into one string — it scores the pred against each reference and aggregates (e.g. take the max BERTScore/NLI across references, or the classic VQA-accuracy soft vote: agreement fraction capped at 1). That's a design call, not a mechanical fix — how you aggregate across references will shift every score, so decide it deliberately rather than let " ".join() make the choice by default.

Throw out or re-derive your hand-labeled sample from step 2 of the original plan. Whatever you labeled correct/incorrect was judged against the old (swapped) pairs — so "correct" may have meant "does the model's babble resemble a garbled ground-truth blob," which isn't the judgment you meant to make. Don't reuse those labels; relabel against the corrected pred/target.

Regenerate all four evaluation_results_*.json files (normalized/weighted/relative/entailment) once 1-2 are settled — they're all scored on the swapped data and aren't safe to look at until then.

Then redo score-distribution eyeballing (step 3) and the BERTScore/NLI disagreement pass (step 4) on the corrected data — the earlier version of both was built on the wrong pairing, so conclusions from it (including the "NLI catches this better" observation) need to be re-checked, not assumed to carry over.

Random baseline (step 5) is still outstanding — do it after the above, on the corrected fields, so the shuffle is meaningful (shuffling real ground-truth answers against unrelated preds, not the swapped version).

Composite decision and per-category breakdown come last, same as the original plan.

---
AUG 22
---

The random baseline (rule 17) is a control on the metric itself, not on the model. The question it answers: is BERTScore/NLI actually detecting semantic agreement, or would they score almost anything that's the same shape of text just as highly?

Mechanics: take your real predictions, but pair each one with the wrong ground truth — a reference list pulled from a different question — and run those mismatched pairs through the exact same scoring pipeline you just built (max-over-references BERTScore, max-over-references NLI, composite). You now have two score distributions: real pairs and random pairs.

If random pairs score close to real pairs → the metric isn't discriminating correctness at all, it's rewarding something superficial (fluent English, similar length, same domain vocabulary). No threshold from step 3 will save it — the metric itself is broken.
If real pairs score clearly higher than random pairs → that's evidence the metric is tracking actual agreement, and the threshold you picked in step 3 is measuring something real.
Two design decisions that matter here, given your data:

What counts as "wrong." Shuffle whole reference-lists between examples, not individual crowd answers within one example — a TextVQA question's 10 answers are still a coherent target, just for the wrong question. Scrambling those internally would test something else (reference-list coherence, not pred/target agreement).

Shuffle within category, not across the whole set. If you randomly pair an MME binary yes/no answer with a TextVQA free-text answer, of course the score tanks — but that's not telling you the metric discriminates correctness, it's telling you it discriminates task type, which you already knew. Shuffle binary-vs-binary and free-text-vs-free-text separately so the comparison is apples-to-apples within each category. Otherwise the baseline looks reassuringly low for the wrong reason.

One more thing worth deciding: with only 90 examples, a single shuffle is noisy — consider running it 3-5 times with different seeds and looking at the range, not one number.

That's the concept — the actual shuffle-and-score harness is yours to write per the project rules, since it's the mechanism that validates your scorer. Happy to review it once you've got a draft, or scaffold just the file-I/O boilerplate around it if you want.

---
PART 2
---
Reviewed it. One real bug against your own design note, plus a design question worth resolving before you trust the output.

Bug: random_category_baseline samples one answer, not the whole reference list.

At eval_random_baseline.py:140:


references.append(str(rng.choice(other_record["answers"])))
Your Aug 22 note is explicit: "Shuffle whole reference-lists between examples, not individual crowd answers within one example." This does the opposite — it grabs the full answers list from a mismatched record, then throws away all but one randomly chosen answer.

That breaks the comparison you're building the whole script around. normal_evaluation scores each prediction against every one of its own references and keeps the max (mirroring the Aug 20 aggregation decision — max-over-references). The random baseline scores against exactly one reference. A single answer gives the scorer far fewer chances to land a match than a full list of ~10 does, so the random distribution will look artificially low — not because the metric is discriminating correctness, but because you've handicapped the number of shots it gets. That's exactly the kind of confound rule 17 is supposed to rule out, and it'll make BERTScore/NLI look more discriminating than they actually are.

Fix direction: when you pick other_record, use its entire answers list the same way normal_evaluation uses a record's own list — expand into (pred, ref) pairs for every answer in that list, score them all, and keep the max per record, per trial. Same aggregation logic, just against the wrong record's list instead of the right one.

Design question, not a bug: which reference gets picked, and by what?

In normal_evaluation (eval_random_baseline.py:110-114), you pick the "best" reference by maximizing composite_score, then report bert_f1 and nli_score from that same reference. But your Aug 14 plan explicitly decided to keep BERTScore and entailment independent so disagreement stays visible ("a blend collapses the diagnostic signal you need"). If BERTScore's best-matching reference and NLI's best-matching reference aren't the same one — plausible, since they measure different things — then reporting both metrics off the composite-winning reference silently mixes them back together, which is the thing you decided against fusing. Worth deciding deliberately: max-per-metric independently (each metric picks its own best reference), or max-by-composite with both readouts riding along. Right now it's the latter, by default rather than by decision.

Small note: --threshold defaults to 0.75 — the exact round number your Aug 14 plan called out as the wrong way to pick a threshold. Fine as a CLI default since you're presumably supplying your own after eyeballing distributions, just flagging in case it gets used unthinkingly.

Once the reference-list fix is in, I'm happy to look at the diff.