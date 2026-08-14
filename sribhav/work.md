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