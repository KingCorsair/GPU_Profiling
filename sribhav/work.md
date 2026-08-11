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