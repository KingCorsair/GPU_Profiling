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

Want me to start on the inference runner (step 2) while you do the hand-check?