# VisPruner / GPU Profiling — Handoff Notes

## Goal
1. Download the ~12GB evaluation dataset needed to run VisPruner benchmarks.
2. Get the local VisPruner code copy (`vis_pruner_copy/`) properly committed into our own repo (`github.com/KingCorsair/GPU_Profiling`) as plain tracked files — not as a broken nested-git reference.

## Environment
- Running on a RunPod pod (GPU: NVIDIA A40).
- Container disk (`/`, `/root`) is only **30GB**, ~8.5G free — too small for large downloads.
- RunPod network volume mounted at **`/workspace`** has **242T free** and is writable.
  → Any large dataset downloads should go to `/workspace`, not `/root`.

## What happened (timeline)

1. `vis_pruner_copy/` (inside `~/gpu_profiling`) was originally cloned as a **nested git repo**, with its own `.git` pointing at the upstream `Theia-4869/VisPruner` repo. We don't have push access to that repo (GitHub account `KingCorsair` got a 403: `Permission to Theia-4869/VisPruner.git denied`).
2. Because it had its own `.git`, git tracked `vis_pruner_copy` in the outer `gpu_profiling` repo as a **gitlink** (submodule-style reference, mode `160000`) rather than as real files.
3. Attempted fix, run from `~/gpu_profiling`:
   ```bash
   git rm --cached vis_pruner_copy   # untrack the gitlink
   rm -rf vis_pruner_copy/.git       # remove nested repo history
   git add vis_pruner_copy           # re-add as plain files
   git commit -m "Add VisPruner reference as plain files"
   git push
   ```
4. **This got interrupted (Ctrl+C) partway through**, before `git add` finished staging the plain files. The commit that went through (`ab26d3f`, pushed to `KingCorsair/GPU_Profiling`) only recorded **deleting the old gitlink** — it did NOT add the actual code back. Net effect at that point: `vis_pruner_copy` was untracked entirely (not a submodule, not plain files), though the files were still on disk.
5. Re-ran the sequence:
   - Confirmed `vis_pruner_copy/.git` was already gone (good, no need to `rm -rf` again).
   - Ran `git add vis_pruner_copy` — **this hung and timed out after 2 minutes.**

## What got interrupted / blocked

- Root cause of the hang: `vis_pruner_copy/checkpoints/llava-v1.5-7b/` contains real model weight files:
  - `pytorch_model-00001-of-00002.bin` — **9.3GB**
  - `pytorch_model-00002-of-00002.bin` — **3.3GB**
- `git add` was trying to hash these multi-GB files (slow), and **even if it succeeded, GitHub rejects pushes containing files over 100MB** — so this approach would fail regardless.
- Nothing is currently staged/committed from this step. The repo is in a safe state: `vis_pruner_copy` shows as untracked, no partial/broken commit exists.

## What's pending — next steps

1. Add a `.gitignore` entry for `checkpoints/` (or specifically `vis_pruner_copy/checkpoints/`) so model weights are never staged.
2. Then run, from `~/gpu_profiling`:
   ```bash
   git add vis_pruner_copy
   git status   # sanity check: should show many "new file:" entries, NOT a single 160000 gitlink
   git commit -m "Add VisPruner reference as plain files"
   git push
   ```
3. Separately: download the ~12GB VisPruner eval dataset (see `VisPruner/EVAL.md` for per-benchmark download links, starting with `eval.zip`) into **`/workspace`**, not `/root`, and symlink it into `vis_pruner_copy/playground/data/eval` (or wherever the code expects it) if the code needs a fixed relative path.

## Key facts for reference
- Repo: `https://github.com/KingCorsair/GPU_Profiling`
- Last pushed commit at time of writing: `ab26d3f` (only contains the gitlink deletion, not the file re-add)
- Model checkpoints (`llava-v1.5-7b`, ~13GB total) must **not** be committed to git — keep them out via `.gitignore`, store on `/workspace` or another artifact store if they need to be shared.
