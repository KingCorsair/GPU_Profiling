import argparse
import time
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Give llava_llama.py's per-question timing its own file when this
    # script is run directly. setdefault, not direct assignment: if
    # scripts/time_sqa_sweep.py launched this as a subprocess, it already
    # set LLAVA_TIMING_FILE to a per-run scratch path -- don't override that.
    os.environ.setdefault(
        "LLAVA_TIMING_FILE",
        "/workspace/GPU_Profiling/results/timing/model_vqa_science_llava_timing.json",
    )

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    torch.cuda.synchronize()
    t_start_load = time.perf_counter()
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        visual_token_num=args.visual_token_num,
        important_ratio=args.important_ratio,
    )
    # Checkpoint: model finished loading, question loop hasn't started yet.
    # Everything before this point is load cost; everything after is generation.
    t_end_load = time.perf_counter()
    torch.cuda.synchronize()
    elapsed_model_load_time =  t_end_load - t_start_load

    # Data
    t_start_data = time.perf_counter()
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    if args.limit is not None:
        questions = questions[:args.limit]
    t_end_data = time.perf_counter()
    elapsed_model_data_time = t_end_data - t_start_data

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    t_start = time.perf_counter()
    data_bar = tqdm(questions)
    for i, line in enumerate(data_bar):
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs

        if 'image' in line:
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = process_images([image], image_processor, model.config)[0]
            images = image_tensor.unsqueeze(0).half().cuda()
            image_sizes = [image.size]
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            images = None
            image_sizes = None

        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids, visual_token_num = model.generate(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=1024,
                use_cache=True,
            )
            data_bar.set_postfix({"visual_token_num": visual_token_num})

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()
    t_end = time.perf_counter()

    if args.timing_file:
        timing_file = os.path.expanduser(args.timing_file)
        os.makedirs(os.path.dirname(timing_file), exist_ok=True)
        with open(timing_file, "w+") as tf:
            json.dump({
                "question_count": len(questions),
                "model_load_s": elapsed_model_load_time,
                "time_to_make_questions": elapsed_model_data_time,
                "generation_s": t_end - t_start,
            }, tf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--visual_token_num", type=int, default=576)
    parser.add_argument("--important_ratio", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None,
                         help="Only run the first N questions (after chunking) -- for quick smoke tests.")
    parser.add_argument("--timing-file", type=str, default=None,
                         help="Optional path to write a JSON sidecar with "
                              "question_count/model_load_s/generation_s")
    args = parser.parse_args()

    eval_model(args)
