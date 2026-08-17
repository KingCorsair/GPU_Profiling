import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import time 

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image

# Runs the heterogeneous eval set (dev.json) through the model and dumps raw
# generations. Need to build custom scorer
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["question"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, conv_mode, num_workers=4):
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, conv_mode)
    return DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)


def eval_model(args):
    # Give llava_llama.py's per-question timing its own file, separate from
    # model_vqa_science.py's, so direct runs of the two scripts never mix
    # their per-question lines into the same fallback file.
    os.environ.setdefault(
        "LLAVA_TIMING_FILE",
        "/workspace/GPU_Profiling/results/timing/model_vqa_heterogeneous_llava_timing.json",
    )

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    #Loading the pretrained model and then timing it.
    torch.cuda.synchronize()
    loading_model_start_time = time.perf_counter()
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        visual_token_num=args.visual_token_num,
        important_ratio=args.important_ratio,
    )
    torch.cuda.synchronize()
    loading_model_end_time = time.perf_counter()
    loading_model_elapsed_time = loading_model_end_time - loading_model_start_time

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    torch.cuda.synchronize()
    loading_data_start_time = time.perf_counter()
    data_loader = create_data_loader(
        questions, args.image_folder, tokenizer, image_processor, model.config, args.conv_mode
    )

    generation_list = []
    data_bar = tqdm(zip(data_loader, questions), total=len(questions))

    #generate the response for each question
    counter=1
    for (input_ids, image_tensors, image_sizes), line in data_bar:
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        image_tensors = image_tensors.to(dtype=torch.float16, device='cuda', non_blocking=True)

        answer_generation_start_time = time.perf_counter()
        with torch.inference_mode():
            output_ids, visual_token_num = model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
            data_bar.set_postfix({"visual_token_num": visual_token_num})
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        answer_generation_end_time = time.perf_counter()
        answer_generation_elapsed_time = answer_generation_end_time - answer_generation_start_time

        ans_file.write(json.dumps({
            "question_id": line["question_id"],
            "category": line["category"],
            "source_dataset": line["source_dataset"],
            "question_type": line["question_type"],
            "question": line["question"],
            "answers": line["answers"],
            "image": line["image"],
            "text": outputs,
            "answer_id": shortuuid.uuid(),
            "model_id": model_name,
            "visual_token_num": visual_token_num,
        }) + "\n")
        ans_file.flush()

        #Creating a dictionary key to 
        dictionary_key = "time to generate response for " + str(counter) + "th question" 

        generation_dictionary = {
            dictionary_key : answer_generation_elapsed_time,
        }

        generation_list.append(generation_dictionary)

        counter += 1

        
    ans_file.close()

    torch.cuda.synchronize()
    loading_data_end_time = time.perf_counter()

    loading_data_elapsed_time =  loading_data_end_time - loading_data_start_time

    output_dictionary = {
        "time to load the data" : loading_data_elapsed_time,
        "time to load the model" : loading_model_elapsed_time, 
        "time to generate each responses" : generation_list,
    }

    with open("/workspace/GPU_Profiling/results/timing/model_vqa_heterogeneous_timing.json","a+") as f:
        json.dump(output_dictionary,f)
        f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/workspace/GPU_Profiling/vis_pruner_copy/checkpoints/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/workspace/GPU_Profiling/vis_pruner_copy/vispruner_eval_dataset")
    parser.add_argument("--question-file", type=str, default="/workspace/GPU_Profiling/vis_pruner_copy/vispruner_eval_dataset/dev.json")
    parser.add_argument("--answers-file", type=str, default="/workspace/GPU_Profiling/vis_pruner_copy/eval/heterogeneous/answers/llava-v1.5-7b.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--visual_token_num", type=int, default=576)
    parser.add_argument("--important_ratio", type=float, default=0.5)
    args = parser.parse_args()

    eval_model(args)
