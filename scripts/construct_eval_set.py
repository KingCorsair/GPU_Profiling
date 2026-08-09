import os
import json
import random
from PIL import Image
from datasets import load_dataset


OUTPUT_DIR = "./vis_pruner_copy/vispruner_eval_dataset"
IMAGES_DIR = os.path.join(OUTPUT_DIR, 'images')
TARGET_PER_CATEGORY = 30

os.makedirs(IMAGES_DIR, exist_ok=True)

def save_images(img, filename):
    path = os.path.join(IMAGES_DIR, filename)
    if(isinstance(img, Image.Image)):
        img.convert("RGB").save(path)
    return os.path.join("images", filename)

categories = {
    "OCR":[],
    "counting":[],
    "spatial_reasoning":[],
    "object_presence":[],
    "coarse_description":[],
    "fine_attributes":[]
}

# TEXTVQA --> OCR
textvqa = load_dataset("facebook/textvqa", split="validation", revision="refs/convert/parquet")
for sample in textvqa:
    if len(categories["OCR"]) >= TARGET_PER_CATEGORY:
        break
    
    img_name = f"textvqa_{sample['image_id']}.jpg"
    rel_path = save_images(sample["image"], img_name)

    categories["OCR"].append({
        "question_id": f"textvqa_{sample['question_id']}",
        "image": rel_path,
        "category": "ocr",
        "source_dataset": "TEXT_VQA",
        "question": sample["question"],
        "answers": sample["answers"],
        "question_type": "short_answer"
    })

# MME --> spatial reasoning, object presence, fine attributes
mme_subtask_map = {
    "existence": "object_presence",
    "count": "counting",
    "position": "spatial_reasoning",
    "color": "fine_attributes",
    "text": "ocr",
    "scene": "coarse_description"
}

mme = load_dataset("lmms-lab/MME", split="test")
for sample in mme:
    subtask = sample.get("category", "").lower()
    target_cat = mme_subtask_map.get(subtask)

    if target_cat and len(categories[target_cat]) < TARGET_PER_CATEGORY:

        raw_qid = str(sample['question_id'])
        clean_qid = os.path.splitext(raw_qid)[0].replace('/', '_')

        img_name = f"mme_{clean_qid}.jpg"
        rel_path = save_images(sample["image"], img_name)

        categories[target_cat].append({
            "question_id": f"mme_{sample['question_id']}",
            "image": rel_path,
            "category": target_cat,
            "source_dataset": "MME",
            "question": sample["question"],
            "answers": [sample["answer"]],
            "question_type": "short answer"
        })

# ScienceQA --> Attributes

# VisWiz --> Coarse Description
