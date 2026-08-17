import os
import json
import random
from PIL import Image
from datasets import load_dataset
import hashlib


# from datasets import load_dataset
# ds = load_dataset("lmms-lab/GQA", "testdev_balanced_instructions", split="testdev")
# print("GQA Sample Keys:", ds[0].keys())

# exit()

OUTPUT_DIR = "./vis_pruner_copy/vispruner_eval_dataset"
IMAGES_DIR = os.path.join(OUTPUT_DIR, 'images')
TARGET_PER_CATEGORY = 50

os.makedirs(IMAGES_DIR, exist_ok=True)

seen_images = set() # track seen images via hash


# function to hash an image, once the image is hashed
def hash_image(img):
    if(isinstance(img, Image.Image)):
        img_rgb = img.convert("RGB")
        m = hashlib.md5(img_rgb.tobytes()).hexdigest()
        return m
    return None

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

    ## THIS SECTION IS COMMON IN ALL TESTS
    ## Hash image to see if the image is already added
    ## Done to ensure unique images across all splits
    img_hash = hash_image(sample["image"])
    if not img_hash or img_hash in seen_images:
        continue
    
    seen_images.add(img_hash)

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

        img_hash = hash_image(sample["image"])
        if not img_hash or img_hash in seen_images:
            continue

        seen_images.add(img_hash)

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
viswiz = load_dataset("lmms-lab-encoder/VizWiz-VQA", split="val")
for idx, sample in enumerate(viswiz):
    if(len(categories["coarse_description"]) >= TARGET_PER_CATEGORY):
        break

    if sample.get("answerable") == 0 or not sample.get("answers"):
        continue

    img_hash = hash_image(sample["image"])
    if not img_hash or img_hash in seen_images:
        continue

    seen_images.add(img_hash)

    answers = [a["answer"] for a in sample["answers"]]
    img_name = f"viswis_{idx}.jpg"
    rel_path = save_images(sample["image"], img_name)

    categories["coarse_description"].append({
        "question_id": f"viswiz_{idx}",
        "image": rel_path,
        "category": "coarse_description",
        "source_dataset": "VizWiz",
        "question": sample["question"],
        "answers": answers,
        "question_type": "short_answer"
    })

gqa = load_dataset("lmms-lab/GQA", "testdev_balanced_instructions", split="testdev")
gqa_images = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev")

print("Indexing GQA image map...")
image_map = {}
for img_sample in gqa_images:
    # Check common key variations for image ID in the image dataset
    img_id = str(img_sample.get("id") or img_sample.get("imageId") or "")
    img_obj = img_sample.get("image") or img_sample.get("img")
    if img_id and img_obj:
        image_map[img_id] = img_obj

print(f"Mapped {len(image_map)} images. Filtering questions...")

for idx, sample in enumerate(gqa):
    image_id = str(sample.get("imageId", ""))
    img = image_map.get(image_id)

    if img is None:
        continue

    question = sample.get("question") or ""
    answer = sample.get("answer") or ""
    
    if not question or not answer:
        continue

    q_lower = question.lower()
    target_cat = None

    if any(kw in q_lower for kw in ["left", "right", "above", "below", "behind", "next to"]):
        target_cat = "spatial_reasoning"
    elif any(kw in q_lower for kw in ["color", "material", "pattern", "wearing"]):
        target_cat = "fine_attributes"
    elif q_lower.startswith("is there") or q_lower.startswith("are there"):
        target_cat = "object_presence"

    if target_cat and len(categories[target_cat]) < TARGET_PER_CATEGORY:
        img_hash = hash_image(img)
        if not img_hash or img_hash in seen_images:
            continue

        seen_images.add(img_hash)

        clean_qid = f"gqa_{image_id}_{idx}"
        img_name = f"{clean_qid}.jpg"
        rel_path = save_images(img, img_name)

        categories[target_cat].append({
            "question_id": clean_qid,
            "image": rel_path,
            "category": target_cat,
            "source_dataset": "GQA",
            "question": question,
            "answers": [str(answer)],
            "question_type": "binary" if str(answer).lower() in ["yes", "no"] else "short_answer"
        })

# SPLIT

dev_split = []
test_split = []

random.seed(42)

for cat_name, items in categories.items():
    print(cat_name, len(items))

for cat_name, items in categories.items():
    random.shuffle(items)
    half = len(items)//2

    dev_split.extend(items[:half])
    test_split.extend(items[half:])

with open(os.path.join(OUTPUT_DIR, "dev.json"), "w") as f:
    json.dump(dev_split, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "test.json"), "w") as f:
    json.dump(test_split, f, indent=2)

print(f"dataset at {OUTPUT_DIR}")
