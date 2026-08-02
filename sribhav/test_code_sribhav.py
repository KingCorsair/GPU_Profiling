from PIL import Image
import numpy as np
import json

img = Image.open("photo.jpg").convert("RGB").resize((224, 224))
arr = np.array(img)

PATCH_SIZE = 16

# we maintain patch size at 14
GRID_SIZE = 224 // PATCH_SIZE
NUM_PATCHES = GRID_SIZE * GRID_SIZE # 14 * 14 = 196

patches = []
scores = []
token_indices = []


for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):
        y0 = row * PATCH_SIZE
        x0 = row * PATCH_SIZE

        patch = arr[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE]

        score = patch.var()

        patches.append(patch)
        scores.append(score)
        token_indices.append(row * GRID_SIZE + col)

scores = np.array(scores)

k = int(NUM_PATCHES*0.25)

top_idx = np.argsort(scores)[:-k]
top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

kept_token_indices = [int(token_indices[i]) for i in top_idx]


with open("kept_patches.json", "w") as f:
    json.dump({
        "image_size": [224, 224],
        "patch_size": PATCH_SIZE,
        "grid_size": GRID_SIZE,
        "num_patches": NUM_PATCHES,
        "kept_indices": kept_token_indices,
    },
    f,
    indent=2
    )

masked = arr.copy()

keep_set = set(kept_token_indices)

for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):
        token = row * GRID_SIZE + col

        if token not in keep_set:
            y0 = row * PATCH_SIZE
            x0 = col * PATCH_SIZE
            masked[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE] = 0

Image.fromarray(masked).save("kept_patches_visualization.png")

