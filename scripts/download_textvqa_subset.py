"""Fetch the TextVQA data the WP2 Layer-2 gate needs, without the 7 GB archive.

Downloads TextVQA_0.5.1_val.json and then only the images used by the
subsample scripts/wp2_layer2_textvqa.py draws (same --n / --seed), by reading
the remote zip's directory and requesting each image's byte range. The full
train_val_images.zip does not fit in this pod's /workspace quota.

  python scripts/download_textvqa_subset.py            # the default 1,000-question subsample
  python scripts/download_textvqa_subset.py --all      # every image in the 5,000-question file
"""
import argparse
import io
import os
import struct
import sys
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wp2_layer2_textvqa import DATA_DIR, DEFAULT_N, DEFAULT_SEED, choose_questions  # noqa: E402

BASE_URL = "https://dl.fbaipublicfiles.com/textvqa"
ZIP_URL = f"{BASE_URL}/images/train_val_images.zip"
ANNOTATION_NAME = "TextVQA_0.5.1_val.json"
LOCAL_HEADER_SLACK = 4096  # local header is 30 bytes + file name + extra field


def fetch_range(start, end):
    request = Request(ZIP_URL, headers={"Range": f"bytes={start}-{end}"})
    with urlopen(request, timeout=120) as response:
        return response.read()


class RemoteFile(io.RawIOBase):
    """Just enough of a seekable file for zipfile to read the directory."""

    def __init__(self):
        with urlopen(Request(ZIP_URL, method="HEAD"), timeout=60) as response:
            self.size = int(response.headers["Content-Length"])
        self.position = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        base = {os.SEEK_SET: 0, os.SEEK_CUR: self.position, os.SEEK_END: self.size}[whence]
        self.position = base + offset
        return self.position

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.position
        n = min(n, self.size - self.position)
        if n <= 0:
            return b""
        data = fetch_range(self.position, self.position + n - 1)
        self.position += len(data)
        return data


def fetch_member(info, dest):
    if os.path.exists(dest):
        return False
    raw = fetch_range(info.header_offset,
                      info.header_offset + LOCAL_HEADER_SLACK + info.compress_size)
    if raw[:4] != b"PK\x03\x04":
        raise RuntimeError(f"{info.filename}: bad local header")
    name_len, extra_len = struct.unpack("<HH", raw[26:30])
    body = raw[30 + name_len + extra_len:][:info.compress_size]
    data = body if info.compress_type == zipfile.ZIP_STORED else zlib.decompress(body, -15)
    if zlib.crc32(data) != info.CRC:
        raise RuntimeError(f"{info.filename}: CRC mismatch")
    with open(dest + ".part", "wb") as f:
        f.write(data)
    os.replace(dest + ".part", dest)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--all", action="store_true", help="every image in the question file")
    args = parser.parse_args()

    image_dir = f"{DATA_DIR}/train_images"
    os.makedirs(image_dir, exist_ok=True)
    annotation_path = f"{DATA_DIR}/{ANNOTATION_NAME}"
    if not os.path.exists(annotation_path):
        with urlopen(f"{BASE_URL}/data/{ANNOTATION_NAME}", timeout=120) as response:
            data = response.read()
        with open(annotation_path, "wb") as f:
            f.write(data)

    questions, _, chosen = choose_questions(args.n, args.seed)
    wanted = sorted({q["image"] for q in questions} if args.all
                    else {questions[i]["image"] for i in chosen})

    archive = zipfile.ZipFile(RemoteFile())
    members = {os.path.basename(info.filename): info for info in archive.infolist()}
    missing = [name for name in wanted if name not in members]
    if missing:
        sys.exit(f"{len(missing)} images are not in the archive, e.g. {missing[:3]}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        fetched = list(pool.map(
            lambda name: fetch_member(members[name], os.path.join(image_dir, name)), wanted))
    print(f"{len(wanted)} images wanted, {sum(fetched)} downloaded, "
          f"{len(wanted) - sum(fetched)} already present -> {image_dir}")


if __name__ == "__main__":
    main()
