"""Verify downloaded checkpoint bytes against a pinned Hugging Face commit."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    from huggingface_hub import HfApi
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--model-id', default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--revision', default='4481d270cc22fd5c4d1bb5df129622006ccd9234')
    args = parser.parse_args()
    if len(args.revision) != 40 or any(c not in '0123456789abcdef' for c in args.revision):
        parser.error('An immutable 40-character commit is required')
    info = HfApi().model_info(args.model_id, revision=args.revision, files_metadata=True)
    verified = []
    for entry in info.siblings:
        path = args.directory / entry.rfilename
        if not path.is_file() or path.stat().st_size != entry.size:
            raise RuntimeError(f'Missing or incorrect size: {entry.rfilename}')
        sha256 = hashlib.sha256()
        git_blob = hashlib.sha1(f'blob {entry.size}\0'.encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                sha256.update(block)
                git_blob.update(block)
        lfs = getattr(entry, 'lfs', None)
        expected = getattr(lfs, 'sha256', None) if lfs is not None else None
        if isinstance(lfs, dict):
            expected = lfs.get('sha256')
        if expected:
            if sha256.hexdigest() != expected:
                raise RuntimeError(f'LFS checksum mismatch: {entry.rfilename}')
        elif git_blob.hexdigest() != entry.blob_id:
            raise RuntimeError(f'Git blob checksum mismatch: {entry.rfilename}')
        verified.append({'file': entry.rfilename, 'bytes': entry.size, 'sha256': sha256.hexdigest()})
        print(f'Verified {entry.rfilename}', flush=True)
    provenance = {'model_id': args.model_id, 'revision': args.revision,
                  'verified_at_utc': datetime.now(timezone.utc).isoformat(),
                  'verification': 'Each file checked against pinned Hub LFS SHA256 or Git blob SHA1', 'files': verified}
    (args.directory / 'download-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')

if __name__ == '__main__':
    main()
