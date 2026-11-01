"""Verify exported artifacts and optionally restore original experiment paths."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];bundle=root/'experiments';meta=json.loads((bundle/'manifest.json').read_text());rows={r['original_path']:r for r in meta['records']}
    if a.output:a.output.mkdir(parents=True,exist_ok=False)
    def safe(base,name):
        target=(base/name).resolve()
        if not target.is_relative_to(base.resolve()):raise ValueError('Unsafe manifest path: '+name)
        return target
    for name,row in rows.items():
        canonical=row
        if 'duplicate_of' in row:canonical=rows[row['duplicate_of']]
        pieces=canonical.get('parts') or [dict(path=canonical['stored_path'])]
        digest=hashlib.sha256();count=0
        target=safe(a.output,name) if a.output else None
        if target:target.parent.mkdir(parents=True,exist_ok=True)
        stream=target.open('wb') if target else None
        try:
            for piece in pieces:
                src=safe(bundle,piece['path']);part_hash=hashlib.sha256()
                with src.open('rb') as f:
                    while block:=f.read(1024*1024):
                        digest.update(block);part_hash.update(block);count+=len(block)
                        if stream:stream.write(block)
                if piece.get('sha256') and part_hash.hexdigest()!=piece['sha256']:raise ValueError('Part hash mismatch: '+piece['path'])
            if digest.hexdigest()!=row['sha256'] or count!=row['bytes']:raise ValueError('Artifact mismatch: '+name)
        finally:
            if stream:stream.close()
    print(f"Verified {len(rows)} artifacts"+(' and restored original paths' if a.output else ' (no files restored)'))
if __name__=='__main__':main()
