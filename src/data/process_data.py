import subprocess
import argparse
from pathlib import Path
import itertools

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--max-docs", type=int, default=0)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    subprocess.call(["tar", "-xzf", f"{data_path/'raw'/'aclImdb_v1.tar.gz'}", "-C",
                     f"{data_path/'processed'}"])

    if args.max_docs > 0:
        for phase_path in [d for d in (data_path / "processed"/ "aclImdb").iterdir() if d.is_dir()]:
            for sent in [d for d in phase_path.iterdir() if d.is_dir()]:
                for j, doc in enumerate(sent.iterdir()):
                    if j > args.max_docs:
                        doc.unlink()



