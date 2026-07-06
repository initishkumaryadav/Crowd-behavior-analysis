"""Download the standard preprocessed ETH/UCY trajectory data.

We use the widely-adopted Social-GAN preprocessing (Gupta et al., CVPR 2018),
which is the de-facto benchmark split reused by NSP-SFM and virtually every
trajectory-prediction paper. Each scene ships train/val/test folders of
whitespace-separated rows:  <frame_id>  <ped_id>  <x>  <y>   (world metres).

Usage:
    python -m data.download            # downloads into ./datasets
"""
import io
import os
import sys
import zipfile

import requests

# Social-GAN preprocessed ETH/UCY data (Dropbox, referenced by sgan's
# scripts/download_data.sh). The zip extracts to a top-level  datasets/  dir.
SGAN_ZIP = "https://www.dropbox.com/s/8n02xqv3l9q18r1/datasets.zip?dl=1"
DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")


def download(dest: str = DEST) -> str:
    os.makedirs(dest, exist_ok=True)
    marker = os.path.join(dest, "eth", "test")
    if os.path.isdir(marker) and os.listdir(marker):
        print(f"[download] datasets already present in {dest}")
        return dest

    print(f"[download] fetching Social-GAN datasets zip ...")
    try:
        resp = requests.get(SGAN_ZIP, timeout=120)
        resp.raise_for_status()
    except Exception as e:  # network / mirror failure -> clear manual instructions
        print(
            "[download] FAILED to fetch automatically:\n"
            f"    {e}\n\n"
            "Manual fallback:\n"
            "  1. Download https://github.com/agrimgupta92/sgan (Code -> Download ZIP)\n"
            "  2. Copy the inner  sgan-master/datasets/  folder to  ./datasets/\n"
            "     so that ./datasets/eth/test/ exists.\n",
            file=sys.stderr,
        )
        raise SystemExit(1)

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    n = 0
    for member in zf.namelist():
        # keep only  <root>/datasets/<...>  entries
        parts = member.split("/")
        if "datasets" not in parts:
            continue
        idx = parts.index("datasets")
        rel = "/".join(parts[idx + 1:])
        if not rel or member.endswith("/"):
            continue
        out_path = os.path.join(dest, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with zf.open(member) as src, open(out_path, "wb") as dst:
            dst.write(src.read())
        n += 1
    print(f"[download] extracted {n} files into {dest}")
    return dest


if __name__ == "__main__":
    download()
