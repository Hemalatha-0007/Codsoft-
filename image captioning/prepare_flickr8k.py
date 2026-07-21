"""
prepare_flickr8k.py
--------------------
Utility to convert the raw Flickr8k annotation file (Flickr8k.token.txt,
where each line looks like "1000268201_693b08cb0e.jpg#0<TAB>A caption.")
into the flat "image,caption" CSV format expected by dataset.py.

Usage:
    python prepare_flickr8k.py --tokens_file Flickr8k.token.txt --out data/flickr8k/captions.txt

The public Kaggle "Flickr8k" release already ships a captions.txt in this
flat format, in which case you can skip this script entirely and just point
config.CAPTIONS_FILE at it directly.
"""

import argparse
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens_file", required=True, help="Path to Flickr8k.token.txt")
    parser.add_argument("--out", required=True, help="Path to write captions.txt (CSV)")
    args = parser.parse_args()

    rows = []
    with open(args.tokens_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            image_and_idx, caption = line.split("\t")
            image_name = image_and_idx.split("#")[0]
            rows.append((image_name, caption))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "caption"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} (image, caption) rows to {args.out}")


if __name__ == "__main__":
    main()
