import json
import os


def convert(src, dst):

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    count = 0

    with open(src, "r", encoding="utf-8") as fin, \
         open(dst, "w", encoding="utf-8") as fout:

        for line in fin:
            item = json.loads(line)

            new_item = {
                "prompt": item["x"],
                "chosen": item["y_plus"],
                "rejected": item["y_minus"]
            }

            fout.write(json.dumps(
                new_item,
                ensure_ascii=False
            ) + "\n")

            count += 1

    print(f"Converted {count} samples")
    print(dst)


if __name__ == "__main__":

    convert(
        "/mnt/bai_data/yang-safe/data/pku_saferlhf_10k_triplet/train.jsonl",
        "data/pku10k_pref_train.jsonl"
    )

    convert(
        "/mnt/bai_data/yang-safe/data/pku_saferlhf_10k_triplet/eval.jsonl",
        "data/pku10k_pref_eval.jsonl"
    )
