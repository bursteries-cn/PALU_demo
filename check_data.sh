python3 - <<'PY'
import json
from datasets import load_from_disk

path = "data/palu/TOFU/forget05_with_common_words_gpt"
ds = load_from_disk(path)

def parse_targets(row):
    return json.loads(row["target_words"])

def highlight(text, spans):
    # 从后向前插入颜色代码，避免破坏原始字符坐标
    for span in sorted(spans, key=lambda x: x["start"], reverse=True):
        start, end = span["start"], span["end"]
        text = (
            text[:start]
            + "\033[1;31m"
            + text[start:end]
            + "\033[0m"
            + text[end:]
        )
    return text

print("Dataset:", ds)
print("Features:", ds.features)
print("=" * 100)

for i in range(min(10, len(ds))):
    row = ds[i]
    targets = parse_targets(row)
    common = [json.loads(x) for x in row["common_words"]]

    print(f"\n[{i}] Question:")
    print(row["question"])

    print("\nAnswer（红色是敏感标注）:")
    print(highlight(row["answer"], targets))

    print("\nTarget spans:")
    for span in targets:
        extracted = row["answer"][span["start"]:span["end"]]
        print(
            f"  {span['start']:>3}:{span['end']:<3}"
            f"  stored={span['word']!r}"
            f"  sliced={extracted!r}"
        )

    print(f"\nCommon words: {len(common)} 个")
    print("  前10个：", [x["word"] for x in common[:10]])
    print("-" * 100)
PY