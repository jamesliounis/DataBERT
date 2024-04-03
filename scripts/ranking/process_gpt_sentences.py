import json
from pathlib import Path
from collections import Counter
import fire

from data_use import parser

DATA_DIR = Path(__file__).parent.parent.parent / "data"
GPT_SENT_DIR = DATA_DIR / "openai" / "generated_sentences"
sent_files = sorted(GPT_SENT_DIR.glob("*.json"))


def main():
    topics = []
    sentences = []

    for sf in sent_files:
        j = json.loads(sf.read_text())

        try:
            jj = parser.parse_misparsed(j["content"], open='[{"', close=']}]')
            sentences.extend([j["sentence"] for j in jj])
        except:
            print(sf)
            continue

        for s in jj:
            topics.extend(s.get("topic", []))

    c = Counter(topics)
    print(c.most_common())

    SENTS_FILE = DATA_DIR / "training/ranking/gpt_generated_sentences.json"
    sentences = [{"text": s, "label": 1} for s in sentences if s != ""]

    SENTS_FILE.write_text(json.dumps(sentences, indent=2))

    print(f"Saved {len(sentences)} sentences to {SENTS_FILE}")


if __name__ == "__main__":
    # python -m scripts.ranking.process_gpt_sentences
    fire.Fire(main)
