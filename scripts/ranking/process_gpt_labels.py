from pathlib import Path
import json
import pandas as pd

from tqdm import tqdm
import fire


DATA_DIR = Path(__file__).parent.parent.parent / "data"
assert DATA_DIR.exists(), f"The data directory does not exist: {DATA_DIR}"


def main():
    # Create the payloads directory. This is where we store the payloads for
    # logging of the interaction with the GPT-3 model.
    PAYLOADS_DIR = DATA_DIR / "openai" / "label_ranking"

    RANKING_DIR = Path("data/training/ranking")
    all_df = pd.read_excel((RANKING_DIR / "ranking_shortlist.xlsx"))
    sent_ids = []

    for path in PAYLOADS_DIR.glob("*.json"):
        data = json.loads(path.read_text())
        data = data["content"].split("\n")[0]

        try:
            data = json.loads(data)
            data = pd.DataFrame(data)
            data = data.loc[data["data_mentioned"], "sent_id"]

            sent_ids.extend(data.tolist())

        except json.decoder.JSONDecodeError:
            print("Error:", path)
            continue

    all_df["gpt_label"] = False
    all_df.loc[all_df["sent_id"].isin(sent_ids), "gpt_label"] = True

    all_df.to_excel(RANKING_DIR / "ranking_shortlist_gpt.xlsx", index=False)

if __name__ == "__main__":
    # python -m scripts.ranking.process_gpt_labels
    fire.Fire(main)
