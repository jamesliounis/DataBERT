import json
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import fire

from openai_tools.prompt import PromptZeros


DATA_DIR = Path(__file__).parent.parent.parent / "data"
assert DATA_DIR.exists(), f"The data directory does not exist: {DATA_DIR}"

PAYLOADS_DIR = DATA_DIR / "openai"
TASK_LABEL = "label_ranking"
MODEL = "gpt-3.5-turbo"

PROMPT = """Forget all previous instructions.

Always return the result in a valid JSON format. Do not truncate the output and never generate ... in the response. The output should not raise a JSONDecodeError when loaded in Python.

Given a list of sentences extracted from papers, extract the sentences that explicitly mention a dataset used in the paper. Only include sentences where the dataset is clearly and unambiguously named, such as LSMS or DHS. Exclude sentences that mention software used for data analysis, equipment used to collect data, or other tools that are not datasets. Also, exclude sentences that only cite a reference without mentioning the dataset. Also exclude sentences that refer to data computed by the author without mentioning the dataset used to derive it.

Your task is to return the sentence ids of the identified sentences as a JSON array along with the reason why you included them. The output format should be [{"sent_id": s_XXXXX, "data_mentioned": true, "reason": <dataset name>}]. If a sentence mentions a dataset clearly and unambiguously, set "data_mentioned" to true and fill in the "reason" field with the name of the dataset

Do not explain.

Text:"""


prompt_service = PromptZeros(
    payloads_dir=PAYLOADS_DIR,
    task_label=TASK_LABEL,
    model=MODEL,
    total_tokens=4050,
)


def main():
    # Load previously labeled data.
    message_hashes = []

    for p in (PAYLOADS_DIR / TASK_LABEL).glob("*/*.json"):
        message_hash = json.loads(p.read_text()).get("message_hash")
        if message_hash:
            message_hashes.append()

    RANKING_DIR = Path("data/training/ranking")
    all_df = pd.read_excel((RANKING_DIR / "ranking_shortlist.xlsx"))

    group_size = 25

    for i in tqdm(sorted(range(0, len(all_df) + 1, group_size))):
        sub = all_df[["sent_id", "text"]].iloc[i:i + group_size]

        sub = sub.dropna(subset=["text"])

        f = sub["text"].str.replace("[^a-zA-Z\d]+", "", regex=True)
        f = f.fillna("")
        sub = sub[f.map(len) > 10]

        if sub.empty:
            continue

        sub = sub.apply(": ".join, axis=1).tolist()
        content = "\n\n".join([PROMPT] + sub)

        pack = prompt_service.build_message(
            user_content=content,
            user_template=None,
            system_content=None,
        )

        if pack["message_hash"] in message_hashes:
            continue

        prompt_service.prompt_label = "gpt-rank-label"
        prompt_service.send_prompt(
            user_content=content,
            user_template=None,
            system_content=None,
            metadata=None,
        )


if __name__ == "__main__":
    # python -m scripts.ranking.gpt_label_ranking_data
    fire.Fire(main)
