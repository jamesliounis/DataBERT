import os
import openai
import json
from pathlib import Path
from tqdm import tqdm
import hashlib
import uuid
import fire
import backoff

from openai_tools.utils import num_tokens_from_messages

# Load the OpenAI API key from the environment variable.
openai.api_key = os.environ.get('OPENAI_API_KEY')

DATA_DIR = Path(__file__).parent.parent / "data"

# Create the payloads directory. This is where we store the payloads for
# logging of the interaction with the GPT-3 model.
PAYLOADS_DIR = DATA_DIR / "openai" / "processed_shortlist"
PAYLOADS_DIR.mkdir(parents=True, exist_ok=True)
# assert PAYLOADS_DIR.exists(), f"The payloads directory does not exist: {PAYLOADS_DIR}"


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


PROMPT = """Forget all previous instructions.

You are given a list of sentences extracted from a paper quoted with triple backticks below.

Your task is to identify the datasets and the data type used in the paper, for example LSMS, DHS, etc. You must only consider sentences that explicitly mention a name of a dataset. You are to return the result in JSON array format with keys "data_name", "data_type",  "is_quantitative", "sentence".

Choose the data_type from this !!!!!-delimited list: !!!!!Surveys!!!!!Census data!!!!!Administrative data!!!!!Remote sensing data!!!!!Geographic Information System (GIS) data!!!!!Financial and economic data!!!!!Climate data!!!!!Health data!!!!!Education data!!!!!Social media data!!!!!Mobile phone data!!!!!Transactional data!!!!!Big data!!!!!Qualitative data!!!!!Experimental data!!!!!Case studies!!!!!Observational data!!!!!Longitudinal data!!!!!Cross-sectional data!!!!!Time series data!!!!!Other"""


DEFAULT_API_KWARGS = dict(
    model="gpt-3.5-turbo",
    temperature=0,
    top_p=0,
    n=1,
    stream=False,
    presence_penalty=0,
)


def generate_structure(shortlist, metadata: dict = None, k: int = 25, **kwargs):
    api_kwargs = {**DEFAULT_API_KWARGS, **kwargs}
    _shortlist = shortlist[:k]
    _shortlist = "\n\n".join([s['text'] for s in _shortlist if s['text'] != ''])

    # Create a unique ID for this combination of prompt, API arguments, and k.
    k_dict = {**api_kwargs}
    k_dict["prompt"] = PROMPT
    k_dict["k"] = k
    jkey = json.dumps(k_dict, sort_keys=True)

    out_id = f"{uuid.UUID(hashlib.md5(jkey.encode('utf-8')).hexdigest())}"

    # Create the content for the user.
    user_content = "\n\n".join([PROMPT, f"```{_shortlist}```"])

    message = [
        dict(role="system", content=""),
        dict(role="user", content=user_content),
    ]

    total_tokens = num_tokens_from_messages(message, model=api_kwargs["model"])

    max_tokens = 4090 - total_tokens
    api_kwargs["max_tokens"] = max_tokens

    response = chat_completion_with_backoff(messages=message, **api_kwargs)
    data = dict(
        message=message,
        response=response,
        content=response["choices"][0]["message"]["content"],
        api_kwargs=api_kwargs,
        metadata=metadata or {},
    )
    fid = f't{response["created"]}_{response["id"]}.json'
    fid = f'gpt-struct_shorts_{fid}'

    out_file = PAYLOADS_DIR / out_id / fid
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)


def main(dataset, k=25, **kwargs):
    data_dir = DATA_DIR / dataset / "shortlist"
    assert data_dir.exists(), f"The data directory does not exist: {data_dir}"

    for path in tqdm(sorted(data_dir.glob("*.json"))):
        data = json.loads(path.read_text())
        shortlist = data["shortlist"]
        metadata = dict(fname=path.name, dataset=dataset, shortlist=shortlist)

        generate_structure(shortlist, metadata, k=k, **kwargs)


if __name__ == "__main__":
    # python -m scripts.generate_gpt_structured_from_shortlist --dataset=prwp --k=25
    fire.Fire(main)
