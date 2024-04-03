import os
import json
from pathlib import Path
import backoff
from tqdm.auto import tqdm
import openai
import fire


# Load the OpenAI API key from the environment variable.
openai.api_key = os.environ.get('OPENAI_API_KEY')

DATA_DIR = Path(__file__).parent.parent / "data"

# Create the payloads directory. This is where we store the payloads for
# logging of the interaction with the GPT-3 model.
PAYLOADS_DIR = DATA_DIR / "openai" / "generated_sentences"
PAYLOADS_DIR.mkdir(parents=True, exist_ok=True)
# assert PAYLOADS_DIR.exists(), f"The payloads directory does not exist: {PAYLOADS_DIR}"


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


PROMPT = """Forget all previous instructions.

You are an expert in following instructions. You are also an expert on various topics, especially on socio-economic development. You know how to read and understand sentences, and based from these sentences generate similar looking sentences but on different topics.

You will generate sentences indicating the use of a dataset and complementary datasets. Always mention specific datasets in the sentence.

You always return your response in JSON format that can be loaded in Python using `json.loads`. The format looks like:  [{"sentence": <sentence>, "dataset": [<dataset>], "topic": [<topic>]}]

Use topic only from this list separated by !!!!!: Agriculture and Food!!!!!Climate Change!!!!!Competitiveness!!!!!Debt!!!!!Digital Development!!!!!Disaster Risk Management !!!!!Education!!!!!Energy!!!!!Environment !!!!!Extractive Industries!!!!!Financial Inclusion!!!!!Financial Sector!!!!!Fragility, Conflict, and Violence!!!!!Gender!!!!!Governance!!!!!Health!!!!!Inequality and Shared Prosperity!!!!!Infrastructure!!!!!Jobs & Development!!!!!Macroeconomics!!!!!Migration!!!!!Nutrition!!!!!Poverty!!!!!Public-Private Partnerships!!!!!Regional Integration!!!!!Social Protection!!!!!Social Sustainability and Inclusion!!!!!Trade!!!!!Transport!!!!!Urban Development!!!!!Water

Generate 10 sentences on various topics that use datasets. The sentence lengths must have large variation with each other.

Consider generating complex sentence structure. Also generate sentences where multiple datasets are mentioned. Try to generate challenging and somewhat ambiguous sentences.

Example output: [{"sentence": "This study draws on data from the DHS conducted in Sub-Saharan African countries with at least two rounds of data.", "dataset": ["DHS"], "topic": ["Health"]}, {"sentence": "This paper uses administrative data on electricity billing records from Ghana and Rwanda.", "dataset": ["administrative data on electricity billing records"], "topic": ["Energy"]}, {"sentence": "We complement the electricity data with monthly data on temperature and total precipitation from the Copernicus Climate Change Service.", "dataset": ["Copernicus Climate Change Service"], "topic": ["Energy", "Climate Change"]}, {"sentence": "We use two main sources of data, both of which are novel: a dataset on the universe of Indonesian exporters in the period 2014-18 and a time-varying dataset of NTMs applied on Indonesian imports.", "dataset": ["dataset on the universe of Indonesian exporters in the period 2014-18", "time-varying dataset of NTMs applied on Indonesian imports"], "topic": ["Trade"]}, {"sentence": "To this end, we used multi-modal imaging and neuropsychological battery data available in the Alzheimer's Disease Neuroimaging Initiative (ADNI) to investigate the relationship between cross-sectional measures of tau, cortical thickness, and different aspects of cognition.", "dataset": ["Alzheimer's Disease Neuroimaging Initiative (ADNI)"], "topic": ["Health"]}]

You will use the seed to always generate reproducible set of output, seed = 42."""

DEFAULT_API_KWARGS = dict(
    model="gpt-3.5-turbo",
    temperature=1,
    top_p=1,
    n=1,
    stream=False,
    presence_penalty=1,
)

def generate_sentences(**kwargs):
    api_kwargs = {**DEFAULT_API_KWARGS, **kwargs}

    message = [
        dict(role="system", content=""),
        dict(role="user", content=PROMPT),
    ]

    response = chat_completion_with_backoff(messages=message, **api_kwargs)
    data = dict(
        message=message,
        response=response,
        content=response["choices"][0]["message"]["content"],
        api_kwargs=api_kwargs,
    )
    fid = f't{response["created"]}_{response["id"]}.json'
    fid = f'gpt-sents_{fid}'

    with open(PAYLOADS_DIR / fid, "w") as f:
        json.dump(data, f, indent=2)


def main(**kwargs):
    if "n" in kwargs:
        print(f"Multiple `n` is not supported. Ignoring `n`={kwargs['n']}")
        kwargs.pop("n")

    if "rounds" in kwargs:
        rounds = kwargs.pop("rounds")
        for _ in tqdm(range(rounds)):
            generate_sentences(**kwargs)
    else:
        generate_sentences(**kwargs)


if __name__ == "__main__":
    # python -m scripts.generate_gpt_sentences --rounds=1
    fire.Fire(main)
