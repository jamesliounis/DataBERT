import copy
import os
from pathlib import Path
import json
from tqdm.auto import tqdm
import backoff
import openai
import tiktoken
import fire

from joblib import Memory


# Load the OpenAI API key from the environment variable.
openai.api_key = os.environ.get('OPENAI_API_KEY')

DATA_DIR = Path(__file__).parent.parent / "data"


# Create a cache directory for the OpenAI API calls.
memory = Memory(DATA_DIR / "cache", verbose=0)


# Create the payloads directory. This is where we store the payloads for
# logging of the interaction with the GPT-3 model.
PAYLOADS_DIR = DATA_DIR / "openai" / "payloads"
PAYLOADS_DIR.mkdir(parents=True, exist_ok=True)
# assert PAYLOADS_DIR.exists(), f"The payloads directory does not exist: {PAYLOADS_DIR}"


_TOKEN_ENC = {
    "gpt-3.5-turbo": tiktoken.encoding_for_model("gpt-3.5-turbo"),
    "gpt-4": tiktoken.encoding_for_model("gpt-4"),
}


def get_tiktoken_model(model):
    global _TOKEN_ENC
    encoding = _TOKEN_ENC.get(model)

    if encoding is None:
        try:
            _TOKEN_ENC[model] = encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            _TOKEN_ENC[model] = encoding = tiktoken.get_encoding("cl100k_base")

    return encoding


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    # https://platform.openai.com/docs/guides/chat/introduction

    encoding = get_tiktoken_model(model)

    if model in ["gpt-3.5-turbo-0301", "gpt-3.5-turbo", "gpt-4"]:  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
@memory.cache
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


class PromptLogger:
    """Class for logging the prompts and responses from the GPT-3 model."""
    def __init__(self, gpt_model_id="gpt-3.5-turbo", total_tokens=4090) -> None:
        self.gpt_model_id = gpt_model_id
        self.total_tokens = total_tokens
        self.prompt_label = None

        self.default_api_kwargs = dict(
            model=self.gpt_model_id,
            temperature=0.7,
            top_p=1,
            n=1,
            stream=False,
        )

    def __str__(self) -> str:
        return self.__repr__()

    def calculate_prompt_tokens(self, message: dict, model_id: str = None) -> int:
        """Calculate the number of tokens in the prompt.
        Args:
            message (dict): The message to measure.
        Returns:
            int: The number of tokens in the prompt.
        """
        prompt_tokens = num_tokens_from_messages(message, model_id or self.gpt_model_id)

        return prompt_tokens

    def send_prompt(self, message, metadata, api_kwargs: dict = None):
        """Send the prompt to the GPT-3 model.
        Args:
            variables (dict): The variables to classify.
        """
        if api_kwargs:
            api_kwargs = {**self.default_api_kwargs, **api_kwargs}
        else:
            api_kwargs = self.default_api_kwargs

        if "tokens" in message:
            del message["tokens"]

        max_tokens = self.total_tokens -  self.calculate_prompt_tokens(message=message, model_id=api_kwargs["model"])
        api_kwargs["max_tokens"] = max_tokens

        # # https://platform.openai.com/docs/api-reference/chat/create
        # response = openai.ChatCompletion.create(
        #     messages=message,
        #     **api_kwargs
        # )
        response = chat_completion_with_backoff(messages=message, **api_kwargs)

        data = dict(
            message=message,
            response=response,
            api_kwargs=api_kwargs,
            metadata=metadata,
        )
        fid = f't{response["created"]}_{response["id"]}.json'

        if self.prompt_label is not None:
            fid = f'{self.prompt_label}-{fid}'

        with open(PAYLOADS_DIR / fid, "w") as f:
            json.dump(data, f, indent=2)

        return dict(
            fid=fid,
            message=message,
            content=response["choices"][0]["message"]["content"],
            response=response,
            metadata=metadata,
            api_kwargs=api_kwargs,
        )


SYSTEM_MESSAGE = """You are an expert in extracting structure information from text. You are also excellent at identifying the use of data and for what policy it was used to inform.

You must not confuse data with indicators. Provide the most precise data name from the text.

You will be asked whether data was used in the text. You will also be asked what data, if any, was used.

Always return the result in a valid JSON format. Do not truncate the output and never generate ... in the response. The output should not raise a JSONDecodeError when loaded in Python.

Example response:  {"data_used": true, "dataset": [{"name": "LSMS", "type": "survey", "country": [<country>], "year": [<year>], "source": <source>}, {"name": "Nighttime lights data", "type": "remote sensing", "country": [], "year": [<year>], "source": <source>}], "themes": [{"theme": "poverty", "source": <source>}], "indicators": [{"indicator": "malnutrition", "source": <source>}, {"indicator": "poverty", "source": <source>}], "analyses": [{"analysis": "poverty measurement", "source": <source>}], "policies": [{"policy": "poverty alleviation", "source": <source>}]}.

You must only fill in the country field if there is an explicit mention of a country name associated with the data; otherwise, leave it empty.

This instruction is very important: you must provide in the "source" field only one sentence from the text supporting your answers.

Was data used in this text? What data was used? What policy was informed by the data?"""


def parse_misparsed(text: str):
    start = text.index('{"')
    end = text.rindex("]}") + 2

    return json.loads(text[start:end])

def main(shortlist_dir: Path):
    shortlist_dir = Path(shortlist_dir).resolve()

    prompt_manager = PromptLogger()

    api_kwargs = dict(
        model="gpt-3.5-turbo",
        temperature=0,
        top_p=0,
        presence_penalty=0,
        frequency_penalty=0,
    )

    assert shortlist_dir.exists()

    for shortlist_file in tqdm(sorted(shortlist_dir.glob("*.shortlist.json"))):
        data = json.loads(shortlist_file.read_text())
        shortlist = data["shortlist"]
        metadata = copy.deepcopy(data)
        del metadata["shortlist"]

        prompt_manager.prompt_label = metadata["doc_id"]

        message = [
            dict(role="system", content=SYSTEM_MESSAGE),
            dict(role="user", content="\n\n=====\n\n".join(shortlist))
        ]

        output = prompt_manager.send_prompt(message, metadata=metadata, api_kwargs=api_kwargs)

        output_file = DATA_DIR / "output" / f"{output['fid']}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        output_file.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    # python -m scripts.process_shortlist --shortlist_dir=data/shortlist/
    fire.Fire(main)
