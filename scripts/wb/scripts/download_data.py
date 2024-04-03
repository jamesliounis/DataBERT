import os
import json
import requests
import backoff
import pandas as pd
# We use the report numbers from the "../data/Results Data.xlsx" file.
from tqdm.auto import tqdm
import fire


@backoff.on_exception(backoff.expo, [requests.exceptions.RequestException], max_tries=6)
def get_metadata(repnb: str, save_path: str = None):
    url = f"https://search.worldbank.org/api/v2/wds?format=json&repnb_exact={repnb}&docty=Policy%20Research%20Working%20Paper&lang_exact=English"

    r = requests.get(url)

    if r.status_code != 200:
        raise ValueError(f"Failed to get metadata for {repnb}.")
    else:
        docs = r.json().get("documents")
        docs.pop("facets", None)
        assert len(docs) == 1, f"Expected only 1 document for {repnb}."

        key = list(docs.keys())[0]
        metadata = docs[key]

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(metadata, f, indent=2)

    return metadata

def download_all(rows: int = None):
    wb_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_file = os.path.join(wb_dir, "data", "Results Data.xlsx")

    if os.path.exists(results_file):
        df = pd.read_excel(results_file)
        repnbs = sorted(df["Report No."].unique())
    else:
        assert rows is not None, "rows must be provided if the results file does not exist."

        # Get the report numbers from the API
        url = f"https://search.worldbank.org/api/v2/wds?format=json&docty=Policy%20Research%20Working%20Paper&lang_exact=English&rows={rows}"
        data = requests.get(url).json()

        assert data["rows"] >= rows, f"Expected at least {rows} rows, but got {data['rows']}."
        repnbs = [doc["repnb"] for doc in data["documents"]["rows"]]

    for repnb in tqdm(repnbs):
        save_path = os.path.join(wb_dir, "data", "metadata", f"{repnb}.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if os.path.exists(save_path):
            metadata = json.load(open(save_path, "r"))
        else:
            metadata = get_metadata(repnb, save_path=save_path)

        # Get pdf link
        pdf_link = metadata["pdfurl"]

        # Download pdf
        pdf_save_path = os.path.join(wb_dir, "data", "pdfs", f"{repnb}.pdf")
        os.makedirs(os.path.dirname(pdf_save_path), exist_ok=True)

        if not os.path.exists(pdf_save_path):
            r = requests.get(pdf_link)

            if r.status_code != 200:
                raise ValueError(f"Failed to download pdf for {repnb}.")
            else:
                with open(pdf_save_path, "wb") as f:
                    f.write(r.content)

if __name__ == "__main__":
    # pipenv run python scripts/wb/scripts/download_data.py --rows=100
    fire.Fire(download_all)
