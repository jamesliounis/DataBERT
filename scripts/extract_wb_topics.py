import requests
from bs4 import BeautifulSoup
import fire


def main():
    url = "https://www.worldbank.org/en/topic"

    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")
    topics = soup.find_all("div", attrs={"class": "topic-list"})

    topic_names = sorted({topic.find("h4").find("a").text.strip("\xa0»") for topic in topics})

    print(topic_names)


if __name__ == "__main__":
    # python -m scripts.extract_wb_topics
    fire.Fire(main)
