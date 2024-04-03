


# Example sentence generation

Source: https://chat.openai.com/c/fb61e33e-1812-4816-896a-00b8ff2cdbfb

```Python

system_message = """You are an expert in following instructions. You are also an expert on various topics, especially on socio-economic development. You know how to read and understand sentences, and based from these sentences generate similar looking sentences but on different topics.

The following sentences are incorrect:
- We analyze the relationship between social media usage and mental health using data from the Pew Research Center's Internet and American Life Project and the National Survey on Drug Use and Health (NSDUH).

You will generate sentences indicating the use of a dataset. Always mention specific datasets in the sentence.

You always return your response in JSON format that can be loaded in Python using `json.loads`. The format looks like:  [{"sentence": <sentence>, "dataset": [<dataset>]}]"""

example_output = """[{"sentence": "This study draws on data from the DHS conducted in Sub-Saharan African countries with at least two rounds of data.", "dataset": ["DHS"]}, {"sentence": "This paper uses administrative data on electricity billing records from Ghana and Rwanda.", "dataset": ["administrative data on electricity billing records"]}, {"sentence": "We complement the electricity data with monthly data on temperature and total precipitation from the Copernicus Climate Change Service.", "dataset": ["Copernicus Climate Change Service"]}, {"sentence": "We use two main sources of data, both of which are novel: a dataset on the universe of Indonesian exporters in the period 2014-18 and a time-varying dataset of NTMs applied on Indonesian imports.", "dataset": ["dataset on the universe of Indonesian exporters in the period 2014-18", "time-varying dataset of NTMs applied on Indonesian imports"]}, {"sentence": "To this end, we used multi-modal imaging and neuropsychological battery data available in the Alzheimer's Disease Neuroimaging Initiative (ADNI) to investigate the relationship between cross-sectional measures of tau, cortical thickness, and different aspects of cognition.", "dataset": ["Alzheimer's Disease Neuroimaging Initiative (ADNI)"]}]"""


user_message = """Generate 10 sentences on various topics that use datasets.

Consider generating complex sentence structure. Also generate sentences where multiple datasets are mentioned. Try to generate challenging and somewhat ambiguous sentences.

<example_output>

You will use the seed to always generate reproducible set of output, seed = 1029."""


```


