# data-use


```Python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import NLTKTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.indexes import VectorstoreIndexCreator


sentence_splitter = NLTKTextSplitter(chunk_size=1000)
tiktoken_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)
# path = "~/WBG/ChatPRWP/data/pdf/curated-en-099829402282385780-pdf-idu03708fc080cc11042590bd3b0e01f66981ddd.pdf"
# path = "~/WBG/ChatPRWP/data/pdf/curated-en-099032406232218786-pdf-idu052aadee5089b304b0b0880605a64276a8e34.pdf"
path = "~/WBG/ChatPRWP/data/pdf/curated-en-099047501242316215-pdf-idu0458baf2e0e6ed045bb095e70cbc841f24bed.pdf"
loader = PyPDFLoader(path)
pages = loader.load_and_split()

sentence_batch = sentence_splitter.split_documents(pages)
token_batch = tiktoken_splitter.split_documents(sentence_batch)

passage_embeddings = HuggingFaceInstructEmbeddings(
    embed_instruction="Represent the passage for retrieval, Input: ",
    query_instruction="Represent the question for retrieving relevant passage, Input: ",
)


index = VectorstoreIndexCreator(
    embedding=passage_embeddings,
    text_splitter=tiktoken_splitter,
).from_loaders([loader])


res = index.vectorstore.similarity_search("Was dataset used in the paper?")
print("\n\n".join([rs.page_content for rs in ref_res]))
```

Dependencies:

- chromadb
- pypdf
- nltk
- tiktoken


# Using the S2ORCTextSplitter

Grobid server must be running. You can use the docker image.

```bash
docker run --rm --init -p 127.0.0.1:8070:8070 -p 127.0.0.1:8071:8071 ucrel/grobid:0.6.1
```

Note that the first time you run this, it will take a while to load the models. Converting the first file will also take a while. Subsequent conversions will be much faster.


# Data use pipeline

```Python
from data_use.indexes import vectorstore as dvectorstore
from data_use.document_loaders import pdf
from data_use import text_splitter as ts

doc_dir = "/Users/avsolatorio/WBG/ChatPRWP/data/pdf/"
doc_name = "curated-en-099032406232218786-pdf-idu052aadee5089b304b0b0880605a64276a8e34.pdf"
doc_name = "curated-en-992471468211165611-pdf-wps4889.pdf"
doc_name = "curated-en-099005106232213093-pdf-idu0ef49c6600ba1e04eca0a30c04d1e2aa727f4.pdf"
doc_name = "curated-en-099047501242316215-pdf-idu0458baf2e0e6ed045bb095e70cbc841f24bed.pdf"
doc_name = "curated-en-099829402282385780-pdf-idu03708fc080cc11042590bd3b0e01f66981ddd.pdf"
doc_fname = doc_dir + doc_name
# doc_name = "curated-en-099829402282385780-pdf-idu03708fc080cc11042590bd3b0e01f66981ddd.pdf"

# "/Users/avsolatorio/WBG/ChatPRWP/data/pdf/curated-en-099005106232213093-pdf-idu0ef49c6600ba1e04eca0a30c04d1e2aa727f4.pdf")
du = ts.DataUseTextSplitter(tokenizer=enc, separator="\n\n", chunk_size=512)

loader = pdf.S2ORCPDFLoader(doc_fname)
docs = loader.load()
documents = du.aggregate_documents(docs)

passage_embeddings = HuggingFaceInstructEmbeddings(
    embed_instruction="Represent the passage for retrieval; Input: ",
    query_instruction="Represent the question for retrieving passages that are the most relevant; Input: ",
)

index = dvectorstore.VectorstoreIndexCreator(
    embedding=passage_embeddings,
).from_documents(documents)

print("\n\n".join([doc[0].page_content for doc in index.vectorstore.similarity_search_with_score("Was data or dataset used? Was data or dataset collected and analyzed?")]))
```

# PRWP Test

Get the 100 most recent PRWP documents. Generate the shortlist of snippets for each document that likely contains the answer to the question. Then, store the shortlist in a file.

Use the shortlist as input to the GPT-3.5 / GPT-4 model. The model will be instructed to extract any data used in the paper as well as some additional information.


```Python
import json
from pathlib import Path
import tiktoken
from langchain.embeddings import HuggingFaceInstructEmbeddings

from data_use.indexes import vectorstore as dvectorstore
from data_use.document_loaders import pdf
from data_use import text_splitter as ts

passage_embeddings = HuggingFaceInstructEmbeddings(
    embed_instruction="Represent the passage for retrieval, Input: ",
    query_instruction="Represent the question for retrieving relevant passage, Input: ",
)

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
du = ts.DataUseTextSplitter(tokenizer=enc, separator="\n\n", chunk_size=512)

proj_root = Path("/Users/avsolatorio/WBG/ChatPRWP")
data_dir = proj_root / "data"
pdf_dir = data_dir / "pdf"
shortlist_dir = data_dir / "shortlist"
shortlist_dir.mkdir(parents=True, exist_ok=True)

cached_json_dir = data_dir / "s2orc-output_dir" / "prwp"

metadata_fname = data_dir / "raw" / "prwp_metadata_full.json"
metadata = json.loads(metadata_fname.read_text())

# Get the 100 most recent PRWP documents
prwp_docs = []

for doc_id in sorted(metadata, key=lambda x: metadata[x].get("datestored", metadata[x].get("docdt", metadata[x].get("last_modified_date", ""))), reverse=True):
    pdf_path = pdf_dir / f"{doc_id}.pdf"
    if pdf_path.exists():
        prwp_docs.append(pdf_path)

    if len(prwp_docs) >= 100:
        break


# Generate the shortlist of snippets for each document that likely contains the answer to the question.
# Then, store the shortlist in a file.

query = "Was data or dataset used? Was data or dataset collected and analyzed?"

for doc_path in prwp_docs:
    loader = pdf.S2ORCPDFLoader(doc_path)
    docs = loader.load()
    documents = du.aggregate_documents(docs)

    index = dvectorstore.VectorstoreIndexCreator(
        embedding=passage_embeddings,
    ).from_documents(documents)

    shortlist = index.vectorstore.similarity_search(query)
    shortlist = [rs.page_content for rs in shortlist]

    shortlist_fname = shortlist_dir / f"{doc_path.stem}.shortlist.json"
    shortlist_fname.write_text(json.dumps(dict(query=query, shortlist=shortlist, doc_id=doc_path.stem), indent=2))

```

loader = pdf.S2ORCPDFLoader(doc_fname)
docs = loader.load()
documents = du.aggregate_documents(docs)

index = dvectorstore.VectorstoreIndexCreator(
    embedding=passage_embeddings,
).from_documents(documents)


```JSON
{"data_used": true, "data": [{"name": "Electricity billing records", "type": "administrative data", "country": ["Ghana", "Rwanda"], "year": ["2018", "2019", "2020"]}, {"name": "Copernicus Climate Change Service", "type": "climate data", "country": [], "year": []}], "theme": ["energy consumption", "COVID-19"], "indicator": ["electricity consumption", "COVID-19 impact"], "analysis": "causal estimation of the effect of the COVID-19 pandemic on electricity consumption", "policy": ["electricity subsidies", "provision of free food and household essentials to households affected by lockdowns", "social interventions such as conferences, workshops, funerals, festivals, political rallies, and religious activities", "stay-at-home orders and associated home-based work"]}
```


# PROMPT

```Python
system_message = """You are an expert in extracting structure information from text. You are also excellent at identifying the use of data and for what policy it was used to inform.

You must not confuse data with indicators.

You will be asked whether data was used in the text. You will also be asked what data, if any, was used.

Always return the result in a valid JSON format. Do not truncate the output and never generate ... in the response. The output should not raise a JSONDecodeError when loaded in Python.

Example response:  {"data_used": true, "data": [{"name": "LSMS", "type": "survey", "country": ["India"], "year": ["2017"], "source": "XXX"}, {"name": "Nighttime lights data", "type": "remote sensing", "country": [], "year": ["2020"], "source": "YYY"}], "theme": ["poverty"], "indicator": ["malnutrition", "poverty"], "analysis": "poverty measurement", "policy": ["poverty alleviation"]}.

You must only fill in the country field if there is an explicit mention of a country name associated with the data; otherwise, leave it empty. You must also provide in the "source" field the sentence from the text indicating the use of data.

Was data used in this text? What data was used? What policy was informed by the data?
"""

"""You are an expert in extracting structure information from text. You are also excellent at identifying the use of data and for what policy it was used to inform.

You must not confuse data with indicators. Provide the most precise data name from the text.

You will be asked whether data was used in the text. You will also be asked what data, if any, was used.

Always return the result in a valid JSON format. Do not truncate the output and never generate ... in the response. The output should not raise a JSONDecodeError when loaded in Python.

Example response:  {"data_used": true, "data": [{"name": "LSMS", "type": "survey", "country": ["India"], "year": ["2017"], "source": "XXX"}, {"name": "Nighttime lights data", "type": "remote sensing", "country": [], "year": ["2020"], "source": "YYY"}], "theme": ["poverty"], "indicator": ["malnutrition", "poverty"], "analysis": "poverty measurement", "policy": ["poverty alleviation"]}.

You must only fill in the country field if there is an explicit mention of a country name associated with the data; otherwise, leave it empty. You must also provide in the "source" field the sentence from the text indicating the use of data.

Was data used in this text? What data was used? What policy was informed by the data?
"""


"""
You are an expert in extracting structure information from text. You are also excellent at identifying the use of data and for what policy it was used to inform.

You must not confuse data with indicators. Provide the most precise data name from the text.

You will be asked whether data was used in the text. You will also be asked what data, if any, was used.

Always return the result in a valid JSON format. Do not truncate the output and never generate ... in the response. The output should not raise a JSONDecodeError when loaded in Python.

Example response:  {"data_used": true, "dataset": [{"name": "LSMS", "type": "survey", "country": ["India"], "year": ["2017"], "source": "XXX"}, {"name": "Nighttime lights data", "type": "remote sensing", "country": [], "year": ["2020"], "source": "XXX"}], "themes": [{"theme": "poverty", "source": "XXX"}], "indicators": [{"indicator": "malnutrition", "source": "XXX"}, {"indicator": "poverty", "source": "XXX"}], "analyses": [{"analysis": "poverty measurement", "source": "XXX"}], "policies": [{"policy": "poverty alleviation", "source": "XXX"}]}.

You must only fill in the country field if there is an explicit mention of a country name associated with the data; otherwise, leave it empty. You must also provide in the "source" field the sentence from the text indicating the use of data.

Was data used in this text? What data was used? What policy was informed by the data?
"""


"""
You are an expert in extracting structure information from text. You are also excellent at identifying the use of data and for what policy it was used to inform.

You must not confuse data with indicators. Provide the most precise data name from the text.

You will be asked whether data was used in the text. You will also be asked what data, if any, was used.

Always return the result in a valid JSON format. Do not truncate the output and never generate ... in the response. The output should not raise a JSONDecodeError when loaded in Python.

Example response:  {"data_used": true, "dataset": [{"name": "LSMS", "type": "survey", "country": ["India"], "year": ["2017"], "source": <source>}, {"name": "Nighttime lights data", "type": "remote sensing", "country": [], "year": ["2020"], "source": <source>}], "themes": [{"theme": "poverty", "source": <source>}], "indicators": [{"indicator": "malnutrition", "source": <source>}, {"indicator": "poverty", "source": <source>}], "analyses": [{"analysis": "poverty measurement", "source": <source>}], "policies": [{"policy": "poverty alleviation", "source": <source>}]}.

You must only fill in the country field if there is an explicit mention of a country name associated with the data; otherwise, leave it empty.

You must also provide in the "source" field the sentence from the text indicating the use of data.

Was data used in this text? What data was used? What policy was informed by the data?"""


response = """{"data_used": true, "dataset": [{"name": "Electricity billing records", "type": "administrative", "country": ["Ghana", "Rwanda"], "year": ["2018", "2019", "2020"], "source": "This paper uses administrative data on electricity billing records from Ghana and Rwanda."}, {"name": "Copernicus Climate Change Service", "type": "climate data", "country": [], "year": [], "source": "We complement the electricity data with monthly data on temperature and total precipitation from the Copernicus Climate Change Service."}], "themes": [{"theme": "COVID-19", "source": "Causal estimation of the effect of the COVID-19 pandemic on electricity consumption is beset with at least two challenges of identification."}, {"theme": "energy demand", "source": "Temperature and rainfall patterns are key drivers of energy demand around the world, and given that these weather patterns vary based on the time of the year, monthly changes in electricity consumption to a large extent are influenced by the prevailing weather conditions."}, {"theme": "electricity subsidies", "source": "The electricity subsidies are particularly important to this paper."}], "indicators": [{"indicator": "electricity consumption", "source": "To address these issues, we follow the approach of Irwin et al. (2021) and implement a difference-in-difference design by comparing the differences in electricity consumption during the post-COVID months and consumption levels in the same month in the previous years (2018-2019) with the average consumption in the months just before the pandemic (i.e. January and February 2020) and the years before (2018-2019)."}], "analyses": [{"analysis": "difference-in-difference design", "source": "To address these issues, we follow the approach of Irwin et al. (2021) and implement a difference-in-difference design by comparing the differences in electricity consumption during the post-COVID months and consumption levels in the same month in the previous years (2018-2019) with the average consumption in the months just before the pandemic (i.e. January and February 2020) and the years before (2018-2019)."}], "policies": [{"policy": "electricity subsidies", "source": "The subsidy program offered a 100% subsidy to lifeline (low-income) customers who consume 0-50 kWh of electricity a month, while non-lifeline customers (consuming above 50 kWh a month) would enjoy a 50% subsidy (Berkouwer et al., 2022)."}]}"""


"""{"data_used": true, "dataset": [{"name": "Electricity billing records", "type": "administrative", "country": ["Ghana", "Rwanda"], "year": ["2018", "2019", "2020"], "source": "This paper uses administrative data on electricity billing records from Ghana and Rwanda."}, {"name": "Copernicus Climate Change Service", "type": "climate data", "country": [], "year": [], "source": "We complement the electricity data with monthly data on temperature and total precipitation from the Copernicus Climate Change Service."}], "themes": [{"theme": "COVID-19", "source": "Causal estimation of the effect of the COVID-19 pandemic on electricity consumption is beset with at least two challenges of identification."}, {"theme": "energy demand", "source": "Temperature and rainfall patterns are key drivers of energy demand around the world, and given that these weather patterns vary based on the time of the year, monthly changes in electricity consumption to a large extent are influenced by the prevailing weather conditions."}, {"theme": "electricity subsidies", "source": "The subsidy program offered a 100% subsidy to lifeline (low-income) customers who consume 0-50 kWh of electricity a month, while non-lifeline customers (consuming above 50 kWh a month) would enjoy a 50% subsidy (Berkouwer et al., 2022)."}], "indicators": [{"indicator": "electricity consumption", "source": "To address these issues, we follow the approach of Irwin et al. (2021) and implement a difference-in-difference design by comparing the differences in electricity consumption during the post-COVID months and consumption levels in the same month in the previous years (2018-2019) with the average consumption in the months just before the pandemic (i.e. January and February 2020) and the years before (2018-2019)."}], "analyses": [{"analysis": "difference-in-difference design", "source": "To address these issues, we follow the approach of Irwin et al. (2021) and implement a difference-in-difference design by comparing the differences in electricity consumption during the post-COVID months and consumption levels in the same month in the previous years (2018-2019) with the average consumption in the months just before the pandemic (i.e. January and February 2020) and the years before (2018-2019)."}], "policies": [{"policy": "electricity subsidies", "source": "The subsidy program offered a 100% subsidy to lifeline (low-income) customers who consume 0-50 kWh of electricity a month, while non-lifeline customers (consuming above 50 kWh a month) would enjoy a 50% subsidy (Berkouwer et al., 2022)."}]}"""


system_message = """You are an expert in extracting structure information from text. You are also excellent at identifying the use of data and for what policy it was used to inform.

You must not confuse data with indicators. Provide the most precise data name from the text.

You will be asked whether data was used in the text. You will also be asked what data, if any, was used.

Always return the result in a valid JSON format. Do not truncate the output and never generate ... in the response. The output should not raise a JSONDecodeError when loaded in Python.

Example response:  {"data_used": true, "dataset": [{"name": "LSMS", "type": "survey", "country": [<country>], "year": [<year>], "source": <source>}, {"name": "Nighttime lights data", "type": "remote sensing", "country": [], "year": [<year>], "source": <source>}], "themes": [{"theme": "poverty", "source": <source>}], "indicators": [{"indicator": "malnutrition", "source": <source>}, {"indicator": "poverty", "source": <source>}], "analyses": [{"analysis": "poverty measurement", "source": <source>}], "policies": [{"policy": "poverty alleviation", "source": <source>}]}.

You must only fill in the country field if there is an explicit mention of a country name associated with the data; otherwise, leave it empty.

You must also provide in the "source" field the sentence from the text indicating the use of data.

Was data used in this text? What data was used? What policy was informed by the data?"""


# enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
# du = ts.DataUseTextSplitter(tokenizer=enc, separator="\n\n", chunk_size=512)
# shortlist = index.vectorstore.similarity_search("Was data or dataset used? Was data or dataset collected?", k=4)
# print("\n===\n".join([o.page_content for o in shortlist]))

## All GPT-3.5 parameters are zero.
user_message = """Notes: Dependent variable is the log of monthly electricity consumption.

Climate Ctrls include the monthly average temperature and total precipitation.

Standard errors clustered at district level in parenthesis.

* Significant at 10 percent level; * * Significant at 5 percent level; * * * Significant at 1 percent level Notes: Dependent variable is the log of monthly electricity consumption.

Climate Ctrls include the monthly average temperature and total precipitation.

Standard errors clustered at district level in parenthesis.

* Significant at 10 percent level; * * Significant at 5 percent level; * * * Significant at 1 percent level 2019 2019 Notes: Dependent variable is the log of monthly electricity consumption.

Climate Ctrls include the monthly average temperature and total precipitation.

Standard errors clustered at district level in parenthesis.

* Significant at 10 percent level * * Significant at 5 percent level * * * Significant at 1 percent level 2019 Notes: Dependent variable is the log of monthly electricity consumption.

Climate Ctrls include the monthly average temperature and total precipitation.

Standard errors clustered at district level in parenthesis.

* Significant at 10 percent level * * Significant at 5 percent level * * * Significant at 1 percent level 2019 Notes: Dependent variable is the log of monthly electricity consumption.

Climate Ctrls include the monthly average temperature and total precipitation.

Standard errors clustered at district level in parenthesis.

* Significant at 10 percent level * * Significant at 5 percent level * * * Significant at 1 percent level

As of September 22, 2022, there was a total of 168,813 confirmed cases, 167,206 recoveries, and 1,459 deaths.

Tragic as these figures are, particularly for those who lost loved ones, the fatality rate in Ghana pales in comparison to the global average.

Ghana owes the relatively low confirmed fatalities to a suite of measures that were instituted before the virus reached the shores of the country and social interventions which were implemented once the first two cases were confirmed 3 Such as conferences, workshops, funerals, festivals, political rallies, and religious activities

Knutsen et al.

(2017) for instance, compares corruption experiences of citizens living in close proximity to gold mining firms with the corruption experiences of citizens living in areas close to a yet to be opened mine in a difference-in-difference design to estimate the effect of mining on corruption in Africa.

i.e., January and February
===
Interestingly, Figure 1c shows a consistent decline in non-residential electricity consumption from March through December 2020.

The largest decline was recorded in one month after the start of the SAH (lockdown), with consumption falling by 24% relative to the average consumption during the same period in the preceding years.

16 Despite signs of recovery from months 2-9 after the start of the SAH order, consumption levels never reached the pre-pandemic levels.

The effect on the industrial sector, however, appears to be short-lived ( Figure 1d ).

Consumption fell significantly by 13.5% and 21% during the first and second months of the SAH, but bounced back to pre-pandemic levels after the lifting of the SAH.

17 To fully understand the significant decline in non-residential electricity consumption during the pandemic, we conduct separate analyses for the main groups within the nonresidential category, namely: hotels, health centers, and others.

The DID estimates are presented in Table 3 and Figure 2 .

18 The results on the effects of COVID on electricity consumption of commercial units (Figure 2a ) mimics the changes in consumption for the overall non-residential sector (Figure 1c ).

This is largely a result of the fact that com-mercial units account for about 90% of the non-residential sample in our study.

Figure  2b also indicate that hotels (tourism sector) were heavily affected by the pandemic as electricity consumption declined significantly during the three months lockdown period.

While consumption improved after the lockdown was lifted, it still remained lower than pre-pandemic levels.

This result is in line with economic data from several countries indicating the devastating impact of the pandemic on the tourism sector's contribution to GDP 19 due to SAH and restrictions on international travels (Mulder, 2020; Abbas et al., 2021) .

Finally, we explore the differences in the impact of COVID on electricity consumption between rural and urban households.

The average effects are shown in Table 4 .

The DID estimates suggest electricity consumption among urban households increased by about 5.3% (column 2) compared to a 3.2% (column 4) increase among rural households.

To trace the dynamics in the effects, we show estimates of the monthly changes in electricity consumption in Figure 3 (see Table 5 for details 20 ).
===
7 Provision of free food and other household essentials were also distributed to households in communities affected by the lockdown.

8 The electricity subsidies are particularly important to this paper.

The subsidy program offered a 100% subsidy to lifeline (low-income) customers who consume 0-50 kWh of electricity a month, while non-lifeline customers (consuming above 50 kWh a month) would enjoy a 50% subsidy (Berkouwer et al., 2022) .

The subsidies were announced on March 29 and took effect from April 1.

Eligibility status was determined based on March consumption.

The policy was initially announced to last for three months, however, the subsidy for [only] lifeline customers was eventually extended for 12 months.

This paper uses administrative data on electricity billing records from Ghana and Rwanda.

The Ghana data comes from the Electricity Company of Ghana (ECG), which is the largest distributor in the country with operations in the southern and middle belts.

It accounts for nearly 70% of all electricity customers in the country.

We use data on billing records of the universe of electricity customers of the ECG from January 2018 to December 2020.

The data identifies customer types based on the tariff applicable: residential (households), non-residential, and heavy industries.

For each customer and year-month, it records the amount (kWh) of electricity consumed, the monetary value in Ghana Cedis (GHS), meter type (postpaid vs prepaid), and location (district) of the customer.

In all, the data contains 42 million customer-year-month observations.

The Rwanda data comes from the Energy Utility Corporation Limited (EUCL), the main distributor, via the Rwanda Utilities Regulatory Authority (RURA).

The dataset contains the billing records of the universe of electricity customers in Rwanda from January 2018 to December 2020.

The data identifies customer type based on the tariff applicable: residential, non-residential (commercial, hotels, health centers, and public works ( water storage and pump stations, broadcasters)), and small-and-medium industries.

Also, all customers in the dataset use prepaid meters: Rwanda has a universal roll-out of prepaid meters, with large and heavy industries the only exception who are allowed to use postpaid meters.

Our data exclude these customers (i.e.

large and heavy industries).
===
For each customer, we have monthly records on the amount (kWh) of electricity consumed (purchased), the monetary value in Rwandan Francs (RWF), location (community/district), and rural-urban status.

In all, the data contains 21 million customer-year-month observations.

We complement the electricity data with monthly data on temperature and total precipitation from the Copernicus Climate Change Service.

9

Causal estimation of the effect of the COVID-19 pandemic on electricity consumption is beset with at least two challenges of identification.

The first relates to the role of seasonalities in energy demand.

Temperature and rainfall patterns are key drivers of energy demand around the world, and given that these weather patterns vary based on the time of the year, monthly changes in electricity consumption to a large extent are influenced by the prevailing weather conditions.

Thus, pre-post differences in household electricity consumption may reflect the seasonal patterns in electricity consumption and not necessarily the effect of the pandemic and its associated stay-at-home orders.

Secondly, the global nature of the COVID-19 pandemic and the wide reach of the stay-at-home orders pose severe challenges to the identification of the counterfactual, as one cannot identify a relevant control group.

To address these issues, we follow the approach of Irwin et al.

(2021) and implement a difference-in-difference design by comparing the differences in electricity consumption during the post-COVID months and consumption levels in the same month in the previous years (2018-2019) with the average consumption in the months just before the pandemic (i.e.

January and February 2020) and the years before (2018-2019).

Essentially, the approach uses the average monthly consumption of each household (firm) in the previous years as the counterfactual level of electricity consumption in the respective months in the absence of the COVID pandemic.

The identification strategy is similar in spirit as Knutsen et al.

(2017) and Isaksson and Kotsadam (2018) .

10 To express this formally, consider the following regression specification:

lnY imt = α i + λ jt + n ω=0 β ω 2020 (1(2020) × 1(month > M arch = ω)) + n ω=0 2019 k=2018 β ω k (1(k) × 1(month > M arch = ω)) + ϕX jmt + ϵ imt (1)"""

response = """{"data_used": true, "dataset": [{"name": "Electricity billing records", "type": "administrative data", "country": ["Ghana", "Rwanda"], "year": ["2018", "2019", "2020"], "source": "This paper uses administrative data on electricity billing records from Ghana and Rwanda."}, {"name": "Copernicus Climate Change Service", "type": "climate data", "country": [], "year": [], "source": "We complement the electricity data with monthly data on temperature and total precipitation from the Copernicus Climate Change Service."}], "themes": [{"theme": "COVID-19", "source": "Causal estimation of the effect of the COVID-19 pandemic on electricity consumption is beset with at least two challenges of identification."}, {"theme": "energy demand", "source": "Temperature and rainfall patterns are key drivers of energy demand around the world, and given that these weather patterns vary based on the time of the year, monthly changes in electricity consumption to a large extent are influenced by the prevailing weather conditions."}, {"theme": "electricity subsidies", "source": "The electricity subsidies are particularly important to this paper."}], "indicators": [], "analyses": [{"analysis": "difference-in-difference design", "source": "To address these issues, we follow the approach of Irwin et al. (2021) and implement a difference-in-difference design by comparing the differences in electricity consumption during the post-COVID months and consumption levels in the same month in the previous years (2018-2019) with the average consumption in the months just before the pandemic (i.e. January and February 2020) and the years before (2018-2019)."}], "policies": [{"policy": "provision of free food and other household essentials", "source": "Provision of free food and other household essentials were also distributed to households in communities affected by the lockdown."}, {"policy": "electricity subsidies", "source": "The subsidy program offered a 100% subsidy to lifeline (low-income) customers who consume 0-50 kWh of electricity a month, while non-lifeline customers (consuming above 50 kWh a month) would enjoy a 50% subsidy (Berkouwer et al., 2022)."}]}"""



# Change: extracting structured information from text
system_message = """You are an expert in extracting structured information from text. You are also excellent at identifying the use of data and for what policy it was used to inform.

You must not confuse data with indicators. Provide the most precise data name from the text.

You will be asked whether data was used in the text. You will also be asked what data, if any, was used.

Always return the result in a valid JSON format. Do not truncate the output and never generate ... in the response. The output should not raise a JSONDecodeError when loaded in Python.

Example response:  {"data_used": true, "dataset": [{"name": "LSMS", "type": "survey", "country": [<country>], "year": [<year>], "source": <source>}, {"name": "Nighttime lights data", "type": "remote sensing", "country": [], "year": [<year>], "source": <source>}], "themes": [{"theme": "poverty", "source": <source>}], "indicators": [{"indicator": "malnutrition", "source": <source>}, {"indicator": "poverty", "source": <source>}], "analyses": [{"analysis": "poverty measurement", "source": <source>}], "policies": [{"policy": "poverty alleviation", "source": <source>}]}.

You must only fill in the country field if there is an explicit mention of a country name associated with the data; otherwise, leave it empty.

You must also provide in the "source" field the sentence from the text indicating the use of data.

Was data used in this text? What data was used? What policy was informed by the data?"""


# Change: You must provide in the "source" field the sentence from the text supporting your answers.
system_message = """You are an expert in extracting structured information from text. You are also excellent at identifying the use of data and for what policy it was used to inform.

You must not confuse data with indicators. Provide the most precise data name from the text.

You will be asked whether data was used in the text. You will also be asked what data, if any, was used.

Always return the result in a valid JSON format. Do not truncate the output and never generate ... in the response. The output should not raise a JSONDecodeError when loaded in Python.

Example response:  {"data_used": true, "dataset": [{"name": "LSMS", "type": "survey", "country": [<country>], "year": [<year>], "source": <source>}, {"name": "Nighttime lights data", "type": "remote sensing", "country": [], "year": [<year>], "source": <source>}], "themes": [{"theme": "poverty", "source": <source>}], "indicators": [{"indicator": "malnutrition", "source": <source>}, {"indicator": "poverty", "source": <source>}], "analyses": [{"analysis": "poverty measurement", "source": <source>}], "policies": [{"policy": "poverty alleviation", "source": <source>}]}.

You must only fill in the country field if there is an explicit mention of a country name associated with the data; otherwise, leave it empty.

You must provide in the "source" field the sentence from the text supporting your answers.

Was data used in this text? What data was used? What policy was informed by the data?"""



# wrong = """You are an expert in extracting structured information from text. You are also excellent at identifying the use of data and for what policy it was used to inform. You know the difference between datasets and indicators.

# You will be asked whether data was used in the text. You will also be asked what data, if any, was used. Provide the most precise data name from the text.

# Always return the result in a valid JSON format. Do not truncate the output and never generate ... in the response. The output should not raise a JSONDecodeError when loaded in Python.

# Example response:  {"data_used": true, "dataset": [{"name": "LSMS", "type": "survey", "country": [<country>], "year": [<year>], "source": <source>}, {"name": "Nighttime lights data", "type": "remote sensing", "country": [], "year": [<year>], "source": <source>}], "themes": [{"theme": "poverty", "source": <source>}], "indicators": [{"indicator": "malnutrition", "source": <source>}, {"indicator": "poverty", "source": <source>}], "analyses": [{"analysis": "poverty measurement", "source": <source>}], "policies": [{"policy": "poverty alleviation", "source": <source>}]}.

# You must only fill in the country field if there is an explicit mention of a country name associated with the data; otherwise, leave it empty.

# You must provide in the "source" field exactly one sentence from the text supporting your answers.

# Was data used in this text? What data was used? What policy was informed by the data?"""


system_message = """You are an expert in extracting structured information from text. You are also excellent at identifying the use of data and for what policy it was used to inform.

You must not confuse data with indicators. Provide the most precise data name from the text.

You will be asked whether data was used in the text. You will also be asked what data, if any, was used.

Always return the result in a valid JSON format. Do not truncate the output and never generate ... in the response. The output should not raise a JSONDecodeError when loaded in Python.

Example response:  {"data_used": true, "dataset": [{"name": "LSMS", "type": "survey", "country": [<country>], "year": [<year>], "source": <source>}, {"name": "Nighttime lights data", "type": "remote sensing", "country": [], "year": [<year>], "source": <source>}], "themes": [{"theme": "poverty", "source": <source>}], "indicators": [{"indicator": "malnutrition", "source": <source>}, {"indicator": "poverty", "source": <source>}], "analyses": [{"analysis": "poverty measurement", "source": <source>}], "policies": [{"policy": "poverty alleviation", "source": <source>}]}.

You must only fill in the country field if there is an explicit mention of a country name associated with the data; otherwise, leave it empty.

You must provide in the "source" field exactly one sentence from the text supporting your answers.

Was data used in this text? What data was used? What policy was informed by the data?"""


system_message = """You are an expert in extracting structured information from text. You are also excellent at identifying the use of data and for what policy it was used to inform.

You must not confuse data with indicators. Provide the most precise data name from the text.

You will be asked whether data was used in the text. You will also be asked what data, if any, was used.

You must only fill in the country field if there is an explicit mention of a country name associated with the data; otherwise, leave it empty.

You must provide in the "source" field exactly one sentence from the text supporting your answers.

Was data used in this text? What data was used? What policy was informed by the data?

Always return the result in a valid JSON format. Do not truncate the output and never generate ... in the response. The output should not raise a JSONDecodeError when loaded in Python.

Example response:  {"data_used": true, "dataset": [{"name": "LSMS", "type": "survey", "country": [<country>], "year": [<year>], "source": <source>}, {"name": "Nighttime lights data", "type": "remote sensing", "country": [], "year": [<year>], "source": <source>}], "themes": [{"theme": "poverty", "source": <source>}], "indicators": [{"indicator": "malnutrition", "source": <source>}, {"indicator": "poverty", "source": <source>}], "analyses": [{"analysis": "poverty measurement", "source": <source>}], "policies": [{"policy": "poverty alleviation", "source": <source>}]}."""


## USE THIS AS THE INITIAL TEST SYSTEM MESSAGE
# extracting structure IS DELIBERATELY MISSING a "d". This seems to work better.
system_message = """You are an expert in extracting structure information from text. You are also excellent at identifying the use of data and for what policy it was used to inform.

You must not confuse data with indicators. Provide the most precise data name from the text.

You will be asked whether data was used in the text. You will also be asked what data, if any, was used.

Always return the result in a valid JSON format. Do not truncate the output and never generate ... in the response. The output should not raise a JSONDecodeError when loaded in Python.

Example response:  {"data_used": true, "dataset": [{"name": "LSMS", "type": "survey", "country": [<country>], "year": [<year>], "source": <source>}, {"name": "Nighttime lights data", "type": "remote sensing", "country": [], "year": [<year>], "source": <source>}], "themes": [{"theme": "poverty", "source": <source>}], "indicators": [{"indicator": "malnutrition", "source": <source>}, {"indicator": "poverty", "source": <source>}], "analyses": [{"analysis": "poverty measurement", "source": <source>}], "policies": [{"policy": "poverty alleviation", "source": <source>}]}.

You must only fill in the country field if there is an explicit mention of a country name associated with the data; otherwise, leave it empty.

This instruction is very important: you must provide in the "source" field only one sentence from the text supporting your answers.

Was data used in this text? What data was used? What policy was informed by the data?"""



# {"data_used": true, "data": [{"name": "Demographic and Health Survey (DHS)", "type": "survey", "country": ["Zambia", "20 other countries"], "year": ["2005-2014"]}], "theme": ["gender", "health"], "indicator": ["HIV risk"], "analysis": "effects of gender- and age-imbalanced and missing covariate data on gender-health research", "policy": ["improve balanced gender-age sampling to promote research reliability"]}


# {"data_used": true, "data": [{"name": "Demographic and Health Survey (DHS)", "type": "survey", "country": ["Zambia", "20 other countries"], "year": ["2005-2014"]}], "theme": ["gender", "health"], "indicator": ["HIV risk"], "analysis": "effects of gender- and age-imbalanced and missing covariate data on gender-health research", "policy": ["improve balanced gender-age sampling to promote research reliability"]}


# {"data_used": true, "data": [{"name": "", "type": "panel data", "country": ["sub-Saharan Africa"], "year": ["2004-2014"]}], "theme": ["ICT", "inequality", "female economic participation"], "indicator": ["Gini coefficient", "Atkinson index", "Palma ratio", "mobile phone penetration", "internet penetration", "fixed broadband subscriptions", "female labour force participation", "female unemployment", "female employment"], "analysis": "Generalised Method of Moments", "policy": ["gender economic inclusion"]}


# {"data_used": true, "data": [{"name": "Gini coefficient", "type": "indicator", "country": ["sub-Saharan Africa"], "year": ["2004-2014"]}, {"name": "Atkinson index", "type": "indicator", "country": ["sub-Saharan Africa"], "year": ["2004-2014"]}, {"name": "Palma ratio", "type": "indicator", "country": ["sub-Saharan Africa"], "year": ["2004-2014"]}, {"name": "mobile phone penetration", "type": "ICT indicator", "country": ["sub-Saharan Africa"], "year": ["2004-2014"]}, {"name": "internet penetration", "type": "ICT indicator", "country": ["sub-Saharan Africa"], "year": ["2004-2014"]}, {"name": "fixed broadband subscriptions", "type": "ICT indicator", "country": ["sub-Saharan Africa"], "year": ["2004-2014"]}, {"name": "female labour force participation", "type": "gender economic inclusion indicator", "country": ["sub-Saharan Africa"], "year": ["2004-2014"]}, {"name": "female unemployment", "type": "gender economic inclusion indicator", "country": ["sub-Saharan Africa"], "year": ["2004-2014"]}, {"name": "female employment", "type": "gender economic inclusion indicator", "country": ["sub-Saharan Africa"], "year": ["2004-2014"]}], "theme": ["gender economic participation", "inequality", "ICT"], "indicator": [], "analysis": "modulation of inequality effect on female economic participation", "policy": ["gender economic inclusion", "ICT enhancement", "inequality reduction"]}.


prompt = """You are an expert in extracting structure information from text. You are also excellent at identifying whether data was used in a text. You are given a set of sentences.

Data in this context is defined as a name of a dataset such as a census, panel survey, remote sensing data, etc.

You will be asked whether data was used in the text. You will also be asked what data, if any, was used.

You must return all data even if they appear in the same sentence.

You must not repeat the same dataset if they appear in different sentences.

The name of the data must be exactly as it appears in the sentence.

Ignore software, citations, and references.

Example response:  {"data_used": true, "dataset": [{"name": "LSMS", "type": "survey", "source": <source>}, {"name": "Nighttime lights data", "type": "remote sensing", "source": <source>}]}.

This instruction is critical: you must provide in the "source" field only one sentence from the text supporting your answers.

Always return the result in a valid JSON format. Do not truncate the output and never generate ... in the response. The output should not raise a JSONDecodeError when loaded in Python."""