
# Coleridge experiment

```Python
passage_embeddings = HuggingFaceInstructEmbeddings(
    embed_instruction="Represent the passage for retrieval optimized for answering questions on the mention of data; Input: ",
    query_instruction="Represent the question for retrieving the most relevant passage mentioning data; Input: ",
)
id_num = 0
loader = sj.StructuredJSONLoader(coleridge_dir / f"{ids[id_num]}.json", jq_schema=".[].text")
docs = loader.load()
documents = du.aggregate_documents(docs)
sdocs = du.split_documents(documents)

# index = dvectorstore.VectorstoreIndexCreator(
#     embedding=passage_embeddings,
# ).from_documents(sdocs)

index = dvectorstore.VectorstoreIndexCreator(
    vectorstore_cls=Qdrant,
    embedding=passage_embeddings,
    vectorstore_kwargs=dict(
        location=":memory:",
        collection_name="coleridge",
    ),
).from_documents(sdocs)

query = "Was data or dataset used in this sentence?"

shortlist = index.vectorstore.similarity_search_with_score(query, k=20)
print("\n\n".join([o[0].page_content for o in shortlist]))
print(len(du._tokenizer.encode("\n\n".join([o[0].page_content for o in shortlist]))))
print(ids[id_num])

system_prompt = """
You are an expert in extracting structure information from text. You are also excellent at identifying whether data was used in a text. You are given a set of sentences.

You will be asked whether data was used in the text. You will also be asked what data, if any, was used.

Example response:  {"data_used": true, "dataset": [{"name": "LSMS", "type": "survey", "source": <source>}, {"name": "Nighttime lights data", "type": "remote sensing", "source": <source>}]}.

This instruction is critical: you must provide in the "source" field only one sentence from the text supporting your answers.

Always return the result in a valid JSON format. Do not truncate the output and never generate ... in the response. The output should not raise a JSONDecodeError when loaded in Python."""


system_prompt2 = """You are an expert in extracting structure information from text. You are also excellent at identifying whether data was used in a text. You are given a set of sentences.

You will be asked whether data was used in the text. You will also be asked what data, if any, was used. You must return all data even if they appear in the same sentence.

Example response:  {"data_used": true, "dataset": [{"name": "LSMS", "type": "survey", "source": <source>}, {"name": "Nighttime lights data", "type": "remote sensing", "source": <source>}]}.

This instruction is critical: you must provide in the "source" field only one sentence from the text supporting your answers.

Always return the result in a valid JSON format. Do not truncate the output and never generate ... in the response. The output should not raise a JSONDecodeError when loaded in Python."""

system_prompt3 = """You are an expert in extracting structure information from text. You are also excellent at identifying whether data was used in a text. You are given a set of sentences.

You will be asked whether data was used in the text. You will also be asked what data, if any, was used. You must return all data even if they appear in the same sentence.

Do not repeat the same dataset. Ignore software.

Example response:  {"data_used": true, "dataset": [{"name": "LSMS", "type": "survey", "source": <source>}, {"name": "Nighttime lights data", "type": "remote sensing", "source": <source>}]}.

This instruction is critical: you must provide in the "source" field only one sentence from the text supporting your answers.

Always return the result in a valid JSON format. Do not truncate the output and never generate ... in the response. The output should not raise a JSONDecodeError when loaded in Python."""

system_prompt4 = """You are an expert in extracting structure information from text. You are also excellent at identifying whether data was used in a text. You are given a set of sentences.

You will be asked whether data was used in the text. You will also be asked what data, if any, was used. You must return all data even if they appear in the same sentence.

Data in this context is defined as being the source of analysis and must not be an abstract concept.

Do not repeat the same dataset. Ignore software.

Example response:  {"data_used": true, "dataset": [{"name": "LSMS", "type": "survey", "source": <source>}, {"name": "Nighttime lights data", "type": "remote sensing", "source": <source>}]}.

This instruction is critical: you must provide in the "source" field only one sentence from the text supporting your answers.

Always return the result in a valid JSON format. Do not truncate the output and never generate ... in the response. The output should not raise a JSONDecodeError when loaded in Python."""


system_prompt5 = """You are an expert in extracting structure information from text. You are also excellent at identifying whether data was used in a text. You are given a set of sentences.

You will be asked whether data was used in the text. You will also be asked what data, if any, was used. You must return all data even if they appear in the same sentence.

Data in this context is defined as being the source of analysis and must not be an abstract concept. The name of the data must be exactly as it appears in the sentence.

Do not repeat the same dataset. Ignore software.

Example response:  {"data_used": true, "dataset": [{"name": "LSMS", "type": "survey", "source": <source>}, {"name": "Nighttime lights data", "type": "remote sensing", "source": <source>}]}.

This instruction is critical: you must provide in the "source" field only one sentence from the text supporting your answers.

Always return the result in a valid JSON format. Do not truncate the output and never generate ... in the response. The output should not raise a JSONDecodeError when loaded in Python."""


system_prompt6 = """You are an expert in extracting structure information from text. You are also excellent at identifying whether data was used in a text. You are given a set of sentences.

Data in this context is defined as a name of a dataset such as a census, panel survey, remote sensing data, etc.

You will be asked whether data was used in the text. You will also be asked what data, if any, was used.

You must return all data even if they appear in the same sentence.

You must not repeat the same dataset if they appear in different sentences.

The name of the data must be exactly as it appears in the sentence.

Ignore software, citations, and references.

Example response:  {"data_used": true, "dataset": [{"name": "LSMS", "type": "survey", "source": <source>}, {"name": "Nighttime lights data", "type": "remote sensing", "source": <source>}]}.

This instruction is critical: you must provide in the "source" field only one sentence from the text supporting your answers.

Always return the result in a valid JSON format. Do not truncate the output and never generate ... in the response. The output should not raise a JSONDecodeError when loaded in Python."""

```

# Idea 2

Generate a list of sentences that are positive examples of data usage and a list of sentences that are negative examples of data usage. Then, train a classifier to predict whether a sentence is a positive or negative example of data usage.

Below is an example list of sentences generate by ChatGPT:

Promp: """Forget all previous instructions. Can you generate a list of sentences indicating the use of a dataset? Try to use specific datasets in the sentence like LSMS, IBTrACS, High School Longitudinal Study, DHS, etc.

For example: "To this end, we used multi-modal imaging and neuropsychological battery data available in the Alzheimer's Disease Neuroimaging Initiative (ADNI) to investigate the relationship between cross-sectional measures of tau, cortical thickness, and different aspects of cognition."

Return the response in JSON."""

```JSON
{
    "sentences": [
        {
            "text": "Our analysis of poverty and health outcomes in developing countries was based on data from the Living Standards Measurement Study (LSMS).",
            "dataset": "LSMS"
        },
        {
            "text": "We used the International Best Track Archive for Climate Stewardship (IBTrACS) dataset to investigate trends in tropical cyclone activity.",
            "dataset": "IBTrACS"
        },
        {
            "text": "The findings of our study on high school dropout rates were based on data from the High School Longitudinal Study (HSLS).",
            "dataset": "HSLS"
        },
        {
            "text": "To examine the relationship between household characteristics and energy use, we analyzed data from the Demographic and Health Surveys (DHS) program.",
            "dataset": "DHS"
        },
        {
            "text": "Our research on the effects of air pollution on respiratory health was based on data from the National Health and Nutrition Examination Survey (NHANES).",
            "dataset": "NHANES"
        },
        {
            "text": "We used the World Values Survey dataset to investigate trends in political attitudes and values across different countries.",
            "dataset": "WVS"
        },
        {
            "text": "The results of our study on the relationship between parenting styles and child development were based on data from the Panel Study of Income Dynamics (PSID).",
            "dataset": "PSID"
        },
        {
            "text": "We analyzed data from the National Longitudinal Study of Adolescent Health (Add Health) to examine the social determinants of health among young adults.",
            "dataset": "Add Health"
        },
        {
            "text": "To investigate the effectiveness of interventions to reduce obesity, we used data from the Childhood Obesity Prevention and Treatment (COPTR) research consortium.",
            "dataset": "COPTR"
        },
        {
            "text": "Our analysis of the relationship between income and educational attainment was based on data from the US Census Bureau's Current Population Survey (CPS).",
            "dataset": "CPS"
        }
    ]
}
```
```Python
ref_vecs = passage_embeddings.embed_documents([r["text"] for r in reference_sentences["sentences"]])
e = index.vectorstore.client.search("coleridge", np.array(ref_vecs).mean(axis=0).tolist())
print("\n\n".join([o.payload["page_content"] for o in e]))
```

```Python
def get_sample_sentences(id_num, passage_embeddings, query=None, k=20):
    loader = sj.StructuredJSONLoader(coleridge_dir / f"{ids[id_num]}.json", jq_schema=".[].text")
    docs = loader.load()
    documents = du.aggregate_documents(docs)
    sdocs = du.split_documents(documents)
    # sdocs = documents

    # index = dvectorstore.VectorstoreIndexCreator(
    #     embedding=passage_embeddings,
    # ).from_documents(sdocs)

    index = dvectorstore.VectorstoreIndexCreator(
        vectorstore_cls=Qdrant,
        embedding=passage_embeddings,
        vectorstore_kwargs=dict(
            location=":memory:",
            collection_name="coleridge",
        ),
    ).from_documents(sdocs)

    # query = "Was data or dataset used in this sentence?"

    if query is None:
        ref_vecs = passage_embeddings.embed_documents([r["text"] for r in reference_sentences["sentences"]])
        e = index.vectorstore.client.search("coleridge", np.array(ref_vecs).mean(axis=0).tolist(), limit=k)
        content = "\n\n".join([o.payload["page_content"] for o in e])
    else:
        shortlist = index.vectorstore.similarity_search_with_score(query, k=k)
        content = "\n\n".join([o[0].page_content for o in shortlist])
    print(content)
    print(len(du._tokenizer.encode(content)))
    print(ids[id_num])

    index.vectorstore.client.delete_collection("coleridge")



shortlist = index.vectorstore.similarity_search_with_score(query, k=20)
content = "\n\n".join([o[0].page_content for o in shortlist])
print(content)
print(len(du._tokenizer.encode(content)))
print(ids[id_num])

```

```Python
import qdrant_client

sentence_embeddings = HuggingFaceInstructEmbeddings(
    embed_instruction="Represent the sentence for retrieval optimized for answering questions on the mention of data; Input: ",
    query_instruction="Represent the question for retrieving the most relevant sentence mentioning data; Input: ",
)

client = qdrant_client.QdrantClient(
    path="/tmp/local_qdrant", prefer_grpc=True
)

qdrant = Qdrant(
    client=client, collection_name="coleridge",
    embedding_function=sentence_embeddings.embed_query
)

```