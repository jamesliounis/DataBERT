# IPython log file

from data_use.document_loaders import pdf, structured_json as sj
import json
import data_use.text_splitter as ts
du = ts.DataUseTextSplitter(tokenizer=enc, separator="\n\n", chunk_size=512)
import tiktoken
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
du = ts.DataUseTextSplitter(tokenizer=enc, separator="\n\n", chunk_size=512)
loader = sj.StructuredJSONLoader("data/coleridgeinitiative/coleridgeinitiative-show-us-the-data (1)/train/0a2c7004-f763-4846-b95f-1fdf537f8a04.json", jq_schema=".[].key")
docs = loader.load()
import importlib
importlib.reload(sj)
loader = sj.StructuredJSONLoader("data/coleridgeinitiative/coleridgeinitiative-show-us-the-data (1)/train/0a2c7004-f763-4846-b95f-1fdf537f8a04.json", jq_schema=".[].key")
docs = loader.load()
o = json.loads(loader.file_path.read_text())
len(o)
o[0]
loader.jq_schema
w = loader.jq_schema.input(o)
next(w)
for i in w:
    print(i)
    break
    
loader = sj.StructuredJSONLoader("data/coleridgeinitiative/coleridgeinitiative-show-us-the-data (1)/train/0a2c7004-f763-4846-b95f-1fdf537f8a04.json", jq_schema=".[].text")
docs = loader.load()
len(docs)
documents = du.aggregate_documents(docs)
importlib.reload(sj)
# du = ts.DataUseTextSplitter(tokenizer=enc, separator="\n\n", chunk_size=512)
loader = sj.StructuredJSONLoader("data/coleridgeinitiative/coleridgeinitiative-show-us-the-data (1)/train/0a2c7004-f763-4846-b95f-1fdf537f8a04.json", jq_schema=".[].text")
docs = loader.load()
len(docs)
documents = du.aggregate_documents(docs)
len(documents)
from data_use.indexes import vectorstore as dvectorstore
from langchain.embeddings import HuggingFaceInstructEmbeddings
passage_embeddings = HuggingFaceInstructEmbeddings(
    embed_instruction="Represent the passage for retrieval optimized for answering questions on the mention of data; Input: ",
    query_instruction="Represent the question for retrieving the most relevant passage mentioning data; Input: ",
)
index = dvectorstore.VectorstoreIndexCreator(
    embedding=passage_embeddings,
).from_documents(documents)
query = "Was data or dataset used? Was data or dataset collected?"
k = 3
shortlist = index.vectorstore.similarity_search(query, k=k)
print("\n=====\n".join([o.page_content for o in shortlist[:3]]))
coleridge_dir = Path("data/coleridgeinitiative/coleridgeinitiative-show-us-the-data (1)/train/")
from pathlib import Path
coleridge_dir = Path("data/coleridgeinitiative/coleridgeinitiative-show-us-the-data (1)/train/")
ids = [j.stem for j in coleridge_dir.glob("*.json")]
len(ids)
ids[:10]
loader = sj.StructuredJSONLoader(coleridge_dir / f"{ids[0]}.json", jq_schema=".[].text")
docs = loader.load()
documents = du.aggregate_documents(docs)
index = dvectorstore.VectorstoreIndexCreator(
    embedding=passage_embeddings,
).from_documents(documents)
shortlist = index.vectorstore.similarity_search(query, k=k)
pri
print("\n=====\n".join([o.page_content for o in shortlist[:3]]))
ids[0]
shortlist = index.vectorstore.similarity_search(query, k=4)
print("\n=====\n".join([o.page_content for o in shortlist]))
"imaging" in ("\n\n".join([o.page_content for o in shortlist]))
"Neuroimaging" in ("\n\n".join([o.page_content for o in shortlist]))
"neuroimaging" in ("\n\n".join([o.page_content for o in shortlist]))
"ADNI" in ("\n\n".join([o.page_content for o in shortlist]))
loader = sj.StructuredJSONLoader(coleridge_dir / f"{ids[1]}.json", jq_schema=".[].text")
docs = loader.load()
documents = du.aggregate_documents(docs)
index = dvectorstore.VectorstoreIndexCreator(
    embedding=passage_embeddings,
).from_documents(documents)
shortlist = index.vectorstore.similarity_search(query, k=4)
print("\n=====\n".join([o.page_content for o in shortlist]))
ids[1]
"continuous" in ("\n\n".join([o.page_content for o in shortlist]))
"Continuous" in ("\n\n".join([o.page_content for o in shortlist]))
"continuum" in ("\n\n".join([o.page_content for o in shortlist]))
"Continuum" in ("\n\n".join([o.page_content for o in shortlist]))
("\n\n".join([o.page_content for o in shortlist])).index("continuum")
("\n\n".join([o.page_content for o in shortlist]))[2500: 2700]
loader = sj.StructuredJSONLoader(coleridge_dir / f"{ids[2]}.json", jq_schema=".[].text")
docs = loader.load()
documents = du.aggregate_documents(docs)
index = dvectorstore.VectorstoreIndexCreator(
    embedding=passage_embeddings,
).from_documents(documents)
shortlist = index.vectorstore.similarity_search(query, k=4)
print("\n=====\n".join([o.page_content for o in shortlist]))
ids[2]
loader = sj.StructuredJSONLoader(coleridge_dir / f"{ids[3]}.json", jq_schema=".[].text")
docs = loader.load()
documents = du.aggregate_documents(docs)
index = dvectorstore.VectorstoreIndexCreator(
    embedding=passage_embeddings,
).from_documents(documents)
shortlist = index.vectorstore.similarity_search(query, k=4)
print("\n=====\n".join([o.page_content for o in shortlist]))
ids[3]
"Baltimore" in ("\n\n".join([o.page_content for o in shortlist]))
"baltimore" in ("\n\n".join([o.page_content for o in shortlist]))
m = [d.page_content for d in documents if ("baltimore" in d.page_content.lower())]
len(m)
m[0]
sdocs = du.split_documents(documents)
len(sdocs)
sdocs[0]
sdocs[1]
sdocs[2]
sdocs[3]
importlib.reload(ts)
du = ts.DataUseTextSplitter(tokenizer=enc, separator="\n\n", chunk_size=512)
sdocs = du.split_documents(documents)
len(sdocs)
sdocs[0]
sdocs[1]
sdocs[2]
sdocs[3]
sdocs[4]
sdocs[5]
sdocs[6]
sdocs[7]
index = dvectorstore.VectorstoreIndexCreator(
    embedding=passage_embeddings,
).from_documents(sdocs)
shortlist = index.vectorstore.similarity_search(query, k=20)
print("\n=====\n".join([o.page_content for o in shortlist]))
"baltimore" in ("\n\n".join([o.page_content for o in shortlist]))
m = [d.page_content for d in sdocs if ("baltimore" in d.page_content.lower())]
len(m)
m[0]
query = "Was data or dataset used in this sentence?"
shortlist = index.vectorstore.similarity_search(query, k=20)
print("\n=====\n".join([o.page_content for o in shortlist]))
passage_embeddings = HuggingFaceInstructEmbeddings(
    embed_instruction="Represent the passage for retrieval, Input: ",
    query_instruction="Represent the question for retrieving relevant passage, Input: ",
)
index = dvectorstore.VectorstoreIndexCreator(
    embedding=passage_embeddings,
).from_documents(sdocs)
shortlist = index.vectorstore.similarity_search(query, k=20)
print("\n=====\n".join([o.page_content for o in shortlist]))
print("\n=====\n".join([o.page_content for o in shortlist]))
"baltimore" in ("\n\n".join([o.page_content for o in shortlist]))
passage_embeddings = HuggingFaceInstructEmbeddings(
    embed_instruction="Represent the passage for retrieval optimized for answering questions on the mention of data; Input: ",
    query_instruction="Represent the question for retrieving the most relevant passage mentioning data; Input: ",
)
index = dvectorstore.VectorstoreIndexCreator(
    embedding=passage_embeddings,
).from_documents(sdocs)
shortlist = index.vectorstore.similarity_search(query, k=20)
"baltimore" in ("\n\n".join([o.page_content for o in shortlist]))
"baltimore" in ("\n\n".join([o.page_content.lower() for o in shortlist]))
print("\n\n".join([o.page_content for o in shortlist]))
len(du._tokenizer.encode("\n=====\n".join([o.page_content for o in shortlist])))
loader = sj.StructuredJSONLoader(coleridge_dir / f"{ids[0]}.json", jq_schema=".[].text")
docs = loader.load()
documents = du.aggregate_documents(docs)
sdocs = du.split_documents(documents)
index = dvectorstore.VectorstoreIndexCreator(
    embedding=passage_embeddings,
).from_documents(sdocs)
shortlist = index.vectorstore.similarity_search(query, k=20)
query
query = "Was data or dataset used in this sentence?"
print("\n\n".join([o.page_content for o in shortlist]))
"adni" in ("\n\n".join([o.page_content.lower() for o in shortlist]))
shortlist = index.vectorstore.similarity_search(query, k=50)
"adni" in ("\n\n".join([o.page_content.lower() for o in shortlist]))
shortlist = index.vectorstore.similarity_search(query, k=25)
"adni" in ("\n\n".join([o.page_content.lower() for o in shortlist]))
print("\n\n".join([o.page_content for o in shortlist]))
shortlist = index.vectorstore.similarity_search(query, k=20)
"adni" in ("\n\n".join([o.page_content.lower() for o in shortlist]))
print("\n\n".join([o.page_content for o in shortlist]))
shortlist = index.vectorstore.similarity_search(query, k=25)
"adni" in ("\n\n".join([o.page_content.lower() for o in shortlist]))
print("\n\n".join([o.page_content for o in shortlist]))
get_ipython().run_line_magic('pinfo', 'index.vectorstore.similarity_search')
shortlist = index.vectorstore.similarity_search_with_relevance_scores(query, k=25)
shortlist = index.vectorstore.similarity_search_with_relevance_scores(query)
shortlist = index.vectorstore.similarity_search_with_relevance_scores(query)
shortlist = index.similarity_search_with_relevance_scores(query)
shortlist = index.vectorstore.similarity_search_with_score(query, k=20)
shortlist[0]
shortlist[1]
shortlist[2]
shortlist[3]
shortlist[4]
shortlist[10]
print(shortlist)
for i in shortlist:
    print(i)
    
for i in shortlist:
    print(i)
    print()
    
shortlist = index.vectorstore.similarity_search_with_score(query, k=25)
for i in shortlist:
    print(i)
    print()
    
# dvectorstore.VectorstoreIndexCreator(vectorstore_cls=
from langchain.vectorstores import Qdrant
index = dvectorstore.VectorstoreIndexCreator(vectorstore_cls=Qdrant, embedding=passage_embeddings).from_documents(sdocs)
# index = dvectorstore.VectorstoreIndexCreator(vectorstore_cls=Qdrant, embedding=passage_embeddings).from_documents(sdocs)
index = dvectorstore.VectorstoreIndexCreator(
    vectorstore_cls=Qdrant,
    embedding=passage_embeddings,
    vectorstore_kwargs=dict(
        location=":memory:",
        collection_name="coleridge",
    ),
).from_documents(sdocs)
shortlist = index.vectorstore.similarity_search_with_score(query, k=25)
len(shortlist)
for i in shortlist:
    print(i)
    print()
    
shortlist = index.vectorstore.similarity_search_with_score(query, k=20)
for i in shortlist:
    print(i)
    print()
    
shortlist = index.vectorstore.similarity_search_with_score(query, k=10)
print("\n\n".join([o.page_content for o in shortlist]))
print("\n\n".join([o.page_content[0] for o in shortlist]))
print("\n\n".join([o[0].page_content for o in shortlist]))
index.vectorstore.delete_collection()
index.vectorstore.collection_name
index.vectorstore
query = "Was data or dataset used in this sentence?"
loader = sj.StructuredJSONLoader(coleridge_dir / f"{ids[1]}.json", jq_schema=".[].text")
ids[0]
loader = sj.StructuredJSONLoader(coleridge_dir / f"{ids[1]}.json", jq_schema=".[].text")
sho
loader = sj.StructuredJSONLoader(coleridge_dir / f"{ids[1]}.json", jq_schema=".[].text")
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
print("\n\n".join([o.page_content for o in shortlist]))
print("\n\n".join([o[0].page_content for o in shortlist]))
ids[1]
loader = sj.StructuredJSONLoader(coleridge_dir / f"{ids[2]}.json", jq_schema=".[].text")
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
print("\n\n".join([o.page_content for o in shortlist]))
print("\n\n".join([o[0].page_content for o in shortlist]))
ids[2]
loader = sj.StructuredJSONLoader(coleridge_dir / f"{ids[3]}.json", jq_schema=".[].text")
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
ids[3]
loader = sj.StructuredJSONLoader(coleridge_dir / f"{ids[4]}.json", jq_schema=".[].text")
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
ids[4]
id_num = 5
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
print(ids[id_num])
m = [d.page_content for d in sdocs if ("neuroimaging" in d.page_content.lower())]
len(d)
len(m)
m[0]
shortlist = index.vectorstore.similarity_search_with_score(query, k=50)
print("\n\n".join([o[0].page_content for o in shortlist]))
len(du._tokenizer.encode("\n=====\n".join([o.page_content for o in shortlist])))
len(du._tokenizer.encode("\n=====\n".join([o[0].page_content for o in shortlist])))
id_num = 6
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
print(ids[id_num])
m = [d.page_content for d in sdocs if ("industrial" in d.page_content.lower())]
len(m)
m[0]
m[1]
m = [d.page_content for d in sdocs if ("industrial research" in d.page_content.lower())]
len(m)
m[0]
shortlist = index.vectorstore.similarity_search_with_score(query, k=50)
print("\n\n".join([o[0].page_content for o in shortlist]))
print(len(du._tokenizer.encode("\n\n".join([o[0].page_content for o in shortlist]))))
print(ids[id_num])
id_num = 7
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
id_num = 8
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
shortlist = index.vectorstore.similarity_search_with_score(query, k=50)
print("\n\n".join([o[0].page_content for o in shortlist]))
print("\n\n".join([o[0].page_content for o in shortlist]))
print(len(du._tokenizer.encode("\n\n".join([o[0].page_content for o in shortlist]))))
print(ids[id_num])
id_num = 9
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
id_num = 10
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
len("2020) .")
shortlist = index.vectorstore.similarity_search_with_score(query, k=50)
print("\n\n".join([o[0].page_content for o in shortlist]))
print(len(du._tokenizer.encode("\n\n".join([o[0].page_content for o in shortlist]))))
print(ids[id_num])
importlib.reload(ts)
du = ts.DataUseTextSplitter(tokenizer=enc, separator="\n\n", chunk_size=512)
loader = sj.StructuredJSONLoader(coleridge_dir / f"{ids[4]}.json", jq_schema=".[].text")
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
id_num = 10
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
len("2019; Menachery et al.")
len("2016; Menachery et al.")
st = [s.page_content for s in sdocs if len(s.page_content) < 25]
len(st)
st
du = ts.DataUseTextSplitter(tokenizer=enc, separator="\n\n", chunk_size=512)
importlib.reload(ts)
du = ts.DataUseTextSplitter(tokenizer=enc, separator="\n\n", chunk_size=512)
id_num = 10
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
m = [d.page_content for d in sdocs if ("genome sequence" in d.page_content.lower())]
len(m)
m
id_num = 11
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
id_num = 12
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
m = [d.page_content for d in sdocs if ("beginning postsecondary" in d.page_content.lower())]
len(m)
m
id_num = 13
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
id_num = 14
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
m = [d.page_content for d in sdocs if ("observational" in d.page_content.lower())]
len(m)
m = [d.page_content for d in sdocs if ("observational study" in d.page_content.lower())]
len(m)
m
passage_embeddings = HuggingFaceInstructEmbeddings(
    embed_instruction="Represent the passage for retrieval optimized for answering questions on the mention of data; Input: ",
    query_instruction="Represent the question for retrieving the most relevant passage mentioning data; Input: ",
)
index.vectorstore.client.delete_collection("colridg")
index.vectorstore.client.delete_collection("colridge")
shortlist = index.vectorstore.similarity_search_with_score(query, k=20)
len(shortlist)
index.vectorstore.client.delete_collection("coleridge")
shortlist = index.vectorstore.similarity_search_with_score(query, k=20)
index.vectorstore.delete_collection()
ss = """The Critical Path Institute, whose director, Louis Kirby, attended the Phoenix meeting, is working to speed up the FDA's formal acceptance of biomarkers. AD scientists are already detecting a subtle opening on the part of FDA regulators, who are increasingly accepting the notion that the presence of AD pathology in a person's brain prior to symptoms represents a disease state. An accelerated approval pathway based in large part on biomarker evidence exists and could potentially serve to guide the API.\nAt the same time, the conventional model of drug testing is being called into question as anti-amyloid and other investigational therapies have so far performed below expectations. \"There is a growing concern that these may be too little too late to exert a profound clinical effect once patients have even mild symptoms. We need to move earlier,\" said Reiman, and this stance now draws widespread support. In Phoenix, one company scientist recounted how the trial results of anti-amyloid drugs played internally at his company: \"The weak trial results hit the hypothesis hard. In 2009, we were asked to present to senior management on the state of the amyloid hypothesis, and the outcome of that was that the best test of the hypothesis were prevention trials both in PS1 and in ApoE4 high-risk groups.\""""
from nltk.tokenize import sent_tokenize
sent_tokenize(ss)
sdocs[0]
passage_embeddings.dict()
Qdrant
Qdrant.__repr__
Qdrant.__repr__()
str(Qdrant)
index = dvectorstore.VectorstoreIndexCreator(
    vectorstore_cls=Qdrant,
    embedding=passage_embeddings,
    vectorstore_kwargs=dict(
        location=":memory:",
        collection_name="coleridge",
    ),
).from_documents(sdocs)
index.vectorstore.client
index.vectorstore
index.vectorstore.similarity_search_with_score
query = "Was data or dataset used in this sentence?"
import pexpect
query = "Was data or dataset mentioned in this sentence?"
shortlist = index.vectorstore.similarity_search_with_score(query, k=20)
print("\n\n".join([o[0].page_content for o in shortlist]))
print(len(du._tokenizer.encode("\n\n".join([o[0].page_content for o in shortlist]))))
print(ids[id_num])
query = "Was data or dataset used in this sentence?"
shortlist2 = index.vectorstore.similarity_search_with_score(query, k=20)
shortlist2 == shortlist
print("\n\n".join([o[0].page_content for o in shortlist2]))
print(len(du._tokenizer.encode("\n\n".join([o[0].page_content for o in shortlist2]))))
print(ids[id_num])
get_ipython().run_line_magic('logstart', '')
reference_sentences = {
    "datasets": [
        {
            "name": "Living Standards Measurement Study",
            "description": "Data on poverty and health outcomes in developing countries.",
            "abbreviation": "LSMS"
        },
        {
            "name": "International Best Track Archive for Climate Stewardship",
            "description": "Data on tropical cyclone activity.",
            "abbreviation": "IBTrACS"
        },
        {
            "name": "High School Longitudinal Study",
            "description": "Data on high school dropout rates.",
            "abbreviation": "HSLS"
        },
        {
            "name": "Demographic and Health Surveys",
            "description": "Data on household characteristics and energy use.",
            "abbreviation": "DHS"
        },
        {
            "name": "National Health and Nutrition Examination Survey",
            "description": "Data on the effects of air pollution on respiratory health.",
            "abbreviation": "NHANES"
        },
        {
            "name": "World Values Survey",
            "description": "Data on political attitudes and values across different countries.",
            "abbreviation": "WVS"
        },
        {
            "name": "Panel Study of Income Dynamics",
            "description": "Data on the relationship between parenting styles and child development.",
            "abbreviation": "PSID"
        },
        {
            "name": "National Longitudinal Study of Adolescent Health",
            "description": "Data on the social determinants of health among young adults.",
            "abbreviation": "Add Health"
        },
        {
            "name": "Childhood Obesity Prevention and Treatment",
            "description": "Data on interventions to reduce obesity.",
            "abbreviation": "COPTR"
        },
        {
            "name": "Current Population Survey",
            "description": "Data on the relationship between income and educational attainment.",
            "abbreviation": "CPS"
        }
    ]
}
reference_sentences = {
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
ref_vecs = passage_embeddings.embed_documents([r["text"] for r in reference_sentences])
ref_vecs = passage_embeddings.embed_documents([r["text"] for r in reference_sentences["sentences"]])
len(ref_vecs)
#ref_vecs[0]
import numpy as np
np.array(ref_vecs).shape
shortlist2 == shortlist
# shortlist2 = 
query
shortlist = index.vectorstore.similarity_search_by_vector(np.array(ref_vecs).mean(axis=0).tolist(), k=20)
index.vectorstore
e = index.vectorstore.client.search("coleridge", np.array(ref_vecs).mean(axis=0).tolist())
len(e)
e[0]
e[1]
e[2]
shortlist = index.vectorstore.similarity_search_by_vector(np.array(ref_vecs).mean(axis=0).tolist(), k=20)
shortlist = index.vectorstore.similarity_search_by_vector(np.array(ref_vecs).mean(axis=0).tolist(), k=20)
shortlist = index.vectorstore.similarity_search_with_score(query, k=20)
print("\n\n".join([o[0].page_content for o in shortlist]))
print(len(du._tokenizer.encode("\n\n".join([o[0].page_content for o in shortlist]))))
print(ids[id_num])
e
print("\n\n".join([o.payload["page_content"] for o in e]))
# shortlist = index.vectorstore.similarity_search_by_vector(np.array(ref_vecs).mean(axis=0).tolist(), k=20)
search_vec = np.array(ref_vecs).mean(axis=0).tolist()
index.vectorstore.client.delete_collection("coleridge")
id_num = 1
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

ref_vecs = passage_embeddings.embed_documents([r["text"] for r in reference_sentences["sentences"]])
e = index.vectorstore.client.search("coleridge", np.array(ref_vecs).mean(axis=0).tolist())
content = "\n\n".join([o.payload["page_content"] for o in e])
print(content)
print(len(du._tokenizer.encode(content)))
print(ids[id_num])
e = index.vectorstore.client.search("coleridge", np.array(ref_vecs).mean(axis=0).tolist(), topk=10)
e = index.vectorstore.client.search("coleridge", np.array(ref_vecs).mean(axis=0).tolist(), topk=20)
len(e)
get_ipython().run_line_magic('pinfo', 'index.vectorstore.client.search')
e = index.vectorstore.client.search("coleridge", np.array(ref_vecs).mean(axis=0).tolist(), limit=20)
len(e)
def get_sample_sentences(id_num, k=20):
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

    ref_vecs = passage_embeddings.embed_documents([r["text"] for r in reference_sentences["sentences"]])
    e = index.vectorstore.client.search("coleridge", np.array(ref_vecs).mean(axis=0).tolist(), limit=k)
    content = "\n\n".join([o.payload["page_content"] for o in e])
    print(content)
    print(len(du._tokenizer.encode(content)))
    print(ids[id_num])

    index.vectorstore.client.delete_collection("coleridge")
    
get_sample_sentences(id_num=2)
get_sample_sentences(id_num=3)
get_sample_sentences(id_num=4)
get_sample_sentences(id_num=5)
get_sample_sentences(id_num=6)
get_sample_sentences(id_num=7)
get_sample_sentences(id_num=8)
get_sample_sentences(id_num=9)
get_sample_sentences(id_num=10)
get_sample_sentences(id_num=11)
get_sample_sentences(id_num=12)
get_sample_sentences(id_num=13)
get_sample_sentences(id_num=14)
get_sample_sentences(id_num=15)
get_sample_sentences(id_num=16)
get_sample_sentences(id_num=17)
get_sample_sentences(id_num=18)
get_sample_sentences(id_num=19)
get_sample_sentences(id_num=20)
get_sample_sentences(id_num=21)
get_sample_sentences(id_num=22)
index = dvectorstore.VectorstoreIndexCreator(
    vectorstore_cls=Qdrant,
    embedding=passage_embeddings,
    vectorstore_kwargs=dict(
        location=":memory:",
        collection_name="coleridge",
    ),
).from_documents(sdocs)
index.vectorstore.client.delete_collection("coleridge")
shortlist = index.vectorstore.similarity_search_with_score(query, k=20)
index.vectorstore.from_documents(sdocs)
index.vectorstore.from_documents(sdocs, embedding=passage_embeddings)
index.vectorstore.client
index.vectorstore.client.create_collection("coleridge")
sentence_embedding
sentence_embeddings = HuggingFaceInstructEmbeddings(
    embed_instruction="Represent the sentence for retrieval optimized for answering questions on the mention of data; Input: ",
    query_instruction="Represent the question for retrieving the most relevant sentence mentioning data; Input: ",
)
import qdrant_client
client = qdrant_client.QdrantClient(
    path="/tmp/local_qdrant", prefer_grpc=True
)
qdrant = Qdrant(
    client=client, collection_name="coleridge", 
    embedding_function=sentence_embedding.embed_query
)
qdrant = Qdrant(
    client=client, collection_name="coleridge", 
    embedding_function=sentence_embeddings.embed_query
)
qdrant.add_documents(sdocs)
client.create_collection("coleridge", None)
client.create_collection("coleridge", {})
qdrant.add_documents(sdocs)
em = passage_embeddings.embed_documents(sdocs)
em = passage_embeddings.embed_documents([i.page_content for i in sdocs])
passage_embeddings.client
em = passage_embeddings.client.encode([[passage_embeddings.embed_instruction, i.page_content] for i in sdocs])
em = passage_embeddings.client.encode([[passage_embeddings.embed_instruction, i.page_content] for i in documents])
len(em)
def get_sample_sentences(id_num, k=20):
    loader = sj.StructuredJSONLoader(coleridge_dir / f"{ids[id_num]}.json", jq_schema=".[].text")
    docs = loader.load()
    documents = du.aggregate_documents(docs)
    # sdocs = du.split_documents(documents)
    sdocs = documents

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

    ref_vecs = passage_embeddings.embed_documents([r["text"] for r in reference_sentences["sentences"]])
    e = index.vectorstore.client.search("coleridge", np.array(ref_vecs).mean(axis=0).tolist(), limit=k)
    content = "\n\n".join([o.payload["page_content"] for o in e])
    print(content)
    print(len(du._tokenizer.encode(content)))
    print(ids[id_num])

    index.vectorstore.client.delete_collection("coleridge")
get_sample_sentences(id_num=22, k=3)
def get_sample_sentences(id_num, k=20):
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

    query = "Was data or dataset used in this sentence?"

    ref_vecs = passage_embeddings.embed_documents([r["text"] for r in reference_sentences["sentences"]])
    e = index.vectorstore.client.search("coleridge", np.array(ref_vecs).mean(axis=0).tolist(), limit=k)
    content = "\n\n".join([o.payload["page_content"] for o in e])
    print(content)
    print(len(du._tokenizer.encode(content)))
    print(ids[id_num])

    index.vectorstore.client.delete_collection("coleridge")
get_sample_sentences(id_num=22, k=3)
get_sample_sentences(id_num=22, k=20)
index.vectorstore.client.create_collection("coleridge")
passage_embeddings
ref_vecs = passage_embeddings.embed_documents([r["text"] for r in reference_sentences["sentences"]])
e = index.vectorstore.client.search("coleridge", np.array(ref_vecs).mean(axis=0).tolist())
print("\n\n".join([o.payload["page_content"] for o in e]))
ref_vecs = passage_embeddings.embed_documents([r["text"] for r in reference_sentences["sentences"]])
def get_sample_sentences(id_num, k=20):
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

    query = "Was data or dataset used in this sentence?"

    ref_vecs = passage_embeddings.embed_documents([r["text"] for r in reference_sentences["sentences"]])
    e = index.vectorstore.client.search("coleridge", np.array(ref_vecs).mean(axis=0).tolist(), limit=k)
    content = "\n\n".join([o.payload["page_content"] for o in e])
    print(content)
    print(len(du._tokenizer.encode(content)))
    print(ids[id_num])

    index.vectorstore.client.delete_collection("coleridge")
get_sample_sentences(id_num=22, k=20)
get_sample_sentences(id_num=20, k=20)
eee = """[{"sentence": "This study draws on data from the DHS conducted in Sub-Saharan African countries with at least two rounds of data.", "dataset": ["DHS"]}, {"sentence": "This paper uses administrative data on electricity billing records from Ghana and Rwanda.", "dataset": ["administrative data on electricity billing records"]}, {"sentence": "We complement the electricity data with monthly data on temperature and total precipitation from the Copernicus Climate Change Service.", "dataset": ["Copernicus Climate Change Service"], {"sentence": "We use two main sources of data, both of which are novel: a dataset on the universe of Indonesian exporters in the period 2014-18 and a time-varying dataset of NTMs applied on Indonesian imports.", "dataset": ["dataset on the universe of Indonesian exporters in the period 2014-18", "time-varying dataset of NTMs applied on Indonesian imports"]}, {"sentence": "To this end, we used multi-modal imaging and neuropsychological battery data available in the Alzheimer's Disease Neuroimaging Initiative (ADNI) to investigate the relationship between cross-sectional measures of tau, cortical thickness, and different aspects of cognition.", "dataset": ["Alzheimer's Disease Neuroimaging Initiative (ADNI)"]}]"""
json.loads(eee)
eee = """[{"sentence": "This study draws on data from the DHS conducted in Sub-Saharan African countries with at least two rounds of data.", "dataset": ["DHS"]}, {"sentence": "This paper uses administrative data on electricity billing records from Ghana and Rwanda.", "dataset": ["administrative data on electricity billing records"]}, {"sentence": "We complement the electricity data with monthly data on temperature and total precipitation from the Copernicus Climate Change Service.", "dataset": ["Copernicus Climate Change Service"]}, {"sentence": "We use two main sources of data, both of which are novel: a dataset on the universe of Indonesian exporters in the period 2014-18 and a time-varying dataset of NTMs applied on Indonesian imports.", "dataset": ["dataset on the universe of Indonesian exporters in the period 2014-18", "time-varying dataset of NTMs applied on Indonesian imports"]}, {"sentence": "To this end, we used multi-modal imaging and neuropsychological battery data available in the Alzheimer's Disease Neuroimaging Initiative (ADNI) to investigate the relationship between cross-sectional measures of tau, cortical thickness, and different aspects of cognition.", "dataset": ["Alzheimer's Disease Neuroimaging Initiative (ADNI)"]}]"""
json.loads(eee)
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")
def get_sample_sentences(id_num, passage_embeddings, k=20):
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

    query = "Was data or dataset used in this sentence?"

    ref_vecs = passage_embeddings.embed_documents([r["text"] for r in reference_sentences["sentences"]])
    e = index.vectorstore.client.search("coleridge", np.array(ref_vecs).mean(axis=0).tolist(), limit=k)
    content = "\n\n".join([o.payload["page_content"] for o in e])
    print(content)
    print(len(du._tokenizer.encode(content)))
    print(ids[id_num])

    index.vectorstore.client.delete_collection("coleridge")
    
get_sample_sentences(id_num=20, embeddings, k=20)
get_sample_sentences(id_num=20, passage_embeddings=embeddings, k=20)
get_sample_sentences(id_num=0, passage_embeddings=embeddings, k=20)
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
    
get_sample_sentences(id_num=0, passage_embeddings=embeddings, query="Was data used in the sentence?", k=20)
get_sample_sentences(id_num=20, passage_embeddings=embeddings, query="Was data used in the sentence?", k=20)
get_sample_sentences(id_num=1, passage_embeddings=embeddings, query="Was data used in the sentence?", k=20)
get_sample_sentences(id_num=1, passage_embeddings=embeddings, k=20)
get_sample_sentences(id_num=2, passage_embeddings=embeddings, k=20)
get_sample_sentences(id_num=2, passage_embeddings=embeddings, query="Was data used in the sentence?", k=20)
get_sample_sentences(id_num=2, passage_embeddings=embeddings, query="Was data used in the sentence?", k=5)
get_sample_sentences(id_num=2, passage_embeddings=embeddings, k=5)
get_sample_sentences(id_num=10, passage_embeddings=embeddings, k=5)
get_sample_sentences(id_num=15, passage_embeddings=embeddings, k=5)
get_sample_sentences(id_num=15, passage_embeddings=embeddings, query="Was data used in the sentence?", k=5)
id_num = 12
get_sample_sentences(id_num=id_num, passage_embeddings=embeddings, query="Was data used in the sentence?", k=5)
get_sample_sentences(id_num=id_num, passage_embeddings=embeddings, k=5)
get_sample_sentences(id_num=id_num, passage_embeddings=embeddings, k=20)
get_sample_sentences(id_num=id_num, passage_embeddings=embeddings, query="Was data used in the sentence?", k=20)
get_sample_sentences(id_num=id_num, passage_embeddings=passage_embeddings, k=20)
get_sample_sentences(id_num=id_num, passage_embeddings=passage_embeddings, query="Was data used in the sentence?", k=20)
id_num = 11
get_sample_sentences(id_num=id_num, passage_embeddings=embeddings, k=10)
get_sample_sentences(id_num=id_num, passage_embeddings=embeddings, query="Was data used in the sentence?", k=10)
get_sample_sentences(id_num=id_num, passage_embeddings=passage_embeddings, k=100)
get_sample_sentences(id_num=id_num, passage_embeddings=passage_embeddings, k=10)
get_sample_sentences(id_num=id_num, passage_embeddings=passage_embeddings, query="Was data used in the sentence?", k=10)
get_sample_sentences(id_num=0, passage_embeddings=passage_embeddings, k=10)
print("To predict sample attrition, we used an indicator for whether the mother was born in the village, the number of boys in the family, whether the father was living in the household in 2003, and whether the male interview was conducted in another language than Punjabi in 2003.\n\nTo predict inclusion in our regression analysis sample, we use the number of schools in the village in 2003.\n\nWe use the Stata package developed by De to carry out the sequential bivariate semi-nonparametric estimation for the sample attrition and selection.\n\nTo capture the skills of our diverse pool of respondents, we worked with an organization to design an adaptive tablet-based test.\n\nThe organization we partnered with developed 324 items ranging from early primary level to college level.\n\nThe test classified respondents into six levels that correspond to different grades.\n\nThe mapping between levels and grades is as follows: -Level 1: Nursery, Grades 1 to 3 (early primary) -Level 2: Grades 4 and 5 (late primary) -Level 3: Grades 6 to 8 (middle school) -Level 4: Grades 9 and 10 (high school) -Level 5: Grades 11 and 12 (intermediate) -Level 6: College\n\nThe items of the tests were designed to capture the following learning domains: (1) mastery over concepts and definitions (e.g., \"what is a pronoun?\n\n\"), (2) application mastery (e.g., \"add 2+2\"), and (3) evaluation mastery (e.g., \"two boys meet two girls, one boy leaves; how many children are left?\n\n\").\n\nAs we expected many respondents to have been out of school for a long time, items were designed to test general mastery (as opposed to specific terms or formulae).\n\nAll the test items were multiplechoice questions with four possible answer choices and one correct answer.\n\nThe logic of the test was as follows.\n\n-Everyone started at the same level -Level 2 for Urdu and Mathematics and Level 1 for English -and answered a batch of 6 questions.\n\n-If the respondent got 5-6 questions right, they moved to the next higher level (or, if at Level 6, remained at Level 6).\n\n-If the respondent got 3-4 questions right, they stayed at the same level.")
print("Of the 140 potential microdata sets explored (seven data categories in 20 countries), 78 had been collected and 22 were accessible.\n\nMoreover, about a third of these 22 datasets were not accessible through the website of the National Statistics Office but had to be found on international microdata repositories, such as the World Bank, International Household Survey Network (IHSN) microdata library, IPUMS, Eurostat, Demographic and Health Surveys (DHS), and Multiple Indicator Cluster Surveys (MICS).\n\nWhile the SCI, SPI, and ODIN indicators capture the quality of the data ecosystem to a large extent, they do not necessarily encompass the extent and availability of high-frequency data.\n\nTable 3 presents the availability of high-frequency data on GDP, industrial production, and unemployment for 19 economies in the MENA region.\n\nThe date of the most recent data available is also indicated.\n\nThis information is compiled from the websites of various statistical offices or data portal initiatives, central banks and ministries of planning, economy, or finance across the MENA region.\n\nThe findings are benchmarked against Mexico, which serves as a good comparator for the MENA region given its upper-middle-income economy status and a well-functioning data ecosystem.\n\nOf the 19 economies in the MENA region, 15 report quarterly data on GDP.\n\nSome economies lack information for the year 2020 entirely.\n\nEconomies in conflict such as Libya (2014) and Yemen (2017) have outdated data.\n\nOnly 10 of the 19 MENA economies have monthly or quarterly information on industrial production-for the remaining nine economies, information is not readily available.\n\nOnly eight economies report quarterly unemployment data, while none have monthly data.\n\nThe benchmark country, Mexico, reports unemployment data monthly.\n\nA caveat is that the table does not consider quality.\n\nFor example, definitions of unemployment may be inconsistent with international standards (Arezki et al., 2020) .\n\nOn top of the lack of published monthly unemployment data in the MENA region, there are also challenges regarding the definitions employed.\n\nFor examples, countries around the world usually follow the ILO's definitions of employment and unemployment, which are consistent with definitions adopted by other developed countries, such as the United States (see Table 4 ).")
get_sample_sentences(id_num=id_num, passage_embeddings=embeddings, query="In this study, data analyses were conducted on two sub-samples derived from the complete ADNI sample.", k=10)
get_sample_sentences(id_num=0, passage_embeddings=embeddings, query="In this study, data analyses were conducted on two sub-samples derived from the complete ADNI sample.", k=10)
get_sample_sentences(id_num=1, passage_embeddings=embeddings, query="In this study, data analyses were conducted on two sub-samples derived from the complete ADNI sample.", k=10)
get_sample_sentences(id_num=2, passage_embeddings=embeddings, query="In this study, data analyses were conducted on two sub-samples derived from the complete ADNI sample.", k=10)
get_sample_sentences(id_num=3, passage_embeddings=embeddings, query="In this study, data analyses were conducted on two sub-samples derived from the complete ADNI sample.", k=10)
get_sample_sentences(id_num=4, passage_embeddings=embeddings, query="In this study, data analyses were conducted on two sub-samples derived from the complete ADNI sample.", k=10)
get_sample_sentences(id_num=4, passage_embeddings=passage_embeddings, query="In this study, data analyses were conducted on two sub-samples derived from the complete ADNI sample.", k=10)
get_sample_sentences(id_num=2, passage_embeddings=passage_embeddings, query="In this study, data analyses were conducted on two sub-samples derived from the complete ADNI sample.", k=10)
get_sample_sentences(id_num=5, passage_embeddings=passage_embeddings, query="In this study, data analyses were conducted on two sub-samples derived from the complete ADNI sample.", k=10)
get_sample_sentences(id_num=5, passage_embeddings=passage_embeddings, k=10)
get_sample_sentences(id_num=6, passage_embeddings=passage_embeddings, query="In this study, data analyses were conducted on two sub-samples derived from the complete ADNI sample.", k=10)
get_sample_sentences(id_num=6, passage_embeddings=passage_embeddings, k=10)
get_sample_sentences(id_num=7, passage_embeddings=passage_embeddings, query="In this study, data analyses were conducted on two sub-samples derived from the complete ADNI sample.", k=10)
get_sample_sentences(id_num=7, passage_embeddings=passage_embeddings, k=10)
print("Of the 140 potential microdata sets explored (seven data categories in 20 countries), 78 had been collected and 22 were accessible.\n\nMoreover, about a third of these 22 datasets were not accessible through the website of the National Statistics Office but had to be found on international microdata repositories, such as the World Bank, International Household Survey Network (IHSN) microdata library, IPUMS, Eurostat, Demographic and Health Surveys (DHS), and Multiple Indicator Cluster Surveys (MICS).\n\nWhile the SCI, SPI, and ODIN indicators capture the quality of the data ecosystem to a large extent, they do not necessarily encompass the extent and availability of high-frequency data.\n\nTable 3 presents the availability of high-frequency data on GDP, industrial production, and unemployment for 19 economies in the MENA region.\n\nThe date of the most recent data available is also indicated.\n\nThis information is compiled from the websites of various statistical offices or data portal initiatives, central banks and ministries of planning, economy, or finance across the MENA region.\n\nThe findings are benchmarked against Mexico, which serves as a good comparator for the MENA region given its upper-middle-income economy status and a well-functioning data ecosystem.\n\nOf the 19 economies in the MENA region, 15 report quarterly data on GDP.\n\nSome economies lack information for the year 2020 entirely.\n\nEconomies in conflict such as Libya (2014) and Yemen (2017) have outdated data.\n\nOnly 10 of the 19 MENA economies have monthly or quarterly information on industrial production-for the remaining nine economies, information is not readily available.\n\nOnly eight economies report quarterly unemployment data, while none have monthly data.\n\nThe benchmark country, Mexico, reports unemployment data monthly.\n\nA caveat is that the table does not consider quality.\n\nFor example, definitions of unemployment may be inconsistent with international standards (Arezki et al., 2020) .\n\nOn top of the lack of published monthly unemployment data in the MENA region, there are also challenges regarding the definitions employed.\n\nFor examples, countries around the world usually follow the ILO's definitions of employment and unemployment, which are consistent with definitions adopted by other developed countries, such as the United States (see Table 4 ).")
se = "Of the 140 potential microdata sets explored (seven data categories in 20 countries), 78 had been collected and 22 were accessible.\n\nMoreover, about a third of these 22 datasets were not accessible through the website of the National Statistics Office but had to be found on international microdata repositories, such as the World Bank, International Household Survey Network (IHSN) microdata library, IPUMS, Eurostat, Demographic and Health Surveys (DHS), and Multiple Indicator Cluster Surveys (MICS).\n\nWhile the SCI, SPI, and ODIN indicators capture the quality of the data ecosystem to a large extent, they do not necessarily encompass the extent and availability of high-frequency data.\n\nTable 3 presents the availability of high-frequency data on GDP, industrial production, and unemployment for 19 economies in the MENA region.\n\nThe date of the most recent data available is also indicated.\n\nThis information is compiled from the websites of various statistical offices or data portal initiatives, central banks and ministries of planning, economy, or finance across the MENA region.\n\nThe findings are benchmarked against Mexico, which serves as a good comparator for the MENA region given its upper-middle-income economy status and a well-functioning data ecosystem.\n\nOf the 19 economies in the MENA region, 15 report quarterly data on GDP.\n\nSome economies lack information for the year 2020 entirely.\n\nEconomies in conflict such as Libya (2014) and Yemen (2017) have outdated data.\n\nOnly 10 of the 19 MENA economies have monthly or quarterly information on industrial production-for the remaining nine economies, information is not readily available.\n\nOnly eight economies report quarterly unemployment data, while none have monthly data.\n\nThe benchmark country, Mexico, reports unemployment data monthly.\n\nA caveat is that the table does not consider quality.\n\nFor example, definitions of unemployment may be inconsistent with international standards (Arezki et al., 2020) .\n\nOn top of the lack of published monthly unemployment data in the MENA region, there are also challenges regarding the definitions employed.\n\nFor examples, countries around the world usually follow the ILO's definitions of employment and unemployment, which are consistent with definitions adopted by other developed countries, such as the United States (see Table 4 ).")
se = "Of the 140 potential microdata sets explored (seven data categories in 20 countries), 78 had been collected and 22 were accessible.\n\nMoreover, about a third of these 22 datasets were not accessible through the website of the National Statistics Office but had to be found on international microdata repositories, such as the World Bank, International Household Survey Network (IHSN) microdata library, IPUMS, Eurostat, Demographic and Health Surveys (DHS), and Multiple Indicator Cluster Surveys (MICS).\n\nWhile the SCI, SPI, and ODIN indicators capture the quality of the data ecosystem to a large extent, they do not necessarily encompass the extent and availability of high-frequency data.\n\nTable 3 presents the availability of high-frequency data on GDP, industrial production, and unemployment for 19 economies in the MENA region.\n\nThe date of the most recent data available is also indicated.\n\nThis information is compiled from the websites of various statistical offices or data portal initiatives, central banks and ministries of planning, economy, or finance across the MENA region.\n\nThe findings are benchmarked against Mexico, which serves as a good comparator for the MENA region given its upper-middle-income economy status and a well-functioning data ecosystem.\n\nOf the 19 economies in the MENA region, 15 report quarterly data on GDP.\n\nSome economies lack information for the year 2020 entirely.\n\nEconomies in conflict such as Libya (2014) and Yemen (2017) have outdated data.\n\nOnly 10 of the 19 MENA economies have monthly or quarterly information on industrial production-for the remaining nine economies, information is not readily available.\n\nOnly eight economies report quarterly unemployment data, while none have monthly data.\n\nThe benchmark country, Mexico, reports unemployment data monthly.\n\nA caveat is that the table does not consider quality.\n\nFor example, definitions of unemployment may be inconsistent with international standards (Arezki et al., 2020) .\n\nOn top of the lack of published monthly unemployment data in the MENA region, there are also challenges regarding the definitions employed.\n\nFor examples, countries around the world usually follow the ILO's definitions of employment and unemployment, which are consistent with definitions adopted by other developed countries, such as the United States (see Table 4 )."
for i, j in enumerate(se.split("\n\n"), 1):
    print(i, j)
    print()
    
for i, j in enumerate(se.split("\n\n"), 1):
    print(f"sent_{i:02}:", j)
    print()
    
import requests
from bs4 import BeautifulSoup


url = "https://www.worldbank.org/en/topic"

response = requests.get(url)
soup = BeautifulSoup(response)
soup = BeautifulSoup(response.content)
lis = soup.find_all("li")
len(lis)
from lxml import html
tree = html.fromstring(response.content)
uls = tree.xpath("/html/body/div[3]/div[2]/div/div/div/div/div[1]/div[1]/div/div/div/div/div/div/div[2]/div/div[2]/div/ul")
len(uls)
uls
lis[0]
lis[1]
lis = soup.find_all("li", attrs=dict(href=re.compile("https://www.worldbank.org/en/topic/")))#=re.compile('Biology'))
import re
lis = soup.find_all("li", attrs=dict(href=re.compile("https://www.worldbank.org/en/topic/")))#=re.compile('Biology'))
len(lis)
lis = soup.find_all("li", attrs=dict(href=re.compile(r"https://www.worldbank.org/en/topic/")))#=re.compile('Biology'))
len(lis)
lis = soup.find_all("li.a", attrs=dict(href=re.compile(r"https://www.worldbank.org/en/topic/")))#=re.compile('Biology'))
len(lis)
lis = soup.find_all("li>a", attrs=dict(href=re.compile(r"https://www.worldbank.org/en/topic/")))#=re.compile('Biology'))
len(lis)
lis = soup.find_all("a", attrs=dict(href=re.compile(r"https://www.worldbank.org/en/topic/")))#=re.compile('Biology'))
len(lis)
lis[0]
topics = {li.text.strip() for li in lis}
len(topics)
topics = sorted({li.text.strip() for li in lis})
topics[0]
topics[1]
topics[2]
topics[3]
topics[4]
topics[5]
topics[6]
topics = soup.find_all("div", attrs={"class": "topic-list"})
len(topics)
topics[0]
soup = BeautifulSoup(response.text, "html.parser")
topics = soup.find_all("div", attrs={"class": "topic-list"})
len(topics)
topics[0].find("h4").find("a").text
topics[0].find("h4").find("a").text.strip("\xa0»")
topic_names = sorted({topic.find("h4").find("a").text.strip("\xa0»") for topic in topics})
topic_names
print("!!!!!".join(topic_names))
tr = """[{"sentence": "We analyze data from the World Bank Enterprise Surveys and find that access to finance is a major constraint for firms in low-income countries.", "dataset": ["World Bank Enterprise Surveys"], "topic": ["Financial Sector"]}, {"sentence": "Using data from the International Energy Agency, we estimate that global carbon dioxide emissions from energy production increased by 1.7% in 2018.", "dataset": ["International Energy Agency"], "topic": ["Climate Change"]}, {"sentence": "This study uses data from the Demographic and Health Surveys to explore the relationship between women's empowerment and child malnutrition in sub-Saharan Africa.", "dataset": ["Demographic and Health Surveys"], "topic": ["Nutrition"]}, {"sentence": "We use satellite data from NASA to estimate the impact of temperature on crop yields in India.", "dataset": ["NASA"], "topic": ["Agriculture and Food"]}, {"sentence": "This paper uses data from the World Values Survey and finds that trust in government is positively associated with economic growth.", "dataset": ["World Values Survey"], "topic": ["Governance"]}, {"sentence": "We combine data from the World Bank's Doing Business Report and the World Economic Forum's Global Competitiveness Report to identify the key drivers of business competitiveness in developing countries.", "dataset": ["Doing Business Report", "Global Competitiveness Report"], "topic": ["Competitiveness"]}, {"sentence": "Using data from the Open Budget Survey, we examine the relationship between budget transparency and corruption in Latin America.", "dataset": ["Open Budget Survey"], "topic": ["Governance"]}, {"sentence": "This study uses data from the World Bank's Enterprise Surveys and finds that access to electricity is a major constraint for firms in Sub-Saharan Africa.", "dataset": ["World Bank's Enterprise Surveys"], "topic": ["Energy"]}, {"sentence": "We use data from the Global Burden of Disease Study to estimate the burden of non-communicable diseases in low-income countries.", "dataset": ["Global Burden of Disease Study"], "topic": ["Health"]}, {"sentence": "This paper analyzes data from the World Inequality Database and finds that income inequality has risen significantly in many countries over the past decade.", "dataset": ["World Inequality Database"], "topic": ["Inequality and Shared Prosperity"]}]"""
print(json.loads(tr, indent=2))
print(json.dumps(json.loads(tr), indent=2))
prompt = """Forget all previous instructions.

You are an expert in following instructions. You are also an expert on various topics, especially on socio-economic development. You know how to read and understand sentences, and based from these sentences generate similar looking sentences but on different topics.

You will generate sentences indicating the use of a dataset and complementary datasets. Always mention specific datasets in the sentence.

You always return your response in JSON format that can be loaded in Python using `json.loads`. The format looks like:  [{"sentence": <sentence>, "dataset": [<dataset>], "topic": [<topic>]}]

Use topic only from this list separated by !!!!!: Agriculture and Food!!!!!Climate Change!!!!!Competitiveness!!!!!Debt!!!!!Digital Development!!!!!Disaster Risk Management !!!!!Education!!!!!Energy!!!!!Environment !!!!!Extractive Industries!!!!!Financial Inclusion!!!!!Financial Sector!!!!!Fragility, Conflict, and Violence!!!!!Gender!!!!!Governance!!!!!Health!!!!!Inequality and Shared Prosperity!!!!!Infrastructure!!!!!Jobs & Development!!!!!Macroeconomics!!!!!Migration!!!!!Nutrition!!!!!Poverty!!!!!Public-Private Partnerships!!!!!Regional Integration!!!!!Social Protection!!!!!Social Sustainability and Inclusion!!!!!Trade!!!!!Transport!!!!!Urban Development!!!!!Water

Generate 10 sentences on various topics that use datasets. The sentence lengths must have large variation with each other.

Consider generating complex sentence structure. Also generate sentences where multiple datasets are mentioned. Try to generate challenging and somewhat ambiguous sentences.

Example output: [{"sentence": "This study draws on data from the DHS conducted in Sub-Saharan African countries with at least two rounds of data.", "dataset": ["DHS"]}, {"sentence": "This paper uses administrative data on electricity billing records from Ghana and Rwanda.", "dataset": ["administrative data on electricity billing records"]}, {"sentence": "We complement the electricity data with monthly data on temperature and total precipitation from the Copernicus Climate Change Service.", "dataset": ["Copernicus Climate Change Service"]}, {"sentence": "We use two main sources of data, both of which are novel: a dataset on the universe of Indonesian exporters in the period 2014-18 and a time-varying dataset of NTMs applied on Indonesian imports.", "dataset": ["dataset on the universe of Indonesian exporters in the period 2014-18", "time-varying dataset of NTMs applied on Indonesian imports"]}, {"sentence": "To this end, we used multi-modal imaging and neuropsychological battery data available in the Alzheimer's Disease Neuroimaging Initiative (ADNI) to investigate the relationship between cross-sectional measures of tau, cortical thickness, and different aspects of cognition.", "dataset": ["Alzheimer's Disease Neuroimaging Initiative (ADNI)"]}]

You will use the seed to always generate reproducible set of output, seed = 42."""


prompt2 = """Forget all previous instructions.

You are an expert in following instructions. You are also an expert on various topics, especially on socio-economic development. You know how to read and understand sentences, and based from these sentences generate similar looking sentences but on different topics.

You will generate sentences indicating the use of a dataset and complementary datasets. Always mention specific datasets in the sentence.

You always return your response in JSON format that can be loaded in Python using `json.loads`. The format looks like:  [{"sentence": <sentence>, "dataset": [<dataset>], "topic": [<topic>]}]

Use topic only from this list separated by !!!!!: Agriculture and Food!!!!!Climate Change!!!!!Competitiveness!!!!!Debt!!!!!Digital Development!!!!!Disaster Risk Management !!!!!Education!!!!!Energy!!!!!Environment !!!!!Extractive Industries!!!!!Financial Inclusion!!!!!Financial Sector!!!!!Fragility, Conflict, and Violence!!!!!Gender!!!!!Governance!!!!!Health!!!!!Inequality and Shared Prosperity!!!!!Infrastructure!!!!!Jobs & Development!!!!!Macroeconomics!!!!!Migration!!!!!Nutrition!!!!!Poverty!!!!!Public-Private Partnerships!!!!!Regional Integration!!!!!Social Protection!!!!!Social Sustainability and Inclusion!!!!!Trade!!!!!Transport!!!!!Urban Development!!!!!Water

Generate 10 sentences on various topics that use datasets. The sentence lengths must have large variation with each other.

Consider generating complex sentence structure. Also generate sentences where multiple datasets are mentioned. Try to generate challenging and somewhat ambiguous sentences.

Example output: [{"sentence": "This study draws on data from the DHS conducted in Sub-Saharan African countries with at least two rounds of data.", "dataset": ["DHS"]}, {"sentence": "This paper uses administrative data on electricity billing records from Ghana and Rwanda.", "dataset": ["administrative data on electricity billing records"]}, {"sentence": "We complement the electricity data with monthly data on temperature and total precipitation from the Copernicus Climate Change Service.", "dataset": ["Copernicus Climate Change Service"]}, {"sentence": "We use two main sources of data, both of which are novel: a dataset on the universe of Indonesian exporters in the period 2014-18 and a time-varying dataset of NTMs applied on Indonesian imports.", "dataset": ["dataset on the universe of Indonesian exporters in the period 2014-18", "time-varying dataset of NTMs applied on Indonesian imports"]}, {"sentence": "To this end, we used multi-modal imaging and neuropsychological battery data available in the Alzheimer's Disease Neuroimaging Initiative (ADNI) to investigate the relationship between cross-sectional measures of tau, cortical thickness, and different aspects of cognition.", "dataset": ["Alzheimer's Disease Neuroimaging Initiative (ADNI)"]}]

You will use the seed to always generate reproducible set of output, seed = 42."""
prompt == prompt2
(1375 / 1000) * 0.002
100 * (1375 / 1000) * 0.002
ex = """[{"sentence": "This study draws on data from the DHS conducted in Sub-Saharan African countries with at least two rounds of data.", "dataset": ["DHS"], "topic": ["Health"]}, {"sentence": "This paper uses administrative data on electricity billing records from Ghana and Rwanda.", "dataset": ["administrative data on electricity billing records"], "topic": ["Energy"]}, {"sentence": "We complement the electricity data with monthly data on temperature and total precipitation from the Copernicus Climate Change Service.", "dataset": ["Copernicus Climate Change Service"], "topic": ["Energy", "Climate Change"]}, {"sentence": "We use two main sources of data, both of which are novel: a dataset on the universe of Indonesian exporters in the period 2014-18 and a time-varying dataset of NTMs applied on Indonesian imports.", "dataset": ["dataset on the universe of Indonesian exporters in the period 2014-18", "time-varying dataset of NTMs applied on Indonesian imports"], "topic": ["Trade"]}, {"sentence": "To this end, we used multi-modal imaging and neuropsychological battery data available in the Alzheimer's Disease Neuroimaging Initiative (ADNI) to investigate the relationship between cross-sectional measures of tau, cortical thickness, and different aspects of cognition.", "dataset": ["Alzheimer's Disease Neuroimaging Initiative (ADNI)"], "topic": ["Health"]}]"""
json.loads(ex)
shortlist = index.vectorstore.similarity_search_with_score(query, k=20)
get_sample_sentences(id_num=7, passage_embeddings=passage_embeddings, query="What dataset was used in the paper?", k=10)
get_sample_sentences(id_num=7, passage_embeddings=passage_embeddings, query="What dataset was used in the paper?", k=10)
get_sample_sentences(id_num=1, passage_embeddings=passage_embeddings, query="What dataset was used in the paper?", k=10)
passage_embeddings.client
passage_embeddings
get_sample_sentences(id_num=9, passage_embeddings=passage_embeddings, query="What dataset was used in the paper?", k=10)
# ids = [j.stem for j in coleridge_dir.glob("*.json")]
# ids = [j.stem for j in coleridge_dir.glob("*.json")]
# coleridge_dir = Path("data/coleridgeinitiative/coleridgeinitiative-show-us-the-data (1)/train/")
get_sample_sentences(id_num=93, passage_embeddings=passage_embeddings, query="What dataset was used in the paper?", k=10)
len(ids)
index.vectorstore
index.vectorstore.embedding_function
def get_sample_sentences_from_path(path, jq_schema, query, sentence_embeddings, k=20, split_docs: bool = False):
    # path = coleridge_dir / f"{ids[id_num]}.json"
    # jq_schema = ".[].text"
    loader = sj.StructuredJSONLoader(path, jq_schema=jq_schema)
    docs = loader.load()
    documents = du.aggregate_documents(docs)

    if split_docs:
        sdocs = du.split_documents(documents)
    else:
        sdocs = documents

    collection_name = None

    if vectorstore_cls == "Qdrant":
        collection_name = "coleridge"
        index = dvectorstore.VectorstoreIndexCreator(
            vectorstore_cls=Qdrant,
            embedding=sentence_embeddings,
            vectorstore_kwargs=dict(
                location=":memory:",
                collection_name=collection_name,
            ),
        ).from_documents(sdocs)
    elif vectorstore_cls == "Chroma":
        index = dvectorstore.VectorstoreIndexCreator(
            vectorstore_cls=Chroma,
            embedding=sentence_embeddings,
        ).from_documents(sdocs)
    else:
        raise ValueError(f"Unknown vectorstore_cls: {vectorstore_cls}")

    shortlist = index.vectorstore.similarity_search(query, k=k)
    shortlist = [rs.page_content for rs in shortlist]

    try:
        # Delete the collection
        # ChromaDB vectorstore
        index.vectorstore.delete_collection()
    except AttributeError:
        try:
            # Delete the collection
            # Qdrant vectorstore
            if collection_name is not None:
                index.vectorstore.client.delete_collection(collection_name)
        except:
            pass

    embedding_config = sentence_embeddings.dict()
    embedding_config["client"] = embedding_config["client"].__repr__()
    doc_options = dict(
        skip_urls=skip_urls,
        min_sentence_len=min_sentence_len,
    )

    return dict(
        doc_id=doc_path.stem,
        query=query,
        k=k,
        shortlist=shortlist,
        embedding_config=embedding_config,
        vectorstore_cls=vectorstore_cls,
        doc_options=doc_options,
    )
    
# out = get_sample_sentences_from_path(Path("data/prwp/D9912072.json"), jq_schema="
p = json.loads(Path("data/prwp/D9912072.json").read_text())
oo = jq.compile(".pdf_parse.body_text").input(p).all()
import jq
oo = jq.compile(".pdf_parse.body_text").input(p).all()
len(oo)
type(oo)
oo[0]
oo = jq.compile(".pdf_parse.body_text.[]text").input(p).all()
oo = jq.compile(".pdf_parse.body_text.text").input(p).all()
oo = jq.compile(".pdf_parse.body_text[]text").input(p).all()
oo = jq.compile(".pdf_parse.body_text[].text").input(p).all()
len(oo)
oo[0]
out = get_sample_sentences_from_path(Path("data/prwp/D9912072.json"), jq_schema=".pdf_parse.body_text[].text", query="What dataset was used in the paper?", sentence_embeddings=passage_embeddings)
def get_sample_sentences_from_path(path, jq_schema, query, sentence_embeddings, k=20, split_docs: bool = False, skip_urls: bool = True, min_sentence_len: int = 25, vectorstore_cls = "Qdrant"):
    # path = coleridge_dir / f"{ids[id_num]}.json"
    # cole_jq_schema = ".[].text"
    # prwp_jq_schema=".pdf_parse.body_text[].text"
    loader = sj.StructuredJSONLoader(path, jq_schema=jq_schema)
    docs = loader.load()
    documents = du.aggregate_documents(docs, skip_urls=skip_urls, min_sentence_len=min_sentence_len)

    if split_docs:
        sdocs = du.split_documents(documents)
    else:
        sdocs = documents

    collection_name = None

    if vectorstore_cls == "Qdrant":
        collection_name = "coleridge"
        index = dvectorstore.VectorstoreIndexCreator(
            vectorstore_cls=Qdrant,
            embedding=sentence_embeddings,
            vectorstore_kwargs=dict(
                location=":memory:",
                collection_name=collection_name,
            ),
        ).from_documents(sdocs)
    elif vectorstore_cls == "Chroma":
        index = dvectorstore.VectorstoreIndexCreator(
            vectorstore_cls=Chroma,
            embedding=sentence_embeddings,
        ).from_documents(sdocs)
    else:
        raise ValueError(f"Unknown vectorstore_cls: {vectorstore_cls}")

    shortlist = index.vectorstore.similarity_search(query, k=k)
    shortlist = [rs.page_content for rs in shortlist]

    try:
        # Delete the collection
        # ChromaDB vectorstore
        index.vectorstore.delete_collection()
    except AttributeError:
        try:
            # Delete the collection
            # Qdrant vectorstore
            if collection_name is not None:
                index.vectorstore.client.delete_collection(collection_name)
        except:
            pass

    embedding_config = sentence_embeddings.dict()
    embedding_config["client"] = embedding_config["client"].__repr__()
    doc_options = dict(
        skip_urls=skip_urls,
        min_sentence_len=min_sentence_len,
    )

    return dict(
        doc_id=doc_path.stem,
        query=query,
        k=k,
        shortlist=shortlist,
        embedding_config=embedding_config,
        vectorstore_cls=vectorstore_cls,
        doc_options=doc_options,
    )
    
out = get_sample_sentences_from_path(Path("data/prwp/D9912072.json"), jq_schema=".pdf_parse.body_text[].text", query="What dataset was used in the paper?", sentence_embeddings=passage_embeddings)
def get_sample_sentences_from_path(doc_path, jq_schema, query, sentence_embeddings, k=20, split_docs: bool = False, skip_urls: bool = True, min_sentence_len: int = 25, vectorstore_cls = "Qdrant"):
    # path = coleridge_dir / f"{ids[id_num]}.json"
    # cole_jq_schema = ".[].text"
    # prwp_jq_schema=".pdf_parse.body_text[].text"
    loader = sj.StructuredJSONLoader(doc_path, jq_schema=jq_schema)
    docs = loader.load()
    documents = du.aggregate_documents(docs, skip_urls=skip_urls, min_sentence_len=min_sentence_len)

    if split_docs:
        sdocs = du.split_documents(documents)
    else:
        sdocs = documents

    collection_name = None

    if vectorstore_cls == "Qdrant":
        collection_name = "coleridge"
        index = dvectorstore.VectorstoreIndexCreator(
            vectorstore_cls=Qdrant,
            embedding=sentence_embeddings,
            vectorstore_kwargs=dict(
                location=":memory:",
                collection_name=collection_name,
            ),
        ).from_documents(sdocs)
    elif vectorstore_cls == "Chroma":
        index = dvectorstore.VectorstoreIndexCreator(
            vectorstore_cls=Chroma,
            embedding=sentence_embeddings,
        ).from_documents(sdocs)
    else:
        raise ValueError(f"Unknown vectorstore_cls: {vectorstore_cls}")

    shortlist = index.vectorstore.similarity_search(query, k=k)
    shortlist = [rs.page_content for rs in shortlist]

    try:
        # Delete the collection
        # ChromaDB vectorstore
        index.vectorstore.delete_collection()
    except AttributeError:
        try:
            # Delete the collection
            # Qdrant vectorstore
            if collection_name is not None:
                index.vectorstore.client.delete_collection(collection_name)
        except:
            pass

    embedding_config = sentence_embeddings.dict()
    embedding_config["client"] = embedding_config["client"].__repr__()
    doc_options = dict(
        skip_urls=skip_urls,
        min_sentence_len=min_sentence_len,
    )

    return dict(
        doc_id=doc_path.stem,
        query=query,
        k=k,
        shortlist=shortlist,
        embedding_config=embedding_config,
        vectorstore_cls=vectorstore_cls,
        doc_options=doc_options,
    )
    
out = get_sample_sentences_from_path(Path("data/prwp/D9912072.json"), jq_schema=".pdf_parse.body_text[].text", query="What dataset was used in the paper?", sentence_embeddings=passage_embeddings)
out[0]
out["shortlist"][0]
print("\n\n".join(out["shortlist"][:3]))
out = get_sample_sentences_from_path(Path("data/prwp/D34010638.json"), jq_schema=".pdf_parse.body_text[].text", query="What dataset was used in the paper?", sentence_embeddings=passage_embeddings)
out["shortlist"][0]
out["shortlist"][1]
out["shortlist"][2]
print("\n\n".join(out["shortlist"][:3]))
out = get_sample_sentences_from_path(Path("data/prwp/D34010638.json"), jq_schema=".pdf_parse.body_text[].text", query="What dataset was used in the paper?", sentence_embeddings=passage_embeddings, split_docs=True)
out["shortlist"][0]
out["shortlist"][1]
print("\n\n".join(out["shortlist"][:10]))
print("\n\n".join(out["shortlist"][:50]))
print("\n\n".join(out["shortlist"][:60]))
out = get_sample_sentences_from_path(Path("data/prwp/D34010638.json"), jq_schema=".pdf_parse.body_text[].text", query="What dataset was used in the paper?", sentence_embeddings=passage_embeddings, split_docs=True, k=50)
print("\n\n".join(out["shortlist"][:25]))
print("\n\n".join(out["shortlist"][:50]))
100 * 50
import json
import random
from pathlib import Path

RANKING_DATA_DIR = Path("data/training/ranking")
N = 100

# Coleridge Initiative
coleridge_dir = Path("data/coleridgeinitiative/coleridgeinitiative-show-us-the-data/train")
ids = list(coleridge_dir.glob("*.json"))
random.shuffle(ids)
ids = ids[:N]

for id_num in range(N):
    name = ids[id_num].name
    path = coleridge_dir / f"{name}"

    with open(path) as f:
        doc = json.load(f)

    with open(RANKING_DATA_DIR / "coleridge" / f"{name}", "w") as f:
        json.dump(doc, f, indent=2)

# World Bank PRWP
prwp_dir = Path("data/prwp")
ids = list(prwp_dir.glob("*.json"))
random.shuffle(ids)
ids = ids[:N]

for id_num in range(N):
    name = ids[id_num].name
    path = prwp_dir / f"{name}"

    with open(path) as f:
        doc = json.load(f)

    with open(RANKING_DATA_DIR / "prwp" / f"{name}", "w") as f:
        json.dump(doc, f, indent=2)
import json
import random
from pathlib import Path

RANKING_DATA_DIR = Path("data/training/ranking")
N = 100

# Coleridge Initiative
coleridge_dir = Path("data/coleridgeinitiative/coleridgeinitiative-show-us-the-data/train")

ids = list(coleridge_dir.glob("*.json"))
random.shuffle(ids)
ids = ids[:N]

for id_num in range(N):
    name = ids[id_num].name
    path = coleridge_dir / f"{name}"

    with open(path) as f:
        doc = json.load(f)

    fname = RANKING_DATA_DIR / "coleridge" / f"{name}"
    fname.parent.mkdir(exist_ok=True, parents=True)

    fname.write_text(json.dumps(doc, indent=2))

# World Bank PRWP
prwp_dir = Path("data/prwp")
ids = list(prwp_dir.glob("*.json"))
random.shuffle(ids)
ids = ids[:N]

for id_num in range(N):
    name = ids[id_num].name
    path = prwp_dir / f"{name}"

    with open(path) as f:
        doc = json.load(f)

    fname = RANKING_DATA_DIR / "prwp" / f"{name}"
    fname.parent.mkdir(exist_ok=True, parents=True)

    fname.write_text(json.dumps(doc, indent=2))
sentence_embeddings = HuggingFaceInstructEmbeddings(
    embed_instruction="Represent the sentence for retrieval optimized for answering questions on the mention of data; Input: ",
    query_instruction="Represent the question for retrieving the most relevant sentence mentioning data; Input: ",
)

QUERY = "What dataset was used in the paper?"


def get_sample_sentences_from_path(doc_path, jq_schema, query, sentence_embeddings, k=20, split_docs: bool = False, skip_urls: bool = True, min_sentence_len: int = 25, vectorstore_cls = "Qdrant"):
    # path = coleridge_dir / f"{ids[id_num]}.json"
    # cole_jq_schema = ".[].text"
    # prwp_jq_schema=".pdf_parse.body_text[].text"
    loader = sj.StructuredJSONLoader(doc_path, jq_schema=jq_schema)
    docs = loader.load()
    documents = du.aggregate_documents(docs, skip_urls=skip_urls, min_sentence_len=min_sentence_len)

    if split_docs:
        sdocs = du.split_documents(documents)
    else:
        sdocs = documents

    collection_name = None

    if vectorstore_cls == "Qdrant":
        collection_name = "coleridge"
        index = dvectorstore.VectorstoreIndexCreator(
            vectorstore_cls=Qdrant,
            embedding=sentence_embeddings,
            vectorstore_kwargs=dict(
                location=":memory:",
                collection_name=collection_name,
            ),
        ).from_documents(sdocs)
    elif vectorstore_cls == "Chroma":
        index = dvectorstore.VectorstoreIndexCreator(
            vectorstore_cls=Chroma,
            embedding=sentence_embeddings,
        ).from_documents(sdocs)
    else:
        raise ValueError(f"Unknown vectorstore_cls: {vectorstore_cls}")

    shortlist = index.vectorstore.similarity_search(query, k=k)
    shortlist = [rs.page_content for rs in shortlist]

    try:
        # Delete the collection
        # ChromaDB vectorstore
        index.vectorstore.delete_collection()
    except AttributeError:
        try:
            # Delete the collection
            # Qdrant vectorstore
            if collection_name is not None:
                index.vectorstore.client.delete_collection(collection_name)
        except:
            pass

    embedding_config = sentence_embeddings.dict()
    embedding_config["client"] = embedding_config["client"].__repr__()
    doc_options = dict(
        skip_urls=skip_urls,
        min_sentence_len=min_sentence_len,
    )

    return dict(
        doc_id=doc_path.stem,
        query=query,
        k=k,
        shortlist=shortlist,
        embedding_config=embedding_config,
        vectorstore_cls=vectorstore_cls,
        doc_options=doc_options,
    )
    
jq_schema_map = dict(
    coleridge=".text",
    prwp=".pdf_parse.body_text[].text",
)
for dataset in ["coleridge", "prwp"]:
    print(f"Processing {dataset}")
    data_dir = RANKING_DATA_DIR / dataset
    out_dir = RANKING_DATA_DIR / f"shortlist"
    out_dir.mkdir(exist_ok=True, parents=True)

    for path in tqdm(list(data_dir.glob("*.json"))):
        out = get_sample_sentences_from_path(
            path, jq_schema=jq_schema_map[dataset], query=QUERY, sentence_embeddings=sentence_embeddings, split_docs=True, k=50, skip_urls=True, min_sentence_len=25, vectorstore_cls="Qdrant")

        out_path = out_dir / f"{dataset}_{path.stem}.json"
        out_path.write_text(json.dumps(out, indent=2))
from tqdm.auto import tqdm
for dataset in ["coleridge", "prwp"]:
    print(f"Processing {dataset}")
    data_dir = RANKING_DATA_DIR / dataset
    out_dir = RANKING_DATA_DIR / f"shortlist"
    out_dir.mkdir(exist_ok=True, parents=True)

    for path in tqdm(list(data_dir.glob("*.json"))):
        out = get_sample_sentences_from_path(
            path, jq_schema=jq_schema_map[dataset], query=QUERY, sentence_embeddings=sentence_embeddings, split_docs=True, k=50, skip_urls=True, min_sentence_len=25, vectorstore_cls="Qdrant")

        out_path = out_dir / f"{dataset}_{path.stem}.json"
        out_path.write_text(json.dumps(out, indent=2))
        
jq_schema_map = dict(
    coleridge=".[].text",
    prwp=".pdf_parse.body_text[].text",
)
for dataset in ["coleridge", "prwp"]:
    print(f"Processing {dataset}")
    data_dir = RANKING_DATA_DIR / dataset
    out_dir = RANKING_DATA_DIR / f"shortlist"
    out_dir.mkdir(exist_ok=True, parents=True)

    for path in tqdm(sorted(data_dir.glob("*.json"))):
        out = get_sample_sentences_from_path(
            path, jq_schema=jq_schema_map[dataset], query=QUERY, sentence_embeddings=sentence_embeddings, split_docs=True, k=50, skip_urls=True, min_sentence_len=25, vectorstore_cls="Qdrant")

        out_path = out_dir / f"{dataset}_{path.stem}.json"
        out_path.write_text(json.dumps(out, indent=2))
import os
import json
import random
from pathlib import Path

RANKING_DATA_DIR = Path("data/training/ranking")
N = 100

# Coleridge Initiative
coleridge_dir = Path("data/coleridgeinitiative/coleridgeinitiative-show-us-the-data/train")
paths = list(coleridge_dir.glob("*.json"))
random.seed(1029)
random.shuffle(paths)
ids = []

for path in paths:
    name = path.name

    if os.path.getsize(path) > 3_000_000: # skip large files
        continue

    with open(path) as f:
        doc = json.load(f)

    fname = RANKING_DATA_DIR / "coleridge" / f"{name}"
    fname.parent.mkdir(exist_ok=True, parents=True)

    fname.write_text(json.dumps(doc, indent=2))

    ids.append(name)
    if len(ids) == N:
        break

# World Bank PRWP
prwp_dir = Path("data/prwp")
paths = list(prwp_dir.glob("*.json"))
random.seed(1029)
random.shuffle(paths)
ids = []

for path in paths:
    name = path.name

    if os.path.getsize(path) > 3_000_000: # skip large files
        continue

    with open(path) as f:
        doc = json.load(f)

    fname = RANKING_DATA_DIR / "prwp" / f"{name}"
    fname.parent.mkdir(exist_ok=True, parents=True)

    fname.write_text(json.dumps(doc, indent=2))

    ids.append(name)
    if len(ids) == N:
        break
sentence_embeddings = HuggingFaceInstructEmbeddings(
    embed_instruction="Represent the sentence for retrieval optimized for answering questions on the mention of data; Input: ",
    query_instruction="Represent the question for retrieving the most relevant sentence mentioning data; Input: ",
)

QUERY = "What dataset was used in the paper?"


def get_sample_sentences_from_path(doc_path, jq_schema, query, sentence_embeddings, k=20, split_docs: bool = False, skip_urls: bool = True, min_sentence_len: int = 25, vectorstore_cls = "Qdrant"):
    # path = coleridge_dir / f"{ids[id_num]}.json"
    # cole_jq_schema = ".[].text"
    # prwp_jq_schema=".pdf_parse.body_text[].text"
    loader = sj.StructuredJSONLoader(doc_path, jq_schema=jq_schema)
    docs = loader.load()
    documents = du.aggregate_documents(docs, skip_urls=skip_urls, min_sentence_len=min_sentence_len)

    if split_docs:
        sdocs = du.split_documents(documents)
    else:
        sdocs = documents

    collection_name = None

    if vectorstore_cls == "Qdrant":
        collection_name = "coleridge"
        index = dvectorstore.VectorstoreIndexCreator(
            vectorstore_cls=Qdrant,
            embedding=sentence_embeddings,
            vectorstore_kwargs=dict(
                location=":memory:",
                collection_name=collection_name,
            ),
        ).from_documents(sdocs)
    elif vectorstore_cls == "Chroma":
        index = dvectorstore.VectorstoreIndexCreator(
            vectorstore_cls=Chroma,
            embedding=sentence_embeddings,
        ).from_documents(sdocs)
    else:
        raise ValueError(f"Unknown vectorstore_cls: {vectorstore_cls}")

    shortlist = index.vectorstore.similarity_search(query, k=k)
    shortlist = [rs.page_content for rs in shortlist]

    try:
        # Delete the collection
        # ChromaDB vectorstore
        index.vectorstore.delete_collection()
    except AttributeError:
        try:
            # Delete the collection
            # Qdrant vectorstore
            if collection_name is not None:
                index.vectorstore.client.delete_collection(collection_name)
        except:
            pass

    embedding_config = sentence_embeddings.dict()
    embedding_config["client"] = embedding_config["client"].__repr__()
    doc_options = dict(
        skip_urls=skip_urls,
        min_sentence_len=min_sentence_len,
    )

    return dict(
        doc_id=doc_path.stem,
        query=query,
        k=k,
        shortlist=shortlist,
        embedding_config=embedding_config,
        vectorstore_cls=vectorstore_cls,
        doc_options=doc_options,
    )
    
for dataset in ["coleridge", "prwp"]:
    print(f"Processing {dataset}")
    data_dir = RANKING_DATA_DIR / dataset
    out_dir = RANKING_DATA_DIR / f"shortlist"
    out_dir.mkdir(exist_ok=True, parents=True)

    for path in tqdm(sorted(data_dir.glob("*.json"))):
        print(path)
        out = get_sample_sentences_from_path(
            path, jq_schema=jq_schema_map[dataset], query=QUERY, sentence_embeddings=sentence_embeddings, split_docs=True, k=50, skip_urls=True, min_sentence_len=25, vectorstore_cls="Qdrant")

        out_path = out_dir / f"{dataset}_{path.stem}.json"
        out_path.write_text(json.dumps(out, indent=2))
for dataset in ["coleridge", "prwp"]:
    print(f"Processing {dataset}")
    data_dir = RANKING_DATA_DIR / dataset
    out_dir = RANKING_DATA_DIR / f"shortlist"
    out_dir.mkdir(exist_ok=True, parents=True)

    for path in tqdm(sorted(data_dir.glob("*.json"))):
        print(path)
        out_path = out_dir / f"{dataset}_{path.stem}.json"

        if out_path.exists():
            continue

        out = get_sample_sentences_from_path(
            path, jq_schema=jq_schema_map[dataset], query=QUERY, sentence_embeddings=sentence_embeddings, split_docs=True, k=50, skip_urls=True, min_sentence_len=25, vectorstore_cls="Qdrant")

        out_path.write_text(json.dumps(out, indent=2))
import json
import pandas as pd
from pathlib import Path

all_df = []
RANKING_DIR = Path("data/training/ranking")
for path in sorted((RANKING_DIR / "shortlist").glob("*.json")):
    with open(path) as f:
        doc = json.load(f)

    df = pd.DataFrame(doc["shortlist"], columns=["text"])
    df["doc_id"] = doc["doc_id"]
    df["rank"] = df.index + 1

    all_df.append(df)

all_df = pd.concat(all_df)
all_df["is_relevant"] = None
all_df.to_csv((RANKING_DIR / "ranking_shortlist.xlsx"), index=None)
all_df.to_excel((RANKING_DIR / "ranking_shortlist.xlsx"), index=None)
all_df.sort_values(["rank", "doc_id"]).to_excel((RANKING_DIR / "ranking_shortlist.xlsx"), index=None)
all_df.groupby("doc_id").cumcount() + 1
all_df.head()
all_df = all_df.sort_values(["rank", "doc_id"])
all_df.groupby("doc_id").cumcount() + 1
enc.encode("01860fa5-2c39-4ea2-9124-74458ae4a4b4")
len(enc.encode("01860fa5-2c39-4ea2-9124-74458ae4a4b4"))
len(enc.encode("sent_001"))
len(enc.encode("sent_002"))
len(enc.encode("sent_100"))
len(enc.encode("sent_999"))
len(enc.encode("sent_0999"))
len(enc.encode("s_0999"))
len(enc.encode("_0999"))
len(enc.encode("0999"))
len(enc.encode("s0999"))
len(enc.encode("s_0999"))
len(enc.encode("s_00999"))
len(enc.encode("s_00099"))
len(enc.encode("s_00009"))
all_df["sent_id"] = list(range(1, all_df.shape[0] + 1))
all_sd
all_df
all_df = all_df.sort_values(["doc_id", "rank"])
all_df["sent_id"] = list(range(1, all_df.shape[0] + 1))
all_df
all_df["sent_id"] = "s_" + all_df["sent_id"].astype(str).str.zfill(5)
all_df
all_df = all_df.sort_values(["rank", "doc_id"])
all_df
all_df.to_excel((RANKING_DIR / "ranking_shortlist.xlsx"), index=None)
all_df[["sent_id", "text"]].head()
all_df[["sent_id", "text"]].head().apply(": ".join)
all_df[["sent_id", "text"]].head().apply(": ".join, axis=1)
all_df[["sent_id", "text"]].head().apply(": ".join, axis=1).tolist()
print("\n\n".join(all_df[["sent_id", "text"]].head().apply(": ".join, axis=1).tolist()))
print("\n\n".join(all_df[["sent_id", "text"]].head(20).apply(": ".join, axis=1).tolist()))
print("\n\n".join(all_df[["sent_id", "text"]].head(40).tail(20).apply(": ".join, axis=1).tolist()))
op = ["s_00001", "s_00101", "s_00151", "s_00448", "s_00498", "s_00598", "s_00640", "s_00790", "s_00840", "s_00890", "s_00940", "s_01040", "s_01190", "s_01340", "s_01390", "s_01440"]
all_df[all_df["sent_id"].isin(op)]
print("\n".join(all_df[all_df["sent_id"].isin(op)]["text"]))
op = ["s_01040", "s_01340", "s_01390", "s_01490", "s_01790", "s_01840", "s_01890"]
op = ["s_01040", "s_01190", "s_01340", "s_01440", "s_01490", "s_01540", "s_01790", "s_01840", "s_01890"]
print("\n".join(all_df[all_df["sent_id"].isin(op)]["text"]))
op = ["s_01040", "s_01190", "s_01440", "s_01490"]
print("\n".join(all_df[all_df["sent_id"].isin(op)]["text"]))
ops = ["s_01040", "s_01190", "s_01340", "s_01440", "s_01490", "s_01790", "s_01840", "s_01890"]
print("\n".join(all_df[all_df["sent_id"].isin(op)]["text"]))
op = ["s_01040", "s_01190", "s_01340", "s_01440", "s_01490", "s_01790", "s_01840", "s_01890"]
print("\n".join(all_df[all_df["sent_id"].isin(op)]["text"]))
print("\n\n".join(all_df[["sent_id", "text"]].head(60).tail(20).apply(": ".join, axis=1).tolist()))
op = ["s_01990", "s_02040", "s_02090", "s_02140", "s_02190", "s_02323", "s_02423", "s_02523", "s_02573", "s_02723", "s_02773", "s_02873", "s_02923"]




print("\n".join(all_df[all_df["sent_id"].isin(op)]["text"]))
op = ["s_01990", "s_02040", "s_02190", "s_02323", "s_02473", "s_02573", "s_02773"]
print("\n".join(all_df[all_df["sent_id"].isin(op)]["text"]))
op = ["s_02040", "s_02090", "s_02140", "s_02190", "s_02323", "s_02473", "s_02573", "s_02773"]
print("\n".join(all_df[all_df["sent_id"].isin(op)]["text"]))
op = ["s_02040", "s_02090", "s_02140", "s_02190", "s_02323", "s_02473", "s_02573", "s_02773"]
print("\n".join(all_df[all_df["sent_id"].isin(op)]["text"]))
print("\n\n".join(all_df[["sent_id", "text"]].head(80).tail(20).apply(": ".join, axis=1).tolist()))
print("\n\n".join(all_df[["sent_id", "text"]].head(100).tail(20).apply(": ".join, axis=1).tolist()))
pl = json.loads("""[{"sent_id": "s_03973", "data_mentioned": true, "reason": "Demographic and Health Surveys"}, {"sent_id": "s_04223", "data_mentioned": false, "reason": ""}, {"sent_id": "s_04273", "data_mentioned": true, "reason": "oil-price-GDP relationship in major OECD economies"}, {"sent_id": "s_04323", "data_mentioned": true, "reason": "combination of survey and administrative data"}, {"sent_id": "s_04373", "data_mentioned": false, "reason": ""}, {"sent_id": "s_04423", "data_mentioned": false, "reason": ""}, {"sent_id": "s_04473", "data_mentioned": true, "reason": "wages of newly-hired employees as opposed to incumbents"}, {"sent_id": "s_04523", "data_mentioned": false, "reason": ""}, {"sent_id": "s_04573", "data_mentioned": false, "reason": ""}, {"sent_id": "s_04623", "data_mentioned": false, "reason": ""}, {"sent_id": "s_04673", "data_mentioned": false, "reason": ""}, {"sent_id": "s_04723", "data_mentioned": false, "reason": ""}, {"sent_id": "s_04823", "data_mentioned": true, "reason": "data used in this study"}, {"sent_id": "s_04873", "data_mentioned": false, "reason": ""}, {"sent_id": "s_04923", "data_mentioned": false, "reason": ""}]""")
pl
print("\n\n".join(all_df[["sent_id", "text"]].head(90).tail(20).apply(": ".join, axis=1).tolist()))
print("\n\n".join(all_df[["sent_id", "text"]].head(89).tail(20).apply(": ".join, axis=1).tolist()))
print("\n\n".join(all_df[["sent_id", "text"]].head(80).tail(20).apply(": ".join, axis=1).tolist()))
print("\n\n".join(all_df[["sent_id", "text"]].head(100).tail(20).apply(": ".join, axis=1).tolist()))
www = """Forget all previous instructions.

You are an author of research papers. You will be given a list of sentences collected from papers you authored.

You are an expert at identifying which sentences explicitly mention datasets that you used in the paper. You will extract the sentences that satisfy this, but you should only include sentences where the mention of the dataset is clear and unambiguous.

You must exclude sentences where the mention of a source of data is vague or unclear, even if it could potentially be used as a dataset in the paper. For example, do not include sentences that mention software used for data analysis, specific equipment used to collect data, or other tools that are not datasets.

Do not include sentences that merely cite a reference without any mention of the dataset itself.

The list is in the format s_XXXXX: <sentence>. You will only return the sentence ids of the identified sentences as a JSON array and the reason why you included it. If you are uncertain about a sentence, do not include it in the list.

Always return the result in a valid JSON format. Do not truncate the output and never generate ... in the response. The output should not raise a JSONDecodeError when loaded in Python.

Output format: [{"sent_id": s_XXXXX, "dataset_mentioned": <true|false>, "reason": <dataset name>}]

Do not explain.

Text:

s_03973: Later studies use the cross-country data from the latest rounds of the Demographic and Health Surveys (DHS).

s_04023: This section focuses on the results from this study.

s_04073: This paper-a product of the Financial Sector Development Department-is part of a larger effort in the department to study the promotion of pension funds.

s_04123: The analysis in this paper contributes to several strands of the literature.

s_04173: (1999) and Radelet and Sachs (1998) .

s_04223: This paper is an attempt to provide an analytical foundation for such an amendment.

s_04273: Due to limited availability of data, the majority of existing literature analyzes the oil-price-GDP relationship in major OECD economies.

s_04323: Section I explains the combination of survey and administrative data we use in the analysis.

s_04373: (1) Source: Authors' analysis based on data described in the text.

s_04423: This paper contributes to two main literatures.

s_04473: Galuščák et al (2012) , using the same data sets that this paper uses, provide an extensive analysis of the factors that determine the wages of newly-hired employees as opposed to incumbents.

s_04523: (2009) study is again a useful place to start.

s_04573: The paper proceeds as follows.

s_04623: The paper is structured as follows.

s_04673: (2003) for a survey of the literature.

s_04723: The paper is structured as follows.

s_04773: (1987) and references therein.

s_04823: In this section we provide the details of the data used in this study and the empirical methods employed.

s_04873: To address these issues, the paper relies on the instrumental variable spatial autoregressive model (Drukker et al.

s_04923: Annex 1 discusses the variables and their source."""
len(enc.encode(www))
all_df.shape[0]
all_df.shape[0] / 20
500 * 694 / 1000
(500 * 694 / 1000) * 0.002
print("\n\n".join(all_df[["sent_id", "text"]].head(150).tail(20).apply(": ".join, axis=1).tolist()))
www = """Forget all previous instructions.

Given a list of sentences extracted from papers that you authored, extract the sentences that explicitly mention a dataset used in the paper. Only include sentences where the dataset is clearly and unambiguously named, such as LSMS or DHS. Exclude sentences that mention software used for data analysis, equipment used to collect data, or other tools that are not datasets. Also, exclude sentences that only cite a reference without mentioning the dataset or refer to data computed by the author without mentioning the dataset used to derive it.

Your task is to return the sentence ids of the identified sentences as a JSON array along with the reason why you included them. The output format should be [{"sent_id": s_XXXXX, "dataset_mentioned": <true|false>, "reason": <dataset name>}]. If a sentence mentions a dataset clearly and unambiguously, set "dataset_mentioned" to true and fill in the "reason" field with the name of the dataset. If a sentence does not mention a dataset, set "dataset_mentioned" to false and leave the "reason" field empty. If you are uncertain about a sentence, do not include it in the list.

Do not explain.

Text:

s_03973: Later studies use the cross-country data from the latest rounds of the Demographic and Health Surveys (DHS).

s_04023: This section focuses on the results from this study.

s_04073: This paper-a product of the Financial Sector Development Department-is part of a larger effort in the department to study the promotion of pension funds.

s_04123: The analysis in this paper contributes to several strands of the literature.

s_04173: (1999) and Radelet and Sachs (1998) .

s_04223: This paper is an attempt to provide an analytical foundation for such an amendment.

s_04273: Due to limited availability of data, the majority of existing literature analyzes the oil-price-GDP relationship in major OECD economies.

s_04323: Section I explains the combination of survey and administrative data we use in the analysis.

s_04373: (1) Source: Authors' analysis based on data described in the text.

s_04423: This paper contributes to two main literatures.

s_04473: Galuščák et al (2012) , using the same data sets that this paper uses, provide an extensive analysis of the factors that determine the wages of newly-hired employees as opposed to incumbents.

s_04523: (2009) study is again a useful place to start.

s_04573: The paper proceeds as follows.

s_04623: The paper is structured as follows.

s_04673: (2003) for a survey of the literature.

s_04723: The paper is structured as follows.

s_04773: (1987) and references therein.

s_04823: In this section we provide the details of the data used in this study and the empirical methods employed.

s_04873: To address these issues, the paper relies on the instrumental variable spatial autoregressive model (Drukker et al.

s_04923: Annex 1 discusses the variables and their source."""
len(enc.encode(www))
www = """Forget all previous instructions.

You are an author of research papers. You will be given a list of sentences collected from papers you authored.

You are an expert at identifying which sentences explicitly mention datasets that you used in the paper. You will extract the sentences that satisfy this, but you should only include sentences where the mention of the dataset is clear and unambiguous.

You must exclude sentences where the mention of a source of data is vague or unclear, even if it could potentially be used as a dataset in the paper. For example, do not include sentences that mention software used for data analysis, specific equipment used to collect data, or other tools that are not datasets.

Do not include sentences that merely cite a reference without any mention of the dataset itself.

The list is in the format s_XXXXX: <sentence>. You will only return the sentence ids of the identified sentences as a JSON array and the reason why you included it. If you are uncertain about a sentence, do not include it in the list.

Always return the result in a valid JSON format. Do not truncate the output and never generate ... in the response. The output should not raise a JSONDecodeError when loaded in Python.

Output format: [{"sent_id": s_XXXXX, "dataset_mentioned": <true|false>, "reason": <dataset name>}]

Do not explain.

Text:

s_03973: Later studies use the cross-country data from the latest rounds of the Demographic and Health Surveys (DHS).

s_04023: This section focuses on the results from this study.

s_04073: This paper-a product of the Financial Sector Development Department-is part of a larger effort in the department to study the promotion of pension funds.

s_04123: The analysis in this paper contributes to several strands of the literature.

s_04173: (1999) and Radelet and Sachs (1998) .

s_04223: This paper is an attempt to provide an analytical foundation for such an amendment.

s_04273: Due to limited availability of data, the majority of existing literature analyzes the oil-price-GDP relationship in major OECD economies.

s_04323: Section I explains the combination of survey and administrative data we use in the analysis.

s_04373: (1) Source: Authors' analysis based on data described in the text.

s_04423: This paper contributes to two main literatures.

s_04473: Galuščák et al (2012) , using the same data sets that this paper uses, provide an extensive analysis of the factors that determine the wages of newly-hired employees as opposed to incumbents.

s_04523: (2009) study is again a useful place to start.

s_04573: The paper proceeds as follows.

s_04623: The paper is structured as follows.

s_04673: (2003) for a survey of the literature.

s_04723: The paper is structured as follows.

s_04773: (1987) and references therein.

s_04823: In this section we provide the details of the data used in this study and the empirical methods employed.

s_04873: To address these issues, the paper relies on the instrumental variable spatial autoregressive model (Drukker et al.

s_04923: Annex 1 discusses the variables and their source."""
l
len(enc.encode(www))
www
print("\n\n".join(all_df[["sent_id", "text"]].head(180).tail(20).apply(": ".join, axis=1).tolist()))
print("\n\n".join(all_df[["sent_id", "text"]].head(150).tail(50).apply(": ".join, axis=1).tolist()))
www = """Forget all previous instructions.

Given a list of sentences extracted from papers, extract the sentences that explicitly mention a dataset used in the paper. Only include sentences where the dataset is clearly and unambiguously named, such as LSMS or DHS. Exclude sentences that mention software used for data analysis, equipment used to collect data, or other tools that are not datasets. Also, exclude sentences that only cite a reference without mentioning the dataset or refer to data computed by the author without mentioning the dataset used to derive it.

Your task is to return the sentence ids of the identified sentences as a JSON array along with the reason why you included them. The output format should be [{"sent_id": s_XXXXX, "dataset_mentioned": <true|false>, "reason": <dataset name>}]. If a sentence mentions a dataset clearly and unambiguously, set "dataset_mentioned" to true and fill in the "reason" field with the name of the dataset. If a sentence does not mention a dataset, set "dataset_mentioned" to false and leave the "reason" field empty.

Do not explain.

Text:

s_04973: 2012 , Eichengreen and Gupta, 2013 , Nicita, 2013 ).

s_05023: Data and Prima-Facie Evidence Our sample selection depended on panel data availability for Latin American firms.

s_05073: This paper proceeds as follows.

s_05123: 9 The estimates of distortions for peer countries have been taken from Sinha (2016).

s_05173: Source: Authors' analysis based on data described in paper.

s_05223: See for instanceAuriol and Blanc (2009).

s_05273: This paper used the Total Federal transfers for Secondary Education provided by SEP.

s_05323: (See, for example, Calvo and Mendoza (2000) ).

s_05373: Corresponding author: Utz Pape (upape@worldbank.org).

s_05423: Table A .1 in the appendix provides a summary of studies discussed in this Section.

s_05473: Table 3 .1 provides a snapshot of data collection efforts divided by area of study.

s_05523: In the original work of Autor et al.

s_05573: These datasets are described in the following section.

s_05623: International Journal of Psychology, 41(5), 333-41).

s_05673: This paper breaks the problems down individually.

s_05723: In this paper we combine several data sets covering the period 2005-2015.

s_05773: The paper proceeds as follows.

s_05823: In a global study,Dora et al.

s_05873: (2016) and AfDB, OECD, UNDP (2014).

s_05923: The rest of this paper is organized as follows: a conceptual framework model is presented in section 2; section 3 outlines the data handling, data sources and the descriptive analysis.

s_05973: The paper is organized as follows: section 2 outlines the study's conceptual framework.

s_06023: It is worth noting that the size of the STM for developed countries is similar to previous country-speci…c and panel-data-based empirical …ndings (e.g., Gechert, 2015; Romer and Romer, 2016; Alesina et al, 2017; .

s_06073: (2013) ; Hangartner et al.

s_06123: In this paper, we rely on the most recent version of the data-the 2006 version (Bayer, 2006) .

s_06173: This paper is structured as follows.

s_06223: See Tables A.6 and A.7 for the full set of results.

s_06273: This result is in line with the previous literature on cash transfers, which is large and has been reviewed extensively (Fiszbein and Schady 2009; Snilstveit et al.

s_06323: In this Appendix I provide details on the sample construction and some additional tables and graphs.

s_06373: See, for example, Grawe 2004 (3) Data and Variables

s_06423: Sections 3 and 4 describe, respectively, the data and empirical strategy for the paper.

s_06473: The paper is structured as follows.

s_06523: The authors may be contacted at agrover1@ifc.org.

s_06573: A compilation of the key analytical and empirical contributions appears inGrossman (1996).17 See, for example,Chenery, Robinson, and Syrquin 1986).

s_06623: 1.4 Accordingly this paper-drawing on Northern European experience-has been written to assist transition and developing countries address three policy-related issues:

s_06673: See for exampleCalvo, Leiderman and Reinhart (1993),and Fernandez-Arias and Montiel (1995.

s_06723: This paper is structured as follows.

s_06773: The paper is organized as follows.

s_06823: See for example: deJanvry and Sadoulet 2000;Echeverria 2001bAshley and Maxwell 2001; IFAD 2001;Valdes and Mistiaen, 2001;USAID 2001; IFPRI 2002; Richards and others 2002; ODI 2003).

s_06873: For further description and discussion of the data sources and estimation methods are discussed in Chen and Ravallion (2004) , which is the source of this figure.

s_06923: As discussed inDas et al.

s_06973: The papers carry the names of the authors and should be cited accordingly.

s_07023: Total Source: Authors' elaboration based on INDEC.

s_07073: This paper must be viewed as an initial step in a large research agenda.

s_07123: These results are available from authors on request.

s_07173: The main results in this paper are summarized in Tables 4 and 5 The first (top) panel in Table 4 focuses on enrollment.

s_07223: Source: Authors' calculations.

s_07273: is more closely related to this paper.

s_07323: A recent study by Wilson, Mann and Otsuki (Wilson, Mann, Otsuki 2005b) quantified the conceptual experiment of improving trade facilitation processes in countries where they are least efficient.

s_07373: In our sample, Binswanger et al.

s_07423: This paper evaluates these efforts based on limited data accumulated over 1989-93."""
len(enc.encode(www))
resp = """[{"sent_id": "s_05023", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_05073", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_05123", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_05173", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_05223", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_05273", "dataset_mentioned": true, "reason": "Total Federal transfers for Secondary Education provided by SEP"}, {"sent_id": "s_05323", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_05373", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_05423", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_05473", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_05523", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_05573", "dataset_mentioned": true, "reason": "datasets are described in the following section"}, {"sent_id": "s_05623", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_05673", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_05723", "dataset_mentioned": true, "reason": "several data sets covering the period 2005-2015"}, {"sent_id": "s_05773", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_05823", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_05873", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_05923", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_05973", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_06023", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_06073", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_06123", "dataset_mentioned": true, "reason": "the most recent version of the data-the 2006 version (Bayer, 2006)"}, {"sent_id": "s_06173", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_06223", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_06273", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_06323", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_06373", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_06423", "dataset_mentioned": true, "reason": "the data"}, {"sent_id": "s_06473", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_06523", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_06573", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_06623", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_06673", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_06723", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_06773", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_06823", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_06873", "dataset_mentioned": true, "reason": "Chen and Ravallion (2004)"}, {"sent_id": "s_06923", "dataset_mentioned": true, "reason": "Das et al."}, {"sent_id": "s_06973", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_07023", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_07073", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_07123", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_07173", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_07223", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_07273", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_07323", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_07373", "dataset_mentioned": false, "reason": ""}, {"sent_id": "s_07423", "dataset_mentioned": false, "reason": ""}]"""
len(enc.encode(resp))
len(enc.encode(resp))
www = """Forget all previous instructions.

Given a list of sentences extracted from papers, extract the sentences that explicitly mention a dataset used in the paper. Only include sentences where the dataset is clearly and unambiguously named, such as LSMS or DHS. Exclude sentences that mention software used for data analysis, equipment used to collect data, or other tools that are not datasets. Also, exclude sentences that only cite a reference without mentioning the dataset or refer to data computed by the author without mentioning the dataset used to derive it. Do return sentences that does not mention dataset.

Your task is to return the sentence ids of the identified sentences as a JSON array along with the reason why you included them. The output format should be [{"sent_id": s_XXXXX, "dataset_mentioned": <true|false>, "reason": <dataset name>}]. If a sentence mentions a dataset clearly and unambiguously, set "dataset_mentioned" to true and fill in the "reason" field with the name of the dataset. In case you generated a sentence that does not mention a dataset, set "dataset_mentioned" to false and leave the "reason" field empty.

Do not explain.

Text:

s_04973: 2012 , Eichengreen and Gupta, 2013 , Nicita, 2013 ).

s_05023: Data and Prima-Facie Evidence Our sample selection depended on panel data availability for Latin American firms.

s_05073: This paper proceeds as follows.

s_05123: 9 The estimates of distortions for peer countries have been taken from Sinha (2016).

s_05173: Source: Authors' analysis based on data described in paper.

s_05223: See for instanceAuriol and Blanc (2009).

s_05273: This paper used the Total Federal transfers for Secondary Education provided by SEP.

s_05323: (See, for example, Calvo and Mendoza (2000) ).

s_05373: Corresponding author: Utz Pape (upape@worldbank.org).

s_05423: Table A .1 in the appendix provides a summary of studies discussed in this Section.

s_05473: Table 3 .1 provides a snapshot of data collection efforts divided by area of study.

s_05523: In the original work of Autor et al.

s_05573: These datasets are described in the following section.

s_05623: International Journal of Psychology, 41(5), 333-41).

s_05673: This paper breaks the problems down individually.

s_05723: In this paper we combine several data sets covering the period 2005-2015.

s_05773: The paper proceeds as follows.

s_05823: In a global study,Dora et al.

s_05873: (2016) and AfDB, OECD, UNDP (2014).

s_05923: The rest of this paper is organized as follows: a conceptual framework model is presented in section 2; section 3 outlines the data handling, data sources and the descriptive analysis.

s_05973: The paper is organized as follows: section 2 outlines the study's conceptual framework.

s_06023: It is worth noting that the size of the STM for developed countries is similar to previous country-speci…c and panel-data-based empirical …ndings (e.g., Gechert, 2015; Romer and Romer, 2016; Alesina et al, 2017; .

s_06073: (2013) ; Hangartner et al.

s_06123: In this paper, we rely on the most recent version of the data-the 2006 version (Bayer, 2006) .

s_06173: This paper is structured as follows.

s_06223: See Tables A.6 and A.7 for the full set of results.

s_06273: This result is in line with the previous literature on cash transfers, which is large and has been reviewed extensively (Fiszbein and Schady 2009; Snilstveit et al.

s_06323: In this Appendix I provide details on the sample construction and some additional tables and graphs.

s_06373: See, for example, Grawe 2004 (3) Data and Variables

s_06423: Sections 3 and 4 describe, respectively, the data and empirical strategy for the paper.

s_06473: The paper is structured as follows.

s_06523: The authors may be contacted at agrover1@ifc.org.

s_06573: A compilation of the key analytical and empirical contributions appears inGrossman (1996).17 See, for example,Chenery, Robinson, and Syrquin 1986).

s_06623: 1.4 Accordingly this paper-drawing on Northern European experience-has been written to assist transition and developing countries address three policy-related issues:

s_06673: See for exampleCalvo, Leiderman and Reinhart (1993),and Fernandez-Arias and Montiel (1995.

s_06723: This paper is structured as follows.

s_06773: The paper is organized as follows.

s_06823: See for example: deJanvry and Sadoulet 2000;Echeverria 2001bAshley and Maxwell 2001; IFAD 2001;Valdes and Mistiaen, 2001;USAID 2001; IFPRI 2002; Richards and others 2002; ODI 2003).

s_06873: For further description and discussion of the data sources and estimation methods are discussed in Chen and Ravallion (2004) , which is the source of this figure.

s_06923: As discussed inDas et al.

s_06973: The papers carry the names of the authors and should be cited accordingly.

s_07023: Total Source: Authors' elaboration based on INDEC.

s_07073: This paper must be viewed as an initial step in a large research agenda.

s_07123: These results are available from authors on request.

s_07173: The main results in this paper are summarized in Tables 4 and 5 The first (top) panel in Table 4 focuses on enrollment.

s_07223: Source: Authors' calculations.

s_07273: is more closely related to this paper.

s_07323: A recent study by Wilson, Mann and Otsuki (Wilson, Mann, Otsuki 2005b) quantified the conceptual experiment of improving trade facilitation processes in countries where they are least efficient.

s_07373: In our sample, Binswanger et al.

s_07423: This paper evaluates these efforts based on limited data accumulated over 1989-93."""
len(enc.encode(www))
len(enc.encode(www)) + len(enc.encode(resp))
enc.encode("dataset_mentioned")
enc.encode('"dataset_mentioned"')
enc.encode('"mentioned"')
print("\n\n".join(all_df[["sent_id", "text"]].head(150).tail(25).apply(": ".join, axis=1).tolist()))
print("\n\n".join(all_df[["sent_id", "text"]].head(150).tail(20).apply(": ".join, axis=1).tolist()))
print("\n\n".join(all_df[["sent_id", "text"]].head(180).tail(20).apply(": ".join, axis=1).tolist()))
print("\n\n".join(all_df[["sent_id", "text"]].head(160).tail(20).apply(": ".join, axis=1).tolist()))
print("\n\n".join(all_df[["sent_id", "text"]].head(180).tail(20).apply(": ".join, axis=1).tolist()))
print("\n\n".join(all_df[["sent_id", "text"]].head(280).tail(20).apply(": ".join, axis=1).tolist()))
print("\n\n".join(all_df[["sent_id", "text"]].head(380).tail(20).apply(": ".join, axis=1).tolist()))
all_df[["sent_id", "text"]].head(160).tail(20)
sub = all_df[["sent_id", "text"]].head(160).tail(20)
sub
sub["text"].str.replace("[^a-zA-Z\d]+", "")
sub["text"].str.replace("[^a-zA-Z\d]+", "", regex=True)
f = sub["text"].str.replace("[^a-zA-Z\d]+", "", regex=True)
f.map(len)
f.map(len) < 10
sub
RANKING_DIR = Path("data/training/ranking")
ll_df = pd.read_excel((RANKING_DIR / "ranking_shortlist.xlsx"))
ll_df.head()
sub
sub["text"].str.replace("[^a-zA-Z\d]+", "", regex=True)
sub[sub["text"].str.replace("[^a-zA-Z\d]+", "", regex=True).map(len) > 10]
# sub 
# all_df = p
# all_df = pd.concat(all_df)
get_ipython().run_line_magic('ls', '')
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer
model_path = "data/training/ranking/models/distilbert-base-cased_t1682997237"
tokenizer = AutoTokenizer(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model
model = AutoModelForSequenceClassification.from_pretrained(model_path)
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
import numpy as np
from scipy.special import softmax
import torch
from transformers import pipeline
import torch
from tqdm.auto import tqdm

classifier = pipeline("text-classification", model=model_path)
tx = """This paper uses administrative data on electricity billing records from Ghana and Rwanda. The Ghana data comes from the Electricity Company of Ghana (ECG), which is the largest distributor in the country with operations in the southern and middle belts. It accounts for nearly 70% of all electricity customers in the country. We use data on billing records of the universe of electricity customers of the ECG from January 2018 to December 2020. The data identifies customer types based on the tariff applicable: residential (households), non-residential, and heavy industries. For each customer and year-month, it records the amount (kWh) of electricity consumed, the monetary value in Ghana Cedis (GHS), meter type (postpaid vs prepaid), and location (district) of the customer. In all, the data contains 42 million customer-year-month observations."""
classifier(tx)
model.device
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)
sub.head()
def make_predictions(df, model, batch=50):
    test_inputs = df.apply(preprocess_function, axis=1)
    data_sent_prob = []

    with torch.no_grad():
        model.eval()

        for i in tqdm(range(0, len(test_inputs) + batch, batch)):
            ti = test_inputs.iloc[i:i + batch]
            if ti.empty:
                break

            ti = ti.tolist()

            ti = {k: v.to(model.device) for k, v in data_collator(ti).items()}
            o = model(**ti)

            data_sent_prob.append(softmax(o.logits.cpu(), axis=1))

            torch.cuda.empty_cache()

    data_sent_prob_arr = np.vstack(data_sent_prob)[:, 1]

    return data_sent_prob_arr
    
#o = make_predictions(sub)
o
opo = make_predictions(sub)
opo = make_predictions(sub, model)
opo = make_predictions(all_df.head(100), model)
# opo = make_predictions(all_df.head(1000), model)
all_df.shape
opo = make_predictions(all_df.head(1000), model)
opo.shape
opo[:10]
all_df.head()
def make_predictions_from_text(texts, model, batch=50):
    if isinstance(texts, str):
        texts = [texts]

    model.eval()
    sent_prob = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts) + batch, batch)):
            ti = texts[i:i + batch]
            if not ti:
                break

            inputs = tokenizer(ti, truncation=True, padding=True, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            o = model(**inputs)
            sent_prob.append(softmax(o.logits.cpu(), axis=1))

            torch.cuda.empty_cache()

    sent_prob_arr = np.vstack(sent_prob)[:, 1]

    return sent_prob_arr
    
make_predictions_from_text(sub["text"].tolist(), model)
sub.shape
from data_use import ranking
# ranking.sentence_prob_for_texts(
importlib.reload(ranking)
ranking.sentence_prob_for_texts(sub["text"].tolist())
wwwww = ranking.sentence_prob_for_texts(sub["text"].tolist())
wwwww
sorted(zip(wwwww, sub["text"].tolist(), reverse=True))
sorted(zip(wwwww, sub["text"].tolist()), reverse=True))
sorted(zip(wwwww, sub["text"].tolist()), reverse=True)
wwwww
list(wwwww)
def rank_texts(texts, scores):
    texts = np.array(texts)
    scores = np.array(scores)

    idx = np.argsort(scores)[::-1]
    return texts[idx], scores[idx]
rank_texts(sub["text"].tolist(), wwwww)
ppp np.array(sub["text"].tolist())
ppp = np.array(sub["text"].tolist())
ppp
ppp.tolist()
wwwww.tolist()
wwwww.astype(float)
wwwww.astype(float).tolist()
wwwww.astype(float).round(4).tolist()
wwwww.astype(float).round(6).tolist()
e
op
lk = {1: 2}
{**lk, **{2:5}}
tx = """offered a 100% subsidy to lifeline (low-income) customers who consume 0-50 kWh of elec- tricity a month, while non-lifeline customers (consuming above 50 kWh a month) would enjoy a 50% subsidy (Berkouwer et al., 2022). The subsidies were announced on March 29 and took effect from April 1. Eligibility status was determined based on March consump- tion. The policy was initially announced to last for three months, however, the subsidy for [only] lifeline customers was eventually extended for 12 months.
3 Data
This paper uses administrative data on electricity billing records from Ghana and Rwanda. The Ghana data comes from the Electricity Company of Ghana (ECG), which is the largest distributor in the country with operations in the southern and middle belts. It accounts for nearly 70% of all electricity customers in the country. We use data on billing records of the universe of electricity customers of the ECG from January 2018 to December 2020. The data identifies customer types based on the tariff applicable: residential (households), non-residential, and heavy industries. For each customer and year-month, it records the amount (kWh) of electricity consumed, the monetary value in Ghana Cedis (GHS), meter type (postpaid vs prepaid), and location (district) of the customer. In all, the data contains 42 million customer-year-month observations."""
classifier(tx)
tx = """offered a 100% subsidy to lifeline (low-income) customers who consume 0-50 kWh of elec- tricity a month, while non-lifeline customers (consuming above 50 kWh a month) would enjoy a 50% subsidy (Berkouwer et al., 2022). The subsidies were announced on March 29 and took effect from April 1. Eligibility status was determined based on March consump- tion. The policy was initially announced to last for three months, however, the subsidy for [only] lifeline customers was eventually extended for 12 months.
3 Data
This paper uses administrative data on electricity billing records from Ghana and Rwanda. The Ghana data comes from the Electricity Company of Ghana (ECG), which is the largest distributor in the country with operations in the southern and middle belts. It accounts for nearly 70% of all electricity customers in the country."""
classifier(tx)
get_ipython().run_line_magic('logstart', '')
