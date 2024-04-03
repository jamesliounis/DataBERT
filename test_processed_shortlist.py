# IPython log file

get_ipython().run_line_magic('logstart', 'test_processed_shortlist.py')
get_ipython().run_line_magic('ls', '')
import pandas as pd
import json
from pathlib import Path
datas = []
for shp in Path("data/output/").glob("*.json"):
    d = json.loads(shp.read_text())
    content = d["content"]
    try:
        datas.append(json.loads(content))
    except json.decoder.JSONDecodeError:
        print("Failed to parse content for:", shp)
        
data = {}
datas = {}
for shp in Path("data/output/").glob("*.json"):
    d = json.loads(shp.read_text())
    content = d["content"]
    try:
        datas[d["metadata"]["doc_id"]] = json.loads(content)
    except json.decoder.JSONDecodeError:
        print("Failed to parse content for:", shp)
        
datas = {}
misparsed = {}
for shp in Path("data/output/").glob("*.json"):
    d = json.loads(shp.read_text())
    content = d["content"]
    try:
        datas[d["metadata"]["doc_id"]] = json.loads(content)
    except json.decoder.JSONDecodeError:
        misparsed[d["metadata"]["doc_id"]] = content
        print("Failed to parse content for:", shp)
        
misparsed["D34009912"]
misparsed["D34039432"]
misparsed["D34039432"].rindex("]}")
end = misparsed["D34039432"].rindex("]}") + 2
start = misparsed["D34039432"].rindex('{"')
misparsed["D34039432"][start:end]
json.loads(misparsed["D34039432"][start:end])
start = misparsed["D34039432"].index('{"')
json.loads(misparsed["D34039432"][start:end])
def parse_misparsed(text: str):
    start = text.index('{"')
    end = text.index("]}") + 2

    return json.loads(text[start:end])
    
datas = {}
misparsed = {}
for shp in Path("data/output/").glob("*.json"):
    d = json.loads(shp.read_text())
    content = d["content"]
    try:
        datas[d["metadata"]["doc_id"]] = json.loads(content)
    except json.decoder.JSONDecodeError:
        try:
            datas[d["metadata"]["doc_id"]] = parse_misparsed(content)
        except json.decoder.JSONDecodeError:
            misparsed[d["metadata"]["doc_id"]] = content
            print("Failed to parse content for:", shp)
            
misparsed["D34009912"]
def parse_misparsed(text: str):
    start = text.index('{"')
    end = text.rindex("]}") + 2

    return json.loads(text[start:end])
    
datas = {}
misparsed = {}
for shp in Path("data/output/").glob("*.json"):
    d = json.loads(shp.read_text())
    content = d["content"]
    try:
        datas[d["metadata"]["doc_id"]] = json.loads(content)
    except json.decoder.JSONDecodeError:
        try:
            datas[d["metadata"]["doc_id"]] = parse_misparsed(content)
        except json.decoder.JSONDecodeError:
            misparsed[d["metadata"]["doc_id"]] = content
            print("Failed to parse content for:", shp)
            
misparsed["D34009866"]
misparsed["D34024438"]
"..." in misparsed["D34024438"]
keys = datas.keys()
datas[keys[0]]
keys = list(datas.keys())
datas[keys[0]]
datas[keys[1]]
datas[keys[2]]
datas[keys[3]]
datas[keys[4]]
datas[keys[5]]
datas[keys[6]]
datas[keys[7]]
datas[keys[8]]
datas[keys[9]]
datas[keys[20]]
datas[keys[20]]
datas[keys[10]]
datas[keys[11]]
datas[keys[12]]
datas[keys[13]]
datas[keys[14]]
datas[keys[15]]
datas[keys[16]]
datas[keys[17]]
datas[keys[18]]
import jq
import pyjq
import jq
from data_use.document_loaders import pdf, structured_json as sj
exi
exit()
