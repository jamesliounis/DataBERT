# IPython log file

get_ipython().run_line_magic('logstart', 'ipython_logs/gpt_generated_sentence_analysis.py')
import json
from pathlib import Path
GPT_SENT_DIR = Path("data/openai/generated_sentences")
sent_files = sorted(GPT_SENT_DIR.glob("*.json"))
len(sent_files)
j = json.loads(sent_files[0].read_text())
jj = json.loads(j["content"])
jj[0]
from collections import Counter
topics = []
for sf in sent_files:
    j = json.loads(sf.read_text())
    jj = json.loads(j["content"])
    topics.extend(jj.get("topic", []))
    
for sf in sent_files:
    j = json.loads(sf.read_text())
    jj = json.loads(j["content"])
    for s in jj:
        topics.extend(s.get("topic", []))
        
len(topics)
Counter(topics).most_commont()
Counter(topics).most_common()
len(Counter(topics).most_common())
sent_files = sorted(GPT_SENT_DIR.glob("*.json"))
topics = []
for sf in sent_files:
    j = json.loads(sf.read_text())
    jj = json.loads(j["content"])
    for s in jj:
        topics.extend(s.get("topic", []))
        
Counter(topics).most_common()
len(topics)
len(sent_files)
sent_files = sorted(GPT_SENT_DIR.glob("*.json"))
topics = []
for sf in sent_files:
    j = json.loads(sf.read_text())
    jj = json.loads(j["content"])
    for s in jj:
        topics.extend(s.get("topic", []))
        
j["content"]
json.loads(j["content"][:-1])
from data_use import parser
topics = []
for sf in sent_files:
    j = json.loads(sf.read_text())
    jj = parser.parse_misparsed(j["content"])
    for s in jj:
        topics.extend(s.get("topic", []))
        
json.loads(j["content"][:-1])
j["content"]
import importlib
importlib.reload(parser)
topics = []
for sf in sent_files:
    j = json.loads(sf.read_text())
    jj = parser.parse_misparsed(j["content"], open='[{"', close=']}]')
    for s in jj:
        topics.extend(s.get("topic", []))
        
j["content"]
json.loads(j["content"])
importlib.reload(parser)
topics = []
for sf in sent_files:
    j = json.loads(sf.read_text())
    jj = parser.parse_misparsed(j["content"], open='[{"', close=']}]')
    for s in jj:
        topics.extend(s.get("topic", []))
        
json.loads(j["content"])
j["content"]
print(j["content"])
for sf in sent_files:
    j = json.loads(sf.read_text())
    try:
        jj = parser.parse_misparsed(j["content"], open='[{"', close=']}]')
    except:
        print(sf)
        continue
    for s in jj:
        topics.extend(s.get("topic", []))
        
topics = []
for sf in sent_files:
    j = json.loads(sf.read_text())
    try:
        jj = parser.parse_misparsed(j["content"], open='[{"', close=']}]')
    except:
        print(sf)
        continue
    for s in jj:
        topics.extend(s.get("topic", []))
        
Counter(topics).most_common()
Counter(topics).most_common(32)
Counter(topics).most_common(35)
Counter(topics).most_common(31)
Counter(topics).most_common(32)
len(sf)
len(sent_files)
# sent_files = sorted(GPT_SENT_DIR.glob("*.json"))
# GPT_SENT_DIR = Path("data/openai/generated_sentences")
jj
sents = [j["sentence"] for j in jj]
sents
