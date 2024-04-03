# IPython log file

get_ipython().run_line_magic('logstart', 'ipython_logs/experiment_process_shortlists.py')
from pathlib import Path
import json
d = json.loads(Path("data/prwp/shortlist/D698970.shortlist.json").read_text())
texts = [i["text"] for i in d["shortlist"] if i["prob"] > 0.05][:10]
len(texts)
print("\n\n".texts)
print("\n\n".join(texts))
texts = [i["text"] for i in d["shortlist"] if i["prob"] > 0.05]
len(texts)
print("\n\n".join(texts))
d = json.loads(Path("data/prwp/shortlist/D700055.shortlist.json").read_text())
texts = [i["text"] for i in d["shortlist"] if i["prob"] > 0.05]; print(len(texts))
print("\n\n".join(texts))
import tiktoken
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
len(enc.encode("\n\n".join(texts)))
print("\n\n".join(texts[:20]))
dtypes = """Surveys
Census data
Administrative data
Remote sensing data
Geographic Information System (GIS) data
Financial and economic data
Climate data
Health data
Education data
Social media data
Mobile phone data
Transactional data
Big data
Qualitative data
Experimental data
Case studies
Observational data
Longitudinal data
Cross-sectional data
Time series data"""
"!!!!!".join(dtypes.split("\n"))
o = json.loads(Path("data/openai/processed_shortlist/f033be6c-74b1-0638-99e0-3aa7f35f189a/gpt-struct_shorts_t1683054080_chatcmpl-7BpXUnZ4WwHbCqlFsahWhdjl366rD.json").read_text())
print(o["content"])
print(json.loads(o["content"]))
print("""Forget all previous instructions.\n\nYou are given a list of sentences extracted from a paper quoted with triple backticks below.\n\nYour task is to identify the datasets and the data type used in the paper, for example LSMS, DHS, etc. You must only consider sentences that explicitly mention a name of a dataset. You are to return the result in JSON array format with keys \"data_name\", \"data_type\",  \"is_quantitative\", \"sentence\".\n\nChoose the data_type from this !!!!!-delimited list: !!!!!Surveys!!!!!Census data!!!!!Administrative data!!!!!Remote sensing data!!!!!Geographic Information System (GIS) data!!!!!Financial and economic data!!!!!Climate data!!!!!Health data!!!!!Education data!!!!!Social media data!!!!!Mobile phone data!!!!!Transactional data!!!!!Big data!!!!!Qualitative data!!!!!Experimental data!!!!!Case studies!!!!!Observational data!!!!!Longitudinal data!!!!!Cross-sectional data!!!!!Time series data!!!!!Other\n\n```Data on External Conflict also comes from the ICRG dataset.\n\nArzaghi and Henderson (2005) examine the determinants of fiscal decentralization using panel data on 48 countries over the period 1960 to 1995.\n\nThese come from the International Country Risk Guide (ICRG) published by the PRS Group.\n\nWe had to reduce our sample to nine countries in the regressions with central government expenditures due to lack of government expenditure data for Djibouti, Lebanon, Libya, Saudi Arabia and Yemen.\n\nWe use Share of Urban Population to find the effect of population concentrations.\n\nTheir evidence on MENA is based on only few countries in the region due to problems with available data on fiscal decentralization.\n\nWe have realized however that data on Panizza also compares his results to two other cross-sectional or panel studies by Oates (1972) and Wallis and Oates (1988) and finds similarities particularly in regards to country size and income per capita.\n\n14 Finally, we use GCC as a dummy variable for the Gulf Cooperation Council (GCC) countries to control for the possible impact of heavy dependence on oil on the government structure of these countries.\n\nEbel and Yilmaz (2003) note that this is mainly due to data imperfections, particularly in the IMF's Government Finance Statistics (GFS).\n\nWhile recent studies followed a comparative perspective and showed similarities and differences between the decentralization efforts in a variety of developing countries, the MENA countries are left out of those comparisons ( Bardhan and Mookherjee, 2006) .\n\nAs so many other studies did before us, we started with data from the GFS.\n\nTable 3 shows we assigned points to central, provincial and municipal government involvement in expenditures.\n\nMuni, State and Author are the names used in DPI under the sub-heading \"Federalism.\"\n\nComponents of these institutional variables, particularly corruption in government, were used in many other studies includingTanzi and Davoodi (2000),Mauro (1996) andKnack and Keefer (1995).\n\nOur choice of explanatory variables is based mainly on the theoretical and empirical analyses by Oates (1972) , Wallis and Oates (1988) , Panizza (1999) , and Arzaghi and Henderson (2005) .\n\nThis is mainly due to lack of appropriate data and information on decentralization in the countries of the region.\n\nTable 8 Results from those regressions are shown in Table 9 .\n\nOates (1972) , Panizza (1999) and Arzaghi and Henderson (2005) all found that these are negatively correlated with centralization.\n\nTotal expenditure assignment scores and average scores are listed in Table 4 .\n\nNote that higher risk points correspond to an improvement in the institutional variable.13 This was recently used as a measure of governance quality byKnack (2001).\n\nArzaghi and Henderson follow Panizza's approach to model decentralization first and then empirically test hypotheses derived from the theoretical model.\n\n(2001) andKeefer (2007) for detailed descriptions of these variables.\n\nIn Section 4 we explore the determinants of centralization and decentralization in the region in a regression analysis.\n\nHence an empirical investigation of decentralization in the MENA region is difficult due to aforementioned data problems.\n\nWe see in Table 4 that this is particularly true in social services, which is one of the most important government expenditures that directly affect the welfare of residents.```""")
# o
pp = json.loads(Path("data/prwp/shortlist-chunks/D33571363.shortlist.json").read_text())
print("\n\n".join([p["text"] for p in pp[:4]]))
print("\n\n".join([p["text"] for p in pp["shortlist"][:4]]))
p = "\n\n".join([p["text"] for p in pp["shortlist"][:4]])
len(enc.encode(p))
o
o["metadata"]
pp = json.loads(Path("data/prwp/shortlist-chunks/D10008558.shortlist.json").read_text())
pp = json.loads(Path("data/prwp/shortlist-chunks/D30651770.shortlist.json").read_text())
p = "\n\n".join([p["text"] for p in pp["shortlist"][:4]])
len(enc.encode(p))
print(p)
pp = json.loads(Path("data/prwp/shortlist-chunks/D10008558.shortlist.json").read_text())
pp = json.loads(Path("data/prwp/shortlist-chunks/D7128835.shortlist.json").read_text())
p = "\n\n".join([p["text"] for p in pp["shortlist"][:4]])
print(p)
pp = json.loads(Path("data/prwp/shortlist-chunks/D10008558.shortlist.json").read_text())
p = "\n\n".join([p["text"] for p in pp["shortlist"][:4]])
print(p)
p = "\n\n".join([p["text"].replace("\n\n", " ") for p in pp["shortlist"][:4]])
print(p)
pp = json.loads(Path("data/prwp/shortlist-chunks/D32028814.shortlist.json").read_text())
p = "\n\n".join([p["text"].replace("\n\n", " ") for p in pp["shortlist"][:4]])
print(p)
p = "\n\n".join([p["text"] for p in pp["shortlist"][:4]])
print(p)
import spacy
import spacy
nlp = spacy.load("en_corel_web_sm")
nlp = spacy.load("en_corel_web_sm")
nlp = spacy.load("en_core_web_sm")
sent = nlp("Interestingly enough, Lopez (1990) finds a similar pattern in macroeconomic data, using different measures for policies affecting imports and exports: export incentives positively affect overall growth, while import restrictions have an insignificant effect.")
sent.ents
for word in sent:
    print(word, word.dep_)
    
s = nlp("As the push for educational reform has increased, so has the perceived importance of large-scale assessments (ETS, 1994: OTA. 1992). Large-scale assessments are used at the national level, the state level, and sometimes the local level. The Office of Technology Assessment (OTA. 1992) described three purposes of assessment: to aid teachers and students in the conduct of classroom learning. to monitor systemwide educational outcomes. to make informed decisions about the selection, placement. and credentialling of individual students. (p. 8) Usually, large-scale assessments focus on the latter two of these purposes. The Policy Information Center at ETS (1994), using data from the State Student Assessment Database (see Bond. 1994). organized the primary functions of state assessment programs into five purposes: Accountability. Instructional Improvement. Program Evaluation.")
s.sents
list(s.sents)
for i in s.sents:
    print(i)
    print()
    
from nltk.tokenize import sent_tokenize
sent_tokenize(s.text)
for i in sent_tokenize(s.text)
for i in sent_tokenize(s.text):
    print(i)
    print()
    
