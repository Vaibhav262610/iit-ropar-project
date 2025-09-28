put all the pdf and other documents etc unparsed in the documents directory obvio
checkout the example .json file put things in similar fashion

I havent added support for hit rate calculation yet but it wont be that difficult
vector store server provides the file name so we can directly compare that and sources present in query_context_answer

question and answer is obvio

supporting_facts is the thing that should be retrieved to get the right answer basically this is human annotated

## 12 nov 2024 (kitu)

1. Download tatdqa documents and dataset from -> https://drive.google.com/drive/folders/1SGpZyRWqycMd_dZim1ygvWhl5KdJYDR2
2. I have used dev documents -> tatdqa_dataset_dev.json has question/answer for dtatdqa_docs_dev only
3. Extract dev docs and remove all files other then pdf (using some command)
   `find . -type f ! -iname "*.pdf" -exec rm -v {} +` this worked for me
4. Update server.py to use this dir.
5. For updating dataset processing for ragas -> refer kitu/
