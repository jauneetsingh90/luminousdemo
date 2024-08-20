from langchain.docstore.document import Document
from langchain.utilities import ApifyWrapper
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import AstraDB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import ApifyDatasetLoader
import os
import csv
import pandas as pd
from langchain.document_loaders import DataFrameLoader

token="ASTRADB_ENDPOINT_TOKEN"
api_endpoint="ASTRA_DB_ENDPOINT"
openai_api_key='OPENAI_API_KEY'
#apify_api_key=os.environ["APIFY_API_TOKEN"]

vstore = AstraDB(
    embedding=OpenAIEmbeddings(),
    collection_name="wakefit",
    api_endpoint=api_endpoint,
    token=token,
)

filename = '/Users/jauneet.singh/Downloads/wakefitdemo.csv'
df = pd.read_csv(filename)
llmtexts = []
start = 1
batch_size = 1000
docs = []
for i in range(start, start+batch_size, batch_size):
    print(f"Processing {i} to {i+batch_size} llm texts")
    batch = df[i:i+batch_size]
    batch = batch.fillna('')
    for id, row in batch.iterrows():
        rawtext = f"productid: {row['productid']} category_name: {row['category_name']} sub_category_name: {row['sub_category_name']} eshop_category_name: {row['eshop_category_name']} product_name: {row['product_name']} description : {row['description']}  link: {row['link']} price: {row['price']} product_specification: {row['product_specification']} usuage: {row['usuage']} link: {row['link']} "
        #rawtext = f"product: {row['product']} size: {row['size']} type: {row['type']} price: {row['price']} description : {row['description']}  offer: {row['offer']} goes_good: {row['goes_good']}  "
        print(row['product'])
        #translated_text = translate_lang(rawtext)
        llmtexts.append(rawtext)
        doc = Document(page_content=rawtext, metadata=row.to_dict())
        docs.append(doc)
        #print(llmtexts)
    batch['llmtext'] = llmtexts

    #batch.to_csv(f"llm_product_{start}.csv", encoding='utf-8', index=False)

inserted_ids = vstore.add_documents(docs)
print(f"\nInserted {len(inserted_ids)} documents.")
