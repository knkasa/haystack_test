from haystack.nodes import BM25Retriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.document_stores import FAISSDocumentStore


# Step 1: Set up a document store and write documents
from haystack.document_stores import InMemoryDocumentStore
document_store = InMemoryDocumentStore()
document_store.write_documents([
    {"content": "Python is a programming language."},
    {"content": "Haystack helps build search systems."}
	])

#Faiss RAG
document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")
document_store.save("my_faiss_index")
# Later load it
document_store = FAISSDocumentStore.load("my_faiss_index")

#Pinecone RAG
from haystack.document_stores import PineconeDocumentStore
document_store = PineconeDocumentStore(
    api_key="your-pinecone-api-key",
    environment="us-west1-gcp"
	)

#SQL Document store RAG
from haystack.document_stores import SQLDocumentStore
document_store = SQLDocumentStore(url="sqlite:///my_db.db")

# AWS ElasticSearch
from haystack.document_stores import ElasticsearchDocumentStore
document_store = ElasticsearchDocumentStore(
    host="your-opensearch-domain.region.es.amazonaws.com",
    port=443,
    username="your-username",  # if basic auth enabled
    password="your-password",
    scheme="https",
    verify_certs=True,
    index="your-index",
    embedding_dim=768,       # Required if using vector search
    similarity="cosine"      # or "dot_product"
	)


# Step 2: Set up retriever and reader
retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

# Step 3: Build QA pipeline
pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever)

# Step 4: Ask a question
result = pipeline.run(query="What is Haystack?", params={"Retriever": {"top_k": 2}})
print(result["answers"][0].answer)
