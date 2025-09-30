from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

document = [
    Document(page_content="langchain hepls developers build llm application easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models.")

]

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma.from_documents(
    documents=document,
    embedding=embedding_model,
    collection_name='my_collection'
)

retreiver = vector_store.as_retriever(search_kwargs={'k':2})
query = "what is chroma used for?"
result = retreiver.invoke(query)
for i, doc in enumerate(result):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)