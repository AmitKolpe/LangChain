from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


docs = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models.")
]


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


vectorstore = FAISS.from_documents(
    documents=docs,          
    embedding=embedding_model
)

# Create MMR retriever
retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 3, 'lambda_mult': 1}
)

query = "what is langchain?"
results = retriever.get_relevant_documents(query)


for i, doc in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print(doc.page_content)
