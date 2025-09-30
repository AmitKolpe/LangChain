from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2, lang="en")

query = "the geopolitical history of India and Pakistan from the perspective of the Chinese"
docs = retriever.invoke(query)

if not docs:
    print("No results found. Try changing the query or installing `wikipedia` package.")
else:
    for i, doc in enumerate(docs, start=1):
        print(f"\n----- Result {i} -----")
        print(f"Content: {doc.page_content[:500]}...")
