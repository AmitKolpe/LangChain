from langchain_community.document_loaders import PyPDFLoader

loader  = PyPDFLoader("C:\\Users\\amitk\\Downloads\\amit_seminar report.pdf")
docs = loader.load()
print(len(docs))
print(docs[0].page_content)
print(docs[1].metadata)