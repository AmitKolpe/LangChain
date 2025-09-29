from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

# Folder path where your PDFs are stored
folder_path = r"C:\Users\amitk\Downloads\MyPDFs"  

# Use DirectoryLoader to load all PDFs
loader = DirectoryLoader(
    path=folder_path,
    glob="*.pdf",           # load only PDF files
    loader_cls=PyPDFLoader   # specify loader class for each file
)

docs = loader.load()

print(docs)                 
print(docs[0].page_content)    # text content of first PDF
print(docs[0].metadata)        
