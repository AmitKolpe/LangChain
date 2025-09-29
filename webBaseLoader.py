from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)

model = ChatHuggingFace(llm  = llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Answer the following question {question} from the following text - \n {text}",
    input_variables=['question','text']
)

url = " https://www.flipkart.com/google-pixel-10-pro-moonstone-256-gb/p/itm1a0b92194d0c1?pid=MOBHEXHRFA2Z5TD7&lid=LSTMOBHEXHRFA2Z5TD7BTXF7R&marketplace=FLIPKART&q=pixel+10+pro&store=tyy%2F4io&srno=s_1_1&otracker=AS_Query_OrganicAutoSuggest_7_5_na_na_na&otracker1=AS_Query_OrganicAutoSuggest_7_5_na_na_na&fm=organic&iid=en_3hS7E9ZwiSN6LHL-R8EYfdNPI50xfVsBZ1y7WYBSq-yTqATMM5d8hTCGlXv0mvH4QzKK47vbjPQB1frlJM7ZqfUFjCTyOHoHZs-Z5_PS_w0%3D&ppt=None&ppn=None&ssid=2e9f8r75n40000001759150202413&qH=1f3f1431a2946f49"

loader = WebBaseLoader(url)
docs  = loader.load()
chain = prompt|model|parser
print(chain.invoke({'question':'what is the price of model?','text':docs[0].page_content}))