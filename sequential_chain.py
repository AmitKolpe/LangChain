from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from  langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
     repo_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
    task="text-generation"

)

model = ChatHuggingFace(llm = llm)


prompt1 = PromptTemplate(
    template='Generate the detail report on \n {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='generate the 5 pointer summary from the following text \n {text}',
    input_variables=['text']

)

parser = StrOutputParser()

chain  = prompt1|model|parser|prompt2|model|parser

result =    chain.invoke({'topic':'cricket'})

print(result)
chain.get_graph().print_ascii