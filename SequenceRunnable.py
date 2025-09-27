from langchain_huggingface import ChatHuggingFace ,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence


load_dotenv()

llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        task="text-generation"
)
 
model = ChatHuggingFace(llm = llm)



prompt = PromptTemplate(
    template='write a joke on {topic}/n',
    input_variables=['topic']

)
parser = StrOutputParser()


prompt2  = PromptTemplate(
    template='explain the following joke {text}/n',
    input_variables=['text']
)


chain = RunnableSequence(prompt,model,parser,prompt2,model,parser)
print(chain.invoke({'topic':'AI'}))