from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence,RunnableParallel
from dotenv import load_dotenv


load_dotenv()

llm = HuggingFaceEndpoint(
     repo_id="openai/gpt-oss-20b",
        task="text-generation"
)

model = ChatHuggingFace(llm=llm)


prompt1 = PromptTemplate(
    template='write tweet on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='generate linkdin post about {topic} /n',
    input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet':RunnableSequence(prompt1,model,parser),
    'linkdin':RunnableSequence(prompt2,model,parser)
})

result = parallel_chain.invoke({'topic':'cricket'})
print(result)