from langchain_huggingface import ChatHuggingFace ,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnablePassthrough


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

jok_gen_chain  = RunnableSequence(prompt,model,parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'explanation':RunnableSequence(prompt2,model,parser)
})


final_chain = RunnableSequence(jok_gen_chain,parallel_chain)

print(final_chain.invoke({'topic':'news'}))