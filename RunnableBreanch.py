from langchain_huggingface import ChatHuggingFace ,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda,RunnableBranch

load_dotenv()

llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        task="text-generation"
)
 
model = ChatHuggingFace(llm = llm)


prompt1 = PromptTemplate(
    template='write a detailed report on {topic} ',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='summarize the following text \n {topic} ',
    input_variables=['topic']
)

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1,model,parser)

branch_chain =RunnableBranch(
    (lambda x: len(x.split())>500,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)
final_chain = RunnableSequence(report_gen_chain,branch_chain)
result  = final_chain.invoke({'topic':'IPL'})
print(result)