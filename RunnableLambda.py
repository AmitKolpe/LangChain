from langchain_huggingface import ChatHuggingFace ,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda


load_dotenv()


def word_count(text):
    return len(text.split())


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



joke_gen_chain = RunnableSequence(prompt,model,parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count':RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_gen_chain,parallel_chain)
result = final_chain.invoke({'topic':'cricket'})

final_result = """{} \n word count - {}""".format(result['joke'],result['word_count'])
print(final_result)