from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from langchain_core.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda

load_dotenv()

model_1 = HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b", task="text-generation")
gpt_model = ChatHuggingFace(llm=model_1)



class Feedback(BaseModel):
    sentiment:Literal['positive','negative'] = Field(description='Give the sentiment of the following feedback text into positive or negative \n {feedback} \n {fromat_instructions}')
parser = PydanticOutputParser(pydantic_object=Feedback) 

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative:\n{feedback}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)
classifier_chain = prompt1 | gpt_model | parser


prompt2 = PromptTemplate(
    template='write a appropriate response to this postive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='write a appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)
branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'postive',prompt2|gpt_model|parser),
    (lambda x:x.sentiment == 'negative',prompt3|gpt_model|parser),
    lambda x: "could not find sentiment"
)
chain = classifier_chain|branch_chain
print(chain.invoke({'feedback':'This is a smartphone'}))