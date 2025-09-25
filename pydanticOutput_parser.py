from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# ✅ Use PascalCase class name
class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(default=18, description="Age of the person")
    city: str = Field(description="City of the person")

# ✅ Parser
parser = PydanticOutputParser(pydantic_object=Person)

# ✅ Fix variables + spelling
template = PromptTemplate(
    template="Generate the name, age, and city of a fictional person in {place}.\n{format_instructions}",
    input_variables=["place"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


'''
# ✅ Invoke template
prompt = template.invoke({"place": "India"})

# ✅ Run model
result = model.invoke(prompt)

# ✅ Parse into Person object
final_result = parser.parse(result.content)
print(final_result)
'''

# using chain
chain = template|model|parser
final_result = chain.invoke({'place':'india'})
print(final_result)