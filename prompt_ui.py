from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Initialize HuggingFace LLM
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Streamlit App Header
st.header("AI Research Paper Summarizer")

# Paper options
paper_input = st.selectbox(
    "Select Research Paper",
    [
        "Attention Is All You Need (Vaswani et al., 2017)",
        "BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2019)",
        "GPT-3: Language Models are Few-Shot Learners (Brown et al., 2020)",
        "Diffusion Models Beat GANs on Image Synthesis (Dhariwal & Nichol, 2021)",
        "LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)",
        "Qwen2: Bridging Language and Multimodality (2024)"
    ]
)

# Style options
style_input = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner-Friendly (Simple Language)",
        "Technical (Research-Oriented)",
        "Code-Oriented (With Examples)",
        "Mathematical (Equations & Derivations)",
        "Critical Review (Strengths & Limitations)"
    ]
)

# Length options
length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (Detailed Explanation, 6-10 paragraphs)",
        "Comprehensive (Full Review)"
    ]
)

# Prompt Template
template = PromptTemplate(
    template="""
Summarize the research paper titled "{paper_input}" with the following specifications:

- Explanation Style: {style_input}  
- Explanation Length: {length_input}  

**Guidelines**:  
1. Mathematical Details:  
   - Include equations if present in the paper.  
   - Provide intuitive explanations or small code snippets when applicable.  

2. Analogies:  
   - Use relatable analogies to simplify complex concepts.  

3. Critical Notes:  
   - If information is unavailable, respond with "Insufficient information available".  

The explanation must be **clear, accurate, and aligned** with the selected style and length.
""",
    input_variables=["paper_input", "style_input", "length_input"]
)

# Fill the placeholders
prompt = template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})

# Summarize button
if st.button("Summarize"):
    result = model.invoke(prompt)
    st.write(result.content)
