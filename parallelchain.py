from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
load_dotenv()

model_1 = HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b", task="text-generation")
gpt_model = ChatHuggingFace(llm=model_1)

model_2 = HuggingFaceEndpoint(repo_id="google/gemma-2-2b-it", task="text-generation")
gemma_model = ChatHuggingFace(llm=model_2)

prompt1 = PromptTemplate(
    template='Generate a summary on following passage: {passage}',
    input_variables=['passage']
)

prompt2 = PromptTemplate(
    template='Generate 5 questions and answers on the following passage: {passage}',
    input_variables=['passage']
)

prompt3 = PromptTemplate(
    template='Merge the provided text into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes','quiz']
)
parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | gpt_model | parser , 
    'quiz': prompt2 | gemma_model | parser
})

merge_chain = prompt3 | gpt_model | parser

chain = parallel_chain | merge_chain    

passage = """Computer Networks and Their Role in Artificial Intelligence Infrastructure
1. Introduction

A computer network is a system of interconnected devices—computers, servers, storage units, IoT devices, and more—that communicate with each other using standard protocols. Networking forms the backbone of modern digital ecosystems, enabling data sharing, remote processing, cloud services, and large-scale distributed computing.
In the context of Artificial Intelligence (AI), networks are not just a medium for communication—they are the lifeline that enables AI models to be trained, deployed, and scaled efficiently.

2. Basics of Computer Networks

Computer networks are typically classified based on their scope and architecture:

Types by Scale:

LAN (Local Area Network) – connects devices in a limited area such as offices or labs.

WAN (Wide Area Network) – spans large geographical areas, often using public infrastructure like the internet.

MAN (Metropolitan Area Network) – covers cities or campuses.

PAN (Personal Area Network) – connects personal devices in close range.

Topologies: Star, Mesh, Ring, Bus, and Hybrid designs, each optimized for reliability, cost, and scalability.

Key Components:

Networking hardware (routers, switches, hubs, firewalls).

Communication media (Ethernet cables, fiber optics, wireless links).

Protocols (TCP/IP, HTTP, FTP, DNS).

3. Networking in AI Infrastructure

AI systems, especially large-scale ones, require significant computing resources. These resources are rarely housed in a single machine—instead, they are distributed across clusters, data centers, and cloud platforms. Networking enables this distribution by:

3.1 Data Acquisition and Transfer

AI models thrive on vast amounts of data. Networks allow:

Real-time data collection from IoT devices, sensors, and APIs.

Transfer of datasets from storage nodes to training nodes in data centers.

Use of high-speed interconnects (InfiniBand, NVLink, 5G) to reduce data transfer bottlenecks.

3.2 Distributed Training

Modern AI models (like GPT-class LLMs or large vision transformers) require distributed computing:

Horizontal scaling across multiple GPUs/TPUs in separate servers.

Model parallelism and data parallelism—splitting workloads across nodes connected through high-bandwidth, low-latency networks.

Use of technologies like RDMA (Remote Direct Memory Access) to speed up parameter synchronization between training nodes.

3.3 Cloud AI Services

Cloud providers (AWS, Azure, Google Cloud) rely on massive networking infrastructure to:

Connect globally distributed data centers.

Offer AI APIs and managed services accessible worldwide.

Support hybrid setups where part of the AI workload runs on-premises and part in the cloud.

3.4 Edge AI

In scenarios like autonomous vehicles or industrial robots, AI inference must happen close to where the data is generated:

Edge computing networks connect local devices to nearby servers for low-latency processing.

5G networks provide high speed and ultra-low latency, enabling real-time decision-making.

4. Challenges in Networking for AI

While networks are essential, AI workloads introduce unique challenges:

Bandwidth demand: AI training involves petabytes of data transfer.

Latency sensitivity: Synchronization delays slow down training speed.

Security risks: AI data pipelines must be protected from breaches.

Scalability: Infrastructure must support rapid growth in AI models and datasets.

5. Future Trends

The intersection of networking and AI will continue to evolve:

AI-driven networking: AI algorithms optimize routing, detect anomalies, and manage network traffic.

Quantum networking: Could revolutionize distributed AI computing with ultra-secure, high-speed links.

Self-healing AI infrastructures: Networks that autonomously detect and fix failures in AI clusters.

6. Conclusion

Computer networks are the foundation of AI infrastructure, enabling everything from data acquisition to large-scale distributed training and real-time inference. As AI models grow larger and more complex, networking technologies will need to advance in parallel—focusing on higher bandwidth, lower latency, and greater security. The future of AI is deeply intertwined with the future of computer networks, making networking expertise a critical component in building the next generation of intelligent systems."""

result = chain.invoke({'passage':passage})

print(result)