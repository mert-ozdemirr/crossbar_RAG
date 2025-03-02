# Enhancing Biomedical Knowledge Access: A Retrieval-Augmented Generation Approach for CROSSBARv2

This project aims to enhance the CROssBARv2 biomedical knowledge graph with a Retrieval-Augmented Generation (RAG) model, enabling‬ users to access complex biomedical data through simple, natural‬ language queries. By converting the database into searchable text‬ chunks, the RAG system will efficiently retrieve relevant information,‬ which will be processed with Large Language Models (LLMs) to‬ generate clear, accessible responses. This approach will improve‬ accessibility to vital information, aiding healthcare professionals and‬ researchers in making informed decisions and advancing biomedical‬ research.‬

**Important Note:** Source codes of components and related documents will be published once the project reaches a more complete stage!

## General RAG-Based Response Generation Pipeline

<img width="820" alt="image" src="https://github.com/user-attachments/assets/ad120725-7c1f-424c-a0d0-0bb7ebe12cfd" />

## Two methods: Vector RAG and Graph RAG

The Vector RAG (Text-Based RAG) approach involves transforming textual data into embeddings and storing them in a vector database for efficient similarity-based retrieval. This method ensures that relevant document chunks are retrieved based on semantic similarity before passing them to an LLM for response generation. On the other hand, the Graph RAG approach leverages a structured knowledge graph to retrieve relevant entities and relationships, providing more context-aware and structured query responses. Both implementations aim to enhance information retrieval and response accuracy by integrating Retrieval-Augmented Generation (RAG) techniques with distinct data representations.

## 1. Vector RAG (Text-Based RAG)

### 1.1. Textualization of Data: 

The data was initially structured as a graph database. During the extraction process, various methods were explored for converting the data into a textual format. After careful evaluation, the most relevant information and attributes were selected from nodes and relationships. Additionally, different textual representations for data chunks were considered, with an initial preference for a direct JSON format. However, preliminary tests unexpectedly indicated that a natural language text format was more effective than JSON. To validate these findings, a more comprehensive experiment will be conducted in the near future to reach a conclusive assessment.

### 1.2. Chunking Strategy:

The primary consideration in the chunking strategy was whether to include relationships within each chunk alongside the corresponding node, and if so, determining the appropriate number of relationship layers to incorporate. After evaluating various approaches, a chunking technique was selected in which a separate chunk is constructed for each node, including first-level neighbor associations. Alternative approaches will be explored for the Graph RAG application.  

Additionally, a sub-chunking module was developed to address instances where certain nodes were too large for the embedding models utilized. This module creates sub-chunks based on byte or token limits while maintaining references to parent chunks to ensure proper retrieval. The sub-chunking algorithm incorporates node-related information in each sub-chunk and systematically distributes the relationships of the parent chunk, appending them accordingly.

### 1.3. Embedding Models:

A variety of embedding models were considered for generating embeddings, as this component was identified as a critical factor influencing system performance. To determine the most suitable model, multiple publicly available embedding models were tested and evaluated.  

The LLaMA 3.1:70B and LLaMA 3.3 models were deemed impractical due to their slow inference speeds on available hardware, particularly given the system's scale of approximately 3 million chunks. Additionally, IBM’s Granite-Embedding model from Ollama was not feasible due to its limited context length of 512 tokens, which remained insufficient even with sub-chunking, as certain chunks exceeded this size even before incorporating relationship information.  

As a result, tests were conducted using Ollama’s LLaMA 3.1:7B, LLaMA 3.2, Nomic-Embed-Text models locally, along with Google’s Text-Embeddings-004 model via its API. For evaluation, 160 chunks were randomly selected from each chunk class using a uniform distribution. A total of 15 user prompts were then generated, each associated with target chunks from this dataset. The system’s performance was assessed through two methods: cosine similarity measurements and retrieval rank analysis.

<img width="822" alt="image" src="https://github.com/user-attachments/assets/c3904bef-512c-4c73-8eba-519632ea89bf" />

*Figure 1: [15 user questions that have been used for testing, as well as the chunk indices with those questions.]

Figure 1 illustrates the 15 user questions used for testing. The chunks corresponding to the listed indices are expected to be retrieved at the highest ranks, as all other chunks in the dataset are irrelevant to these questions.

![cosine_similarities](https://github.com/user-attachments/assets/42b8851b-cc55-41da-a882-02d702efb921)

*Figure 2: Cosine similarities between each questions’ embedding and the correspondent chunk’s embedding.

![ranking_results](https://github.com/user-attachments/assets/4eb1f296-e9fe-4ca7-b178-9e882a8b75fb)

*Figure 3: Ranking results for each model. Each question Is used to make a retrieval test with a vector database (ChromaDB) and the ranks of the related chunks. (Lower results are better here)

As illustrated in **Figure 2** and **Figure 3**, **Ollama’s nomic-embed-text model** and **Google’s text-embeddings-004 model** demonstrated superior performance, with **text-embeddings-004** emerging as the most effective among them. However, this model cannot be run locally and is subject to speed limitations when accessed via its API. Given the requirement to generate embeddings over **3 million chunks**, including sub-chunks, its use is not feasible. Consequently, the **nomic-embed-text model** has been selected as the preferred option for now.  

Nonetheless, these findings highlight the potential for improved system performance with more advanced embedding models. Future iterations of the system may incorporate a commercial embedding model to further enhance retrieval effectiveness and overall efficiency.

### 1.4. Generative LLMs:

Similar to embedding models, multiple large language models (LLMs) were available for use. However, unlike embeddings, API-based models were more viable options since they do not require millions of executions within a short timeframe. Additionally, allowing users to select models they are familiar with provides greater flexibility. With this in mind, testing initially focused on locally runnable models using Ollama. The following models were evaluated:  

- deepseek-llm  
- qwen2.5:3b  
- qwen2.5:7b  
- phi3:3.8b  
- gemma2:2b
- llama3.1:8b
- llama3.2
- mistral
- mistral-nemo  

Systematic testing was not conducted for some models due to their poor performance. Among the tested models, **qwen2.5:7b** and **llama3.1:8b** yielded the best results, with their performance being highly similar. To evaluate these models, **7 distinct user prompts** (questions) were used, tested across **17 different scenarios**.

<img width="814" alt="image" src="https://github.com/user-attachments/assets/94321c89-2970-4424-9c78-b1842f1074b7" />

*Figure 4: Example structure of testing schema (qwen2.5:7b)

The evaluation scenarios were designed based on prompt questions and the supplied chunks provided to the LLM. The "supplied chunks" column indicates the number of chunks used as input for the LLM, while the "related chunks" column specifies how many of those chunks contained relevant information to answer the prompt question. The "expected response" column defines the evaluation criteria and references the information necessary to generate an accurate answer based on the provided chunks. Additionally, the "coherency evaluation" column is used to assess the generated response, providing qualitative comments and assigning a subjective response score ranging from 1 to 5.  

Beyond what is shown in Figure 4, response times were also recorded for each scenario. All tests were conducted under the following conditions:  

* The system prompt provided to the LLM:  
  _“Please use the structured data I provide to you as a knowledge base and answer the question in the prompt accordingly. Do not use any additional external information.”_  
* The prompt structure:  
  _"chunk1 + chunk2 + … + chunkN + prompt_question"._  

As previously mentioned, for the **17 test scenarios**, both **qwen2.5:7b** and **llama3.1:8b** models demonstrated **very similar performance**. They exhibited the same errors in certain cases and successfully answered some scenarios. Their **evaluated response scores** are presented in **Figure 5**.

<img width="817" alt="image" src="https://github.com/user-attachments/assets/47e2396a-f753-4770-aa22-ccf62c4e17c3" />

*Figure 5: Comparison of the qwen2.5:7b and llama3.1:8b models based on their scores of each question.

As shown in Figure 5, there was no significant difference between the qwen2.5:7b and llama3.1:8b models. However, both models exhibited fundamental issues in their responses for certain scenarios. To further evaluate performance and explore more robust alternatives, API-based models were tested. The following models were evaluated using their respective APIs:  

- **GPT-4o-mini**  
- **GPT-4o**  
- **Gemini 1.5 Flash**  
- **Gemini 1.5 Pro**  
- **LLaMA 3.1:70B** (via NVIDIA's API platform)  

All of these models successfully passed the test scenarios that were correctly handled by llama3.1:8b and qwen2.5:7b (e.g., scenarios 1, 2, 3, 6, and 10, as shown in Figure 5). Additionally, they demonstrated superior performance in scenarios where the smaller models had previously failed.  

The test results for llama3.1:8b in three sample scenarios (scenarios 7, 11, and 17 from Figure 5) are illustrated in Figure 6.

<img width="819" alt="image" src="https://github.com/user-attachments/assets/d26c3405-a7ba-4467-b82e-a3ffa38017c3" />

*Figure 6: Scenarios 7. 11. And 17 from the Fig. 5. Llama3.1:8b results.

<img width="818" alt="image" src="https://github.com/user-attachments/assets/261f5410-e2f8-453f-bc67-e91005dda393" />

*Figure 7: Scenarios 7. 11. And 17 from the Fig. 5. Gpt-4o mini results.

The results presented in Figure 6 and Figure 7 highlight the significant performance gap between LLaMA 3.1:8B and GPT-4o-mini. Similar trends were observed across all models tested via APIs, indicating that they consistently outperformed the smaller models evaluated locally. This performance disparity reinforces the advantage of using larger, API-based models for improved response quality and accuracy.

![image](https://github.com/user-attachments/assets/0e020396-ac78-4e20-a648-642001dbe452)

*Figure 8: Test results for models accessed through API’s with a new set of scenarios. Scenarios do not correlate with those from Fig. 5 and they are more complex. Models are ordered best-performing to worst-performing top to bottom.

The results clearly favor the **Gemini 1.5** models and raise concerns about the utility of the other models tested, though they may still be considered viable. It is important to note that the current test scenarios did not encompass all possible scenario types (e.g., multi-hop user queries). Future evaluations, potentially including **Deepseekv3** and additional model options, can be conducted within the project workflow.  

Moreover, employing various prompt engineering strategies for both system and user prompts, as well as transitioning from JSON-based to natural-language chunk structures, may further influence performance. Nonetheless, these findings demonstrate that smaller models are insufficient for this use case, while the **Gemini 1.5** models offer a promising solution.

**Important Note:** This information is current as of January 21, 2025. The documentation will be updated later as the project evolves.

## Technologies and Tools

The implementation is primarily developed in **Python**, utilizing powerful libraries from **Ollama** and **Hugging Face**. These libraries have played a crucial role in facilitating efficient integration with modern AI models, streamlining both the development and testing processes. **ChromaDB** has been used as the vector store for embeddings thus far; however, alternative tools will be explored to determine the most effective solution for future use.








