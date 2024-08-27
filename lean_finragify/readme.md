*This repo is for the light weight version of the FinRAGify_APP, which uses the [Cohere Rerank API,](https://docs.cohere.com/reference/rerank) instead of the open source [*CrossEncoder model (ms-marco-MiniLM-L-6-v2)*](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) for reranking the retrieved data chunks. This *reduces the RAM requirements from 300- 600 MB to about 150-300 MB*, which can be very helpful in deploying the app to smaller free-tier cloud instances like the AWS EC2 t3.micro*


## Project Structure

- main.py: The main Streamlit application script.
- backend_functions: The functions for the app are defined here. 
- requirements.txt: A list of required Python packages for the project.
- .env: Configuration file for storing your OpenAI, FinancialModelingPrep, and additionally the **Cohere API** keys:  
- dockerfile: The docker file to create the docker image if the user prefers to run the app by containerizing and deploying via Docker.

## Major Changes

The major change in this lightweight version of the app is redefining the *rerank_documents* function in *backend_functions.py* to use the rerank endpoint from the Cohere API. This reduces memory requirements by eliminating the need to load the CrossEncoder model into memory. Disk space requirements are also reduced because the *sentence-transformer* module (specifically `from sentence_transformers import CrossEncoder`) is no longer needed, which means that large packages like PyTorch (2+ GB) are not required. Additionally, there were other minor changes, such as adjustments to function arguments etc.


```python
cohere_api_key= os.getenv('COHERE_API_KEY')  #This is required when ranking is not done based on CrossEncoder model (helps to save on RAM req. for cloud deployment)
co = cohere.Client(cohere_api_key)

def rerank_documents(docs_langchain, query, n): # n is the top n results to be returned   
    try:
        
        # Rerank using Cohere's rerank endpoint        
        results = co.rerank(query=query, documents=[doc.page_content for doc in docs_langchain], top_n=n, model='rerank-english-v3.0')

        # Access the results attribute correctly
        top_docs_langchain = [docs_langchain[result.index - 1] for result in results.results]
        api_error = "\n\n" # blank string for no error
        
        return [top_docs_langchain, api_error]
    
    except Exception: # When error occurs e.g. when max API limit reached or any other reason then no reranking used        

        api_error= '### Max API calls allowed per month for Cohere Rerank API reached, no reranking algorithm was used \n\n'
        return [docs_langchain[:n], api_error]
```



## Installation

### Method 1: Cloning GitHub Repo to Local Machine

1. Clone this repository to your local machine using:

```bash
git clone https://github.com/apsinghAnalytics/FinRAGify_App.git
```

2. Navigate to the project directory:

```bash
cd FinRAGify_APP/lean_finragify
```

3. Create a local Python environment and activate it:

```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

4. Install the required packages (unlike the main-version of this app, this version doesn't need torch) from `requirements.txt`:

```bash
pip install -r requirements.txt
```

5. Set up your API keys by creating a `.env` file in the project root and adding your keys:

```bash
OPENAI_API_KEY='your_openai_api_key_here'
FMP_API_KEY='your_fmp_api_key_here'
COHERE_API_KEY='your_cohere_api_key_here'
```

<p align="center"> <img width="600" src="https://raw.githubusercontent.com/apsinghAnalytics/FinRAGify_App/main/images/env_file_cohere.png"> </p>

6. Run the Streamlit app by executing:

```bash
streamlit run main.py
```

### Method 2: Docker Containerization

Same as that of the main app, so please refer to this [readme](https://github.com/apsinghAnalytics/FinRAGify_App/blob/main/README.md). *Ensure that the docker file in this repo is used to create the docker image, and that the .env file used contains the Cohere API keys.*

The memory usage for this lean version when deployed using Docker is in the range of 150 -300 MB:

<p align="center"> <img width="600" src="https://raw.githubusercontent.com/apsinghAnalytics/FinRAGify_App/main/images/lean_finragify_dockerMemoryReq.png"> </p>