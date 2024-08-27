*An LLM app leveraging RAG with LangChain and GPT-4 mini to analyze earnings call transcripts, assess company performance, evaluate management's track record by using natural language queries (NLP), FAISS (vector database), and Hugging Face re-ranking models.*

Checkout out the deployed app here: [http://ec2-40-177-46-181.ca-west-1.compute.amazonaws.com:8501](http://ec2-40-177-46-181.ca-west-1.compute.amazonaws.com:8501)


# FinRAGify_App: 

<p align="center"> <img width="150" src="https://raw.githubusercontent.com/apsinghAnalytics/FinRAGify_App/main/images/finragify.png"> </p>

FinRAGify is a user-friendly research tool designed to simplify the process of retrieving information from earnings calls of publicly traded companies. Users can select a company from a limited list (available for this proof-of-concept) and ask questions from a set of presets or create custom queries, such as *"Were any new products launched?"* or *"What are the company’s future plans and outlook?"* The app then searches (using embeddings) the last two years (8 quarters) of quarterly earnings calls by leveraging **RAG (Retrieval-Augmented Generation)** technology, a machine learning technique that combines retrieval-based and generative models (GPT, LLMs), to find and present contextually relevant answers.

<p align="center"> <img width="800" src="https://raw.githubusercontent.com/apsinghAnalytics/FinRAGify_App/main/images/finragify_UI.gif"> </p>



## Features

- **Load and Process Earnings Call Transcripts:** Fetch earnings call transcripts for selected stocks through the [*FinancialModelingPrep API,*](https://site.financialmodelingprep.com/developer/docs#earnings-transcripts) retrieving up to 8 quarters of data and sorting them by year and quarter.
- **Embedding and Vector Store Creation:** Construct embedding vectors using *OpenAI's embeddings* and store them in a [*FAISS (Facebook AI Similarity Search) vector store*](https://python.langchain.com/v0.2/docs/integrations/vectorstores/faiss/) for fast and effective retrieval of relevant transcript chunks.
- **Re-rank Documents for Relevance:** Use a [*CrossEncoder model (ms-marco-MiniLM-L-6-v2)*](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) available on *hugging face* to re-rank retrieved transcript chunks and choose a smaller pool of the most relevant informatio for answering user queries.
- **Preset and Custom Financial Questions:** Offer a selection of preset financial questions focused on key business areas (e.g., future plans, product launches) with the flexibility to input custom queries.
- **Management Consistency Analysis:** Evaluate management's track record by comparing past promises with actual outcomes across multiple quarters, summarizing how often targets were met.

## Project Structure

- main.py: The main Streamlit application script.
- backend_functions: The functions for the app are defined here. 
- requirements.txt: A list of required Python packages for the project.
- .env: Configuration file for storing your OpenAI and FinancialModelingPrep API keys:  
- dockerfile: The docker file to create the docker image if the user prefers to run the app by containerizing and deploying via Docker.
- lean_finragify: The repo for the light weight version of this app, which uses the [Cohere Rerank API,](https://docs.cohere.com/reference/rerank) instead of the open source [*CrossEncoder model (ms-marco-MiniLM-L-6-v2)*](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) for reranking the retrieved data chunks. This *reduces the RAM requirements from 300- 600 MB to about 150-300 MB*, which can be very helpful in deploying the app to smaller cloud compute instances like the AWS EC2 t3.micro. Please refer to the [readme](https://github.com/apsinghAnalytics/FinRAGify_App/blob/main/lean_finragify/README.md) inside for installation instructions of that light version. 

## Installation

### Method 1: Cloning GitHub Repo to Local Machine

1. Clone this repository to your local machine using:

```bash
git clone https://github.com/apsinghAnalytics/FinRAGify_App.git
```

2. Navigate to the project directory:

```bash
cd FinRAGify_APP
```

3. Create a local Python environment and activate it:

```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

4. Install the required packages, starting with the specific version of Torch (**this must be installed before installing from requirements.txt**):

```bash
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cpu
```

5. Install the remaining dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

6. Set up your API keys by creating a `.env` file in the project root and adding your keys:

```bash
OPENAI_API_KEY='your_openai_api_key_here'
FMP_API_KEY='your_fmp_api_key_here'
```

<p align="center"> <img width="600" src="https://raw.githubusercontent.com/apsinghAnalytics/FinRAGify_App/main/images/env_file.png"> </p>

7. Run the Streamlit app by executing:

```bash
streamlit run main.py
```

### Method 2: Docker Containerization

**Note:** Using a dockerized container to deploy this app requires about 200 MB more in terms of RAM

1. Copy the `Dockerfile` and `.env` file to the same folder on your local machine.

2. Open PowerShell (or your preferred terminal) and navigate to this folder:

```bash
cd path_to_your_folder
```

3. Build the Docker image using the following command:

```bash
docker build -t finragify_app:latest .
```
**Note:** *Ensure that you have docker (docker desktop for Windows) installed and running before using docker commands*

4. Once the Docker image is created, run the Docker container by mapping the exposed port `8501` (see the dockerfile) to an available port on your local machine (e.g., `8501`, `8502`, `8503`):

```bash
docker run -d -p 8503:8501 --name finragify_container finragify_app:latest #this maps 8503 of local machine to exposed port 8501 of the app
```

5. Access the Streamlit app by navigating to `http://localhost:8503` (or the port you've mapped) in your web browser.

### Deployment Note

If you prefer to **deploy this application on an AWS EC2 instance**, you can follow the general EC2 Streamlit app deployment steps mentioned in my previous README for another app [here](https://github.com/apsinghAnalytics/streamlit_VentureGen).


