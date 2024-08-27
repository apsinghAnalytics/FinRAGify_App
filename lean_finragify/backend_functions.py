import os
import requests
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import cohere # not required when using pre-trained CrossEncoder Model

# Load environment variables
load_dotenv() #the open api key, the Cohere api key, and the finanicalmodelingprep api key, from where transcripts are sourced

chatLLM = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.6) #LLM model


"""
# Initialize the CrossEncoder model
#ranking_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512) # 'ms-marco-MiniLM-L-6-v2' is one of the best minimal language models pre-trained on passage ranking tasks #max_length of 512 tokens per query-passage pair

# Define the reranking function using CrossEncoder
def rerank_documents(docs, query, ranking_model): #Reranking with CrossEncoderModel
    
    # Prepare input pairs (query, doc.page_content) for the CrossEncoder
    input_pairs = [(query, doc.page_content) for doc in docs]

    # Get the relevance scores
    scores = ranking_model.predict(input_pairs)

    # Attach scores to documents and sort them
    ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked_docs]
"""

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



def get_stocks(): # Define the list of stocks to be available to the user here
    stocks = [
    "MSFT-Microsoft",
    "GOOGL-Google",
    "AAPL-Apple",
    "NFLX-Netflix",
    "META-Meta",
    "NVDA-Nvidia",
    "TSLA-Tesla",
    "CRWD-Crowdstrike",
    "ABNB-Airbnb",
    "LULU-Lululemon",
    "HUBS-Hubspot",
    "LLY-Eli Lily",
    "ISRG-Intuitive Surgical",
    "BX-Black Stone"
]
    return stocks

def get_preset_questions(): # returns dictionary containing the intuitive question visible to user (as keys) as well as the detailed one that will be requested to the LLM (as values)
    preset_questions_dict = {
    "What are the future plans and outlook?": "Review and summarize the company's planned initiatives for the upcoming quarters, focusing on strategic goals and future developments, for example, in areas of new markets, new products, new services, share buybacks",
    "What were the key highlights?": "Identify and summarize the key points mentioned, focusing on major achievements, management changes,or developments (e.g. entering new markets, launching new products, new services, share buybacks announcements, ETF inclusion etc.)",
    "Any new product launches?": "List any new products that were launched or are planned to be launched. Provide details on the expected impact of these products on the company's market share and financial performance.",    
    "How is the market responding?": "Describe how the market has reacted to the company's recent announcements, financial results, or product launches.",
    #"Any changes in strategy in the most recent quarter?": "Detail any strategic shifts or major decisions that the company has announced, including changes in leadership, shifts in market focus, or adjustments in operational tactics. Only use the context from the latest quarter",
    #"What are the current or expected tailwinds or headwinds in the industry?": "Identify and summarize any tailwinds or headwinds impacting the industry, such as regulatory changes, economic conditions, technological advancements, or competitive pressures",
    #"What are the expected headwinds in key revenue-generating regions?": "Focus on any challenges or obstacles the company anticipates in its most significant markets, including economic downturns, regulatory issues, competitive pressures, or geopolitical risks",
    "Has there been any recent acquisitions or mergers?": "Identify and provide details on any recent or announced mergers, acquisitions, or partnerships. Include information on the strategic rationale behind these moves",
    #"Is the company facing competitive pressures from major rivals?": "Identify and summarize any significant competitive threats (key competitors) the company is currently facing or expects to face in the near future that might be challenging the company’s market share, pricing power, or strategic position."

    }
    return preset_questions_dict


def get_transcripts(symbol): 
    api_key = os.getenv("FMP_API_KEY")    
    base_url = f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{symbol}"

    def get_transcript(year, quarter):
        url = f"{base_url}?year={year}&quarter={quarter}&apikey={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data:
                return data[0]
        return None
    
    # Retrieve the latest transcript to determine the current year and quarter
    response = requests.get(f"{base_url}?apikey={api_key}")
    if response.status_code != 200:
        print(f"Failed to retrieve latest transcript: {response.status_code}")
        return None
    
    latest_data = response.json()
    if not latest_data:
        print("No transcripts found")
        return None
    
    latest_transcript = latest_data[0]
    latest_year = latest_transcript['year']
    latest_quarter = latest_transcript['quarter']
    
    transcripts = []
    for i in range(8): # 2 years or 8 quarters of transcripts
        quarter_offset = latest_quarter - i
        year = latest_year

        # Adjust year and quarter if necessary
        if quarter_offset <= 0:
            year -= (abs(quarter_offset) // 4) + 1
            quarter = quarter_offset % 4
            if quarter == 0:
                quarter = 4
        else:
            quarter = quarter_offset

        transcript = get_transcript(year, quarter)
        if transcript:
            transcripts.append(transcript)
        else:
            print(f"Transcript not found for {year} Q{quarter}")

    return transcripts


#Breaks transcript documents in small (700 character) chunks and creates a vectorstore with the embeddings
def process_transcripts(transcripts):
    # Step 1: Sort transcripts by year and quarter
    transcripts_sorted = sorted(transcripts, key=lambda x: (x['year'], x['quarter']), reverse=True)
    
    # Step 2: Identify the latest 4 entries
    latest_transcripts = transcripts_sorted[:4]
    latest_ids = {id(entry) for entry in latest_transcripts}  # Use id() to uniquely identify each entry
    
    # Step 3: Create metadata with 'Last 4 Quarters' key
    transcript_metadata = []
    for entry in transcripts:
        is_latest = 'yes' if id(entry) in latest_ids else 'no'
        metadata = {
            "source": f"Earnings Call Transcript Year: {entry['year']}, Quarter: {entry['quarter']}, Date: {entry['date'][0:10]}",
            "Last 4 Quarters": is_latest
        }
        transcript_metadata.append(metadata)
    
    # Step 4: Prepare document contents and embeddings
    transcript_contents = [entry['content'] for entry in transcripts]
    
    #text splitter is defined to break combined transcript data into chunks of 700 characters eac.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )
    
    docs = text_splitter.create_documents(transcript_contents, metadatas=transcript_metadata) # text splitter breaks transcript data to chunks of data in langChain document type
    
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)   #Vectorestore created with FAISS (Facebook AI similarity search) using openAI embeddings
    
    return vectorstore_openai
 
    
# retrieve the embeddings based on the chosen questions (and custom qs) and combine to generate answers with LLM
def get_answers(vectorstore_openai, questions, preset_questions_dict, rerank_documents):
    # Set up the retrieval mechanism
    retriever = vectorstore_openai.as_retriever(search_kwargs={"k": 25}) #25 chunks will be retrieved
    
    # Set up the LangChain LLM with custom prompt
    prompt_template = """You are a financial analyst. Using the following context from earnings call transcripts, answer the question below. Each paragraph includes details about the quarter and year, which helps establish the chronological order of the information. 

    Keep your answer concise, under 200 words.

    Context: {context}

    Question: {question}

    Answer:"""

    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain= prompt | chatLLM | StrOutputParser()

    results = []
    
    for question in questions:
        # Step 1: Retrieve relevant transcript chunks        
        question_orig= question #storing the original_question to return in results for the user to view

        if question in preset_questions_dict.keys():
           
           question=  preset_questions_dict[question]  # For pre_set_questions, pulls the detailed question tuned for better results from LLM
        
        docs=  retriever.invoke(question) #retrieves the data chunks with the most similarity match to the question
    
        # Apply metadata filter to keep only the chunks from last 4 quarters 
        filtered_docs = [doc for doc in docs if doc.metadata.get('Last 4 Quarters') == 'yes']
                
        # Select the top 5 documents after reranking (since only 4 quarters of data, we will take 5)
        # or if less than 5 documents after filtered for last 4 quarters then no of docs available in filtered_docs

        top_n= min(5, len(filtered_docs)) #no of documents to be returned after re-ranking, chosen as 5
        [docs, api_error]= rerank_documents(filtered_docs, question, top_n)  

        # Sort the docs based on the metadata, this would chronologically sort it based on the year, quarters in the metadata["source"]
        docs = sorted(docs, key=lambda doc: doc.metadata["source"]) # a state-of-art llm Gpt-4 should be able to pick up the relevant info even from middle of the full context, so we are ordering for better readability to user

        # Combine the chunks into a single context string and also keeping the metadata associated with each chunk
        # Formating the metadata to bold, and adding block quotes to content in markdown
        context = "\n\n".join([('**'+ doc.metadata["source"][25:] +'**     \n' +'>'+ doc.page_content) for doc in docs]) #Adding the source metada to chunk but skipping the first 25 characters of metadaata string i.e. 'Earnings Call Transcript ' part
        
        # Step 2: Generate the answer using the retrieved chunks                    
        answer= chain.invoke({"context": context, "question": question})        

        # Step 3: Prepare the result, including both the answer and the chunks
        result = {
            "question": question_orig, #the original qs, which is the more concise, and intuitive is shown to user rather than the detailed one for retriever and llm prompting
            "answer": answer,
            "sources": "\n".join([doc.metadata["source"] for doc in docs]),
            "transcript_chunks": api_error + context  # The api_error will inform the user when the Cohere API reranking fails due to exceeding monthly limit or other reasons
        }
        
        results.append(result)
    
    return results

def check_management_consistency(vectorstore_openai, rerank_documents):

    retriever = vectorstore_openai.as_retriever(search_kwargs={"k": 25}) #25 chunks will be retrieved

    # Define the custom prompt for management consistency
    prompt_template =  """
    You are a financial analyst. Given transcripts of earnings calls across multiple quarters, each paragraph contains details about the quarter and year, which helps establish the chronological order of the information.

    Analyze the statements made in previous quarters about specific expectations for future quarters and compare them with the outcomes reported for those subsequent quarters in order to validate whether those expectations were met. 

    Identify any delays, missed expectations, or discrepancies between promises and outcomes. Summarize the findings in three lines, including the number of times management met their expectations versus the number of times they did not. Keep your summary concise, under 300 words.

    Context: {context}

    Answer:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    chain = prompt | chatLLM | StrOutputParser()

    # Retrieve relevant transcript chunks
    retriever_search_qs= """What were the specific targets, deadlines, or expectations—such as those to be met by a certain quarter—in areas of product launches, strategic initiatives, cost-cutting measures, growth in new markets, share buybacks etc., that the management set to deliver on future quarters, and have they delivered on them?."""
   
    docs = retriever.invoke(retriever_search_qs)      

        
    # Select the top 10 documents after reranking
    [docs, api_error]=  rerank_documents(docs, retriever_search_qs, 10)   #reranked_docs[:10] #more no of chunks used because here 2 years of transcript data is used instead of 1 year in get_answers

    # Sort the docs based on the metadata
    docs = sorted(docs, key=lambda doc: doc.metadata["source"]) # Ordering chronologically for better readability to user

    # Combine the chunks into a single context string
    context = "\n\n".join([('**'+ doc.metadata["source"][25:] +'**     \n' +'>'+ doc.page_content) for doc in docs]) #similar formatting as in 'get_answers'
    
    # Generate the answer
    answer = chain.invoke({"context": context})    

    # Prepare the result
    result = {
        "question": "How consistent is the management in delivering on past promises?",
        "answer": answer,
        "sources": "\n".join([doc.metadata["source"] for doc in docs]),
        "transcript_chunks": api_error + context # The api_error will inform the user when the Cohere API reranking fails due to exceeding monthly limit or other reasons
    }
    
    return result

