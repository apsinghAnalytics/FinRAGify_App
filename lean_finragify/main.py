   
import streamlit as st
from backend_functions import get_stocks, get_preset_questions, get_transcripts, process_transcripts, get_answers, check_management_consistency, rerank_documents
from annotated_text import annotated_text

st.set_page_config(page_title="FinRAGify", page_icon='finragify.png', layout="wide")

# Annotations to give a brief introduction about the app 
annotated_text(
            "Hi there! This",
            ("app", "Proof of Concept", "yellow"),
            "leverages ",
            ("RAG", "Retrieval-Augmented Generation", "pink"),
            " technology, powered by ",
            ("LangChain", "LLM App Framework", "green"),
            " and ",
            ("GPT-4o mini", "OpenAI LLM", "grey"),
            ". It allows you to ",
            ("analyze", "earnings call discussions", "orange"),
            " and ",
            ("generate insights", "by asking questions and querying", "magenta"),
            " earnings transcripts using ",
            ("natural language processing", "machine learning: NLP", "lightblue"),            
            " to assess company performance across multiple quarters.",
            " This app also utilizes ",
            ("FAISS (Facebook AI Similarity Search)", "vector database", "teal"),
            "to store and retrieve ",
            ("embeddings", "OpenAI", "grey"),
            ". The retrieved chunks of data were re-ranked using ",
            ("ms-marco-MiniLM-L-6-v2", "mini language model", "red"),
            ", a cross-encoder model pre-trained on ranking tasks and available on ",
            ("Hugging Face", "AI Model Repository", "yellow"),
            ". The transcripts data was sourced from the ",
            ("FinancialModelingPrep", "Financial Data API", "lightgreen"),
            ".\n\n"

        )


# Display copyright, name, and GitHub link 
st.markdown("""
<p style='text-align: left;'>
    © Aditya Prakash Singh
    <a href="https://github.com/apsinghAnalytics/FinRAGify_App" target="_blank">
        <img src="https://simpleicons.org/icons/github.svg" alt="GitHub" style="height:24px; display:inline-block; vertical-align: middle;">
    </a>
</p>
""", unsafe_allow_html=True)

st.title("FinRAGify: Company Earnings Call RAG Research Tool 📈")
st.sidebar.title("Select Company Ticker")

selected_stock = st.sidebar.selectbox("Choose a Ticker:", get_stocks())  #choose ticker-companyName  from available
selected_ticker= selected_stock.split('-')[0] # Extract the ticker part from 'ticker-companyName' format

preset_questions_dict= get_preset_questions() # dictionary where the key is the question visible to user, while value is the question requested to llm

# Predefined questions
preset_questions = preset_questions_dict.keys() # list of questions visible to user, these are less-detailed and more intuitive

# Allow user to select up to 3 predefined questions
selected_questions = st.sidebar.multiselect("Select up to 3 Questions (Last 1 year data)", preset_questions, max_selections=3)

# Add the management consistency checkbox
check_consistency = st.sidebar.checkbox("How consistent is the management in delivering on past promises? (Last 2 years data)")

# Option to add a custom question from the user
custom_question = st.sidebar.text_input("Add a custom question:")

if custom_question:
    selected_questions.append(custom_question)

run_clicked = st.sidebar.button("Run")

# Initialize session state if not already initialized
if "last_ticker" not in st.session_state: 
    st.session_state.last_ticker = None
    st.session_state.vectorstore_openai = None # the same vectorstore, saved in the session state, is used unless the ticker is changed to another

if run_clicked: 
    if selected_ticker != st.session_state.last_ticker: # If the selected ticker has changed since the last run
        st.session_state.last_ticker = selected_ticker # Update the last_ticker in the session state
        st.session_state.vectorstore_openai = None # Reset the vectorstore to None as the ticker has changed

    if st.session_state.vectorstore_openai is None:   # If the vectorstore is not already loaded
        main_placeholder = st.empty()  # Create a placeholder for displaying status messages
        main_placeholder.text("Retrieving transcripts...")

        transcripts = get_transcripts(selected_ticker) # Retrieve transcripts for the selected ticker

        if transcripts: # If transcripts are found for the selected ticker
            st.session_state.vectorstore_openai = process_transcripts(transcripts)  # Process the retrieved transcripts and store the result in the session state
            main_placeholder.text("Transcripts processed. Answering questions...")
        else:
            st.write("No transcripts found for the selected ticker.") # Display a message if no transcripts are found
            st.session_state.last_ticker = None  # Reset the ticker and vectorstore in the session state
            st.session_state.vectorstore_openai = None
            
    # Process the general questions using the vectorstore and reranking model
    results = get_answers(st.session_state.vectorstore_openai, selected_questions, preset_questions_dict, rerank_documents)

    # If the consistency check is enabled, process the consistency question separately
    if check_consistency:
        consistency_result = check_management_consistency(st.session_state.vectorstore_openai, rerank_documents)
        results.append(consistency_result)

    # Display the results
    st.header("Answers")
    for i, result in enumerate(results):
        st.subheader(f"Question {i+1}: {result['question']}") # Display the question and its corresponding answer
        st.write(result["answer"])

        # Optionally display the relevant transcript text if available
        transcript_chunks = result.get("transcript_chunks", "No relevant transcript text")

        with st.expander(f"**Click to Show Relevant Transcript Text 📜 for Question {i+1}**"):
            st.markdown(transcript_chunks, unsafe_allow_html=False)
                
        sources = result.get("sources", "") # Display the sources, if any, and order them by the latest quarter
        if sources:
            st.markdown("**Sources:**")
            sources_list = sources.split("\n")
            sources_list = list(set(sources_list))  # Convert to set to keep unique entries of sources
            sources_list.sort(reverse=True)  # Order by latest quarter
            
            for source in sources_list:  # Display each source
                st.write(source)

        


        
                    
