from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# Re-use the setup from the previous step
# llm = ChatOpenAI(...)
# ensemble_retriever = ...

# --- 1. Define the MAP Step ---
# This prompt runs on EACH individual document chunk.
# It extracts raw facts without trying to write the full report yet.
map_prompt = ChatPromptTemplate.from_template("""
You are a data analyst extracting key points for a larger report.
Analyze the following document chunk strictly based on the user's query.

User Query: {question}
Document Chunk: {context}

Instructions:
1. Extract every single relevant metric, date, subject, and status.
2. Do not summarize broadly; capture specific details.
3. If the chunk has no relevant info, return "NO_INFO".

Extracted Details:
""")

map_chain = (
    map_prompt 
    | llm 
    | StrOutputParser()
)

# --- 2. Define the REDUCE Step ---
# This prompt takes the list of extracted details and writes the final report.
reduce_prompt = ChatPromptTemplate.from_template("""
You are a professional report writer. Below is a collection of extracted details from various project documents.
Your job is to synthesize these into a robust, structured report.

User Query: {question}

Extracted Details from Search:
{mapped_results}

Report Guidelines:
1. Group related topics together.
2. Highlight contradictions if any (e.g., one doc says 'Done' and another says 'Pending').
3. Use professional formatting (Bullet points, Bold headers).
4. Ignore any "NO_INFO" entries.

Final Report:
""")

reduce_chain = (
    reduce_prompt 
    | llm 
    | StrOutputParser()
)

# --- 3. Build the Pipeline ---

# Helper function to perform the Map step over a list of docs
def map_over_docs(input_dict):
    question = input_dict["question"]
    docs = input_dict["docs"]
    
    # Run the map_chain on each document individually
    # This results in a list of strings (extracted points)
    mapped_results = [
        map_chain.invoke({"question": question, "context": doc.page_content}) 
        for doc in docs
    ]
    
    # Filter out empty results to save tokens
    valid_results = [res for res in mapped_results if "NO_INFO" not in res]
    
    return "\n---\n".join(valid_results)

# The Full Map-Reduce Chain
map_reduce_chain = (
    {
        "question": lambda x: x, 
        "docs": ensemble_retriever # 1. Retrieve Docs
    }
    | RunnableLambda(lambda x: {
        "question": x["question"], 
        "mapped_results": map_over_docs(x) # 2. Map Step
    })
    | reduce_chain # 3. Reduce Step
)

# --- 4. Execution ---
query = "Generate a detailed status report including all subject matter experts opinions"
final_report = map_reduce_chain.invoke(query)

print("### ROBUST MAP-REDUCE REPORT ###")
print(final_report)
