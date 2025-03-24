from transformers import pipeline

# Load a local LLM (T5 for summarization)
llm = pipeline("text2text-generation", model="google/flan-t5-large")

def generate_response(query, retrieved_chunks):
    context = " ".join(retrieved_chunks[:5])  # Use Top 5 chunks

    prompt = f"""
    You are an AI assistant answering based on a document.
    Context: {context}
    Question: {query}
    Provide a clear and detailed answer using ONLY the provided context.
    """

    print(f"📝 LLM Prompt:\n{prompt}")  # Debugging: Print what is being sent to LLM

    response = llm(prompt, max_length=256, min_length=50)
    return response[0]['generated_text']


if __name__ == "__main__":
    text = "Your retrieved text chunk here..."
    print(generate_response(text))