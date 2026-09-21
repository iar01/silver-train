from pipeline import RAGPipeline

def start_bot():
    print("Initializing Professional Recipe RAG...")
    rag_system = RAGPipeline()

    print("\n--- Chef Bot Ready! Type 'exit' to quit. ---")
    while True:
        query = input("\nYou: ")
        if query.lower() in ['exit', 'quit']:
            break
            
        answer, _ = rag_system.answer(query)
        print(f"\nChef Bot: {answer}")

if __name__ == "__main__":
    start_bot()