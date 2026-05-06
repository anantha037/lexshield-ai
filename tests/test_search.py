from rag.vectorstore import vectorstore

def run_test():
    # We will test two very different queries to ensure it understands context
    queries = [
        "The landlord is not returning my security deposit. What can I do?",
        "What is the punishment for cheating and dishonestly inducing delivery of property?"
    ]
    
    for query in queries:
        print(f"\nSearching for: '{query}'")
        print("=" * 70)
        
        # Search the database for the top 3 most relevant chunks
        results = vectorstore.search(query, n_results=3)
        
        if not results:
            print("No results found. Database might be empty.")
            continue

        for i, res in enumerate(results, 1):
            print(f"Rank {i} (Similarity Score: {res['score']})")
            print(f"Source: {res['source']}")
            print(f"Section: {res['section'] or 'N/A'}")
            print(f"Type: {res['doc_type'].upper()}")
            # Print just the first 250 characters of the text so it's easy to read
            print(f"Snippet: {res['text'][:250]}...\n")
            print("-" * 70)

if __name__ == "__main__":
    print("Booting up LexShield Semantic Search Test...")
    run_test()