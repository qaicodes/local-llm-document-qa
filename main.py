import argparse
import timeit
from src.local_llm import CTransformers
from src.utils import setup_vectordbqa, set_qa_prompt, build_vectordb_retriever_qa

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a prompt-based language model')
    parser.add_argument('input', type=str)
    args = parser.parse_args()
    start = timeit.default_timer()


    # Setup qa object
    vectordbqa = setup_vectordbqa()


    # Parse response
    response = vectordbqa({'query' : args.input})
    end = timeit.default_timer()

    # Print response
    print(f'\nAnswer: {response["result"]}')
    print("*"*50)
    print(response)
    #  # Process source documents for better display
    # source_docs = response['source_documents']
    # for i, doc in enumerate(source_docs):
    #     print(f'\nSource Document {i+1}\n')
    #     print(f'Source Text: {doc.page_content}')
    #     print(f'Document Name: {doc.metadata["source"]}')
    #     print(f'Page Number: {doc.metadata["page"]}\n')
    #     print('='* 50) # Formatting separator
        
    # Display time taken for CPU inference
    print(f"Time to retrieve response: {end - start}")

