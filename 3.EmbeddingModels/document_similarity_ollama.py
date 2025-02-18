from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
load_dotenv()
from sklearn.metrics.pairwise import cosine_similarity



embeddings = OllamaEmbeddings(model="mxbai-embed-large")

documents = [
    "Lionel Messi, an Argentine forward, has won eight Ballon d'Or awards and led his national team to World Cup victory.",
    "Cristiano Ronaldo, hailing from Portugal, is the all-time highest goalscorer and has secured five Ballon d'Or titles.",
    "Pelé, the legendary Brazilian footballer, is celebrated as one of the greatest players in history, with three FIFA World Cup victories.",
    "Diego Maradona, an Argentine football maestro, made an indelible impression on the global sports scene with his extraordinary talent.",
    "Zinedine Zidane, a French midfielder, is renowned for his supreme skill in critical matches, including leading France to World Cup glory."
]

query = "Tell me about pele"

document_embeddings = embeddings.embed_documents(documents)
print("document_embeddings ===> ",document_embeddings)
query_embedding = embeddings.embed_query(query)
print("query_embedding ===> ",query_embedding)

# PERFORM SEMANTIC SEARCH using cosine similarity


similarities = cosine_similarity([query_embedding], document_embeddings) # this will provide the similarity of the query with each document
print("similarities ===> ",similarities)
# we will have to sort the similarities to get the most similar document
# for that we will have to persist the index of the document in the documents list
indexed_similarities = enumerate(similarities[0])
print("indexed_similarities ===> ",indexed_similarities)

sorted_similarities = sorted(indexed_similarities, key=lambda x: x[1])
print("sorted_similarities ===> ",sorted_similarities)

most_similar_index = sorted_similarities[-1][0]
print("most_similar_index ===> ",most_similar_index)

most_similar_document = documents[most_similar_index]
print("most_similar_document ===> ",most_similar_document)