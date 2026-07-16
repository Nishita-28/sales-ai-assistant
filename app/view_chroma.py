#python -m app.view_chroma

from app.retriever import get_collection

collection = get_collection()

print(f"Total chunks indexed: {collection.count()}")

data = collection.get()

for i in range(min(5, len(data["documents"]))):
    print("\n" + "=" * 60)
    print("Document:", data["metadatas"][i]["document_name"])
    print("Metadata:", data["metadatas"][i])
    print("Chunk:")
    print(data["documents"][i][:300])   # first 300 characters