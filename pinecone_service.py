import pinecone
import os
import aiohttp

class Models: 
    DAVINCI = "text-davinci-003"
    CURIE = "text-curie-001"
    EMBEDDINGS = "text-embedding-ada-002"


class PineconeService:
    def __init__(self, index: pinecone.Index):
        self.index = index

    def upsert_basic(self, text, embeddings):
        self.index.upsert([(text, embeddings)])

    def get_all_for_conversation(self, conversation_id: int):
        response = self.index.query(
            top_k=100, filter={"conversation_id": conversation_id}
        )
        return response

    async def upsert_conversation_embedding(
        self, model, conversation_id: int, text, timestamp, custom_api_key=None
    ):
        # If the text is > 512 characters, we need to split it up into multiple entries.
        first_embedding = None
        if len(text) > 500:
            # Split the text into 512 character chunks
            chunks = [text[i : i + 500] for i in range(0, len(text), 500)]
            for chunk in chunks:
                print("The split chunk is ", chunk)

                # Create an embedding for the split chunk
                embedding = await OpenAIEmbeddingService.send_embedding_request(
                    chunk, custom_api_key=OPENAI_API_KEY
                )
                if not first_embedding:
                    first_embedding = embedding
                self.index.upsert(
                    [(chunk, embedding)],
                    metadata={
                        "conversation_id": conversation_id,
                        "timestamp": timestamp,
                    },
                )
            return first_embedding
        else:
            embedding = await OpenAIEmbeddingService.send_embedding_request(
                text, custom_api_key=custom_api_key
            )
            self.index.upsert(
                [
                    (
                        text,
                        embedding,
                        {"conversation_id": conversation_id, "timestamp": timestamp},
                    )
                ]
            )
            return embedding

    def get_n_similar(self, conversation_id: int, embedding, n=10):
        response = self.index.query(
            vector=embedding,
            top_k=n,
            include_metadata=True,
            filter={"conversation_id": conversation_id},
        )
        print(response)
        relevant_phrases = [
            (match["id"], match["metadata"]["timestamp"], match["score"])
            for match in response["matches"]
        ]
        # Sort the relevant phrases based on the timestamp ? but can also sort based on score
        relevant_phrases.sort(key=lambda x: x[1])
        return relevant_phrases

class OpenAIEmbeddingService:
    @staticmethod
    async def send_embedding_request( text, custom_api_key=None):
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": Models.EMBEDDINGS,
                "input": text,
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer { OPENAI_API_KEY if not custom_api_key else custom_api_key}",
            }
            async with session.post(
                "https://api.openai.com/v1/embeddings", json=payload, headers=headers
            ) as resp:
                response = await resp.json()

                try:
                    return response["data"][0]["embedding"]
                except Exception as e:
                    print(response)
                    traceback.print_exc()
                    return





