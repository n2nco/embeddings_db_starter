

from pinecone_service import PineconeService
from pinecone_service import Models
import pinecone
import os
import asyncio
import datetime


# get PINECONE_INDEX from env if it exists

"""
The pinecone service is used to store and retrieve conversation embeddings.
"""
try:
    PINECONE_TOKEN = os.environ.get("PINECONE_TOKEN")
    PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
except:
    PINECONE_TOKEN = None

if PINECONE_TOKEN:
    pinecone.init(api_key=PINECONE_TOKEN, environment="us-west1-gcp")
try: 
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
except:
    OPENAI_API_KEY = None

# catch exception if PINECONE_TOKEN is not set

# except Exception:
#     message = "Something went wrong, please try again later. This may be due to upstream issues on the API, or rate limiting."
pinecone_service = None
if PINECONE_TOKEN:
    pinecone.init(api_key=PINECONE_TOKEN, environment="us-west1-gcp")
    PINECONE_INDEX = "conversation-embeddings"
    if PINECONE_INDEX not in pinecone.list_indexes():
        print("Creating pinecone index. Please wait...")
        pinecone.create_index(
            "conversation-embeddings",
            dimension=1536,
            metric="dotproduct",
            pod_type="s1",
        )
    pinecone_service = PineconeService(pinecone.Index(PINECONE_INDEX))
    print("Got the pinecone service")

# import aioflags

async def main():
    print("\nWhat text would u like to embed?")
    global i
    i = input()
    if len(i) > 0:
	    text_to_embed = i
 
    timestamp = datetime.datetime.now().timestamp()
    # create an embedding
    embedding = await pinecone_service.upsert_conversation_embedding(
        Models.EMBEDDINGS,
        conversation_id='test',
        text=text_to_embed,
        timestamp = timestamp,
        custom_api_key=os.environ.get("OPENAI_API_KEY")
    )
    print(embedding)


    relevant_phrases =  pinecone_service.get_n_similar(conversation_id='test', embedding=embedding, n=100)
    # relevant_phrases_excluding_this_runs_upsert = list(filter(lambda x: matches['score'] != 1.0, relevant_phrases['matches']))
    relevant_phrases_excluding_this_input = remove_tuples_by_string(relevant_phrases, i)

    print("given the text: '", text_to_embed, "' here are the most relevant embedding matches: ",  relevant_phrases_excluding_this_input )

def remove_tuples_by_string(relevant_phrases, i):
    print("i is ", i)
    return [phrase for phrase in relevant_phrases if i not in phrase[0]]

if __name__ == '__main__':      
    asyncio.run(main())


