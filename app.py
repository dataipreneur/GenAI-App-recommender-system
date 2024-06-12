import openai
from google.colab import userdata
import pymongo
import streamlit as st
openai.api_key = "key"
EMBEDDING_MODEL = "text-embedding-3-small"

def get_mongo_client(mongo_uri):
  """Establish connection to the MongoDB."""
  try:
    client = pymongo.MongoClient(mongo_uri)
    print("Connection to MongoDB successful")
    return client
  except pymongo.errors.ConnectionFailure as e:
    print(f"Connection failed: {e}")
    return None

mongo_uri = "server"
if not mongo_uri:
  print("MONGO_URI not set in environment variables")

mongo_client = get_mongo_client(mongo_uri)

# Ingest data into MongoDB
db = mongo_client['apps']
collection = db['apps_data']

def get_embedding(text):
    """Generate an embedding for the given text using OpenAI's API."""

    # Check for valid input
    if not text or not isinstance(text, str):
        return None

    try:
        # Call OpenAI API to get the embedding
        embedding = openai.embeddings.create(input=text, model=EMBEDDING_MODEL).data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error in get_embedding: {e}")
        return None

apps_data["embedding_optimised"] = apps_data['description'].apply(get_embedding)

def vector_search(user_query, collection):
    """
    Perform a vector search in the MongoDB collection based on the user query.

    Args:
    user_query (str): The user's query string.
    collection (MongoCollection): The MongoDB collection to search.

    Returns:
    list: A list of matching documents.
    """

    # Generate embedding for the user query
    query_embedding = get_embedding(user_query)

    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    # Define the vector search pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding_optimised",
                "numCandidates": 150,  # Number of candidate matches to consider
                "limit": 5  # Return top 5 matches
            }
        },
        {
            "$project": {  # Exclude the _id field
                "app_package": 1,  # Include the plot field
                "app_name": 1,  # Include the title field
                "developer_name": 1, # Include the genres field
                "content_rating": 1,
                "price": 1,
                "avg_rating": 1,
                "score": {
                    "$meta": "vectorSearchScore"  # Include the search score
                }
            }
        }
    ]

    # Execute the search
    results = collection.aggregate(pipeline)
    return list(results)

# 6. Conduct query with retrival of sources

st.set_page_config(page_title="App Recommendation System")
st.header("What app are you looking for?")
query = st.text_input("Query")
if query != "":
  response, source_information = handle_user_query(query, collection)
  print(source_information)
  print(response)
  st.write(f"Response: {response}\n\nSource Information: {source_information}")
