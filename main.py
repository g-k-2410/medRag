import openai
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from transformers import AutoTokenizer, AutoModel
import gradio as gr

# Connect to Milvus
ngrok_host = '0.tcp.ngrok.io'
ngrok_port = '12345'
connections.connect("default", host=ngrok_host, port=ngrok_port)

# Define schema for Milvus collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
]
schema = CollectionSchema(fields)
collection = Collection(name="healthcare_data", schema=schema)

# Initialize OpenAI API key
openai.api_key = "your-api-key"

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Function to generate embeddings
def generate_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings

# Function to insert embeddings into Milvus
def insert_embeddings(texts):
    embeddings = [generate_embedding(text).tolist() for text in texts]
    collection.insert([embeddings])
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 100}
    }
    collection.create_index("embedding", index_params)

# Function to retrieve similar documents
def retrieve_similar_documents(query_embedding, top_k=5):
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search([query_embedding], "embedding", search_params, top_k=top_k)
    return results

# Function to generate a response using OpenAI GPT-4
def generate_response(prompt):
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Function to generate an augmented response
def generate_augmented_response(query):
    query_embedding = generate_embedding(query).tolist()
    similar_docs = retrieve_similar_documents(query_embedding)
    context = " ".join([str(doc) for doc in similar_docs])
    augmented_prompt = f"Given the following medical context, provide a detailed response: {context}"
    return generate_response(augmented_prompt)

# Gradio interface
def run_gradio_app():
    iface = gr.Interface(fn=generate_augmented_response, inputs="text", outputs="text")
    iface.launch()

if __name__ == "__main__":
    run_gradio_app()
