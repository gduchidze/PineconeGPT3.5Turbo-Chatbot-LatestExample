import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index('test', 'https://test-a9vgyyl.svc.aped-4627-b74a.pinecone.io')


def get_embedding(text):
    response = openai_client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def search_pinecone(query, top_k=1):
    query_embedding = get_embedding(query)
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace='knowledge-base'
    )
    return results['matches']


def use_pinecone_data(query):
    matches = search_pinecone(query)
    if matches:
        return matches[0].metadata['text']
    return None


def get_openai_response(messages):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message.content


def chatbot():
    messages = [
        {"role": "system", "content": """
        You are an AI assistant for PSP, a Georgian pharmacy chain. 
        Your task is to provide information about PSP's services, products, and policies in English.
        Use the information provided from the knowledge base, but formulate brief, 
        context-aware responses. If asked about specific locations or prices, 
        interpret the information intelligently.
        Begin by greeting the user warmly and asking how you can assist them today.
        Start with a short, friendly greeting .

        """},
    ]

    initial_greeting = get_openai_response(messages)
    print("Chatbot:", initial_greeting)
    messages.append({"role": "assistant", "content": initial_greeting})

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            farewell = get_openai_response(messages + [{"role": "user",
                                                        "content": "The user wants to end the conversation. Provide a friendly farewell message."}])
            print("Chatbot:", farewell)
            break

        knowledge_base_info = use_pinecone_data(user_input)

        if knowledge_base_info:
            prompt = f"""
            User query: {user_input}

            Relevant information from knowledge base: {knowledge_base_info}

            Please provide a brief, context-aware response based on this information in English.
            If the query is about a specific location or price, interpret the information intelligently.
            For example, if asked about delivery to Plekhanov, understand it's in Tbilisi and respond accordingly.
            Aim for a concise, direct answer that addresses the user's specific question.
            """
        else:
            prompt = f"""
            User query: {user_input}

            No specific information found in the knowledge base for this query.
            Please provide the best response you can based on general knowledge about pharmacies in the Georgian context.
            Keep the response brief and tailored to PSP pharmacy services. Respond in English.
            """

        messages.append({"role": "user", "content": prompt})
        response = get_openai_response(messages)

        print("Chatbot:", response)
        messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    chatbot()