# კატეგორიზაცია(Best Practice) , საინფორმაციო ტიპის ბოტი უნდა მიდიოდეს პაინქონის ბაზაში
# უნდა მოქონდეს რელევანტური დოკუმენტი და იმაზე დაყრდნობით უნდა გცემდეს პასუხებს

import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

text_data = [
    "Two orders have been placed and they need to be tracked.",
    "Unable to combine orders. Two couriers will deliver your orders.",
    "No, payment with the courier is not possible.",
    "City to city delivery is available. Unfortunately, we cannot offer long-distance delivery service yet.",
    "You will receive the order within 1 hour.",
    "Delivery fee is 3 GEL. Delivery is carried out to Tbilisi, Rustavi, Batumi, Kutaisi, Gori, Telavi, Kobuleti, Poti, Zugdidi.",
    "For returned orders, you will need to pay the courier and contact the operator for assistance.",
    "The following payment methods are available on the PSP website: VISA and Mastercard cards. ApplePay or Gpay. TBC bank transfer. Bank of Georgia payment in installments.",
    "Payment with PLUS card/ Amex card/ Liberty social card is not possible.",
    "It is not possible to purchase the product online with cash payment.",
    "1+1 promotion is not valid during online purchase, however, a 50% discount on the desired product is possible. Which product are you interested in?"
]

response = openai_client.embeddings.create(
    input=text_data,
    model="text-embedding-3-small"
)

embeddings = [emb.embedding for emb in response.data]

index = pc.Index('test', 'https://test-a9vgyyl.svc.aped-4627-b74a.pinecone.io')
vectors = []
for v_id, emb, txt in zip(range(len(embeddings)), embeddings, text_data):
    vectors.append({'id': str(v_id), 'values': emb, 'metadata': {'text': txt}})

index.upsert(vectors, 'knowledge-base')


