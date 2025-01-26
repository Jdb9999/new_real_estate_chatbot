from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()

# Load pre-trained model for chatbot
print("Loading model...")
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-large")
print("Model loaded successfully!")

# Predefined responses for real estate questions
real_estate_faq = {
    "What is the price of 123 Main St?": "The price of 123 Main St is $350,000.",
    "How many bedrooms does this house have?": "This house has 4 bedrooms.",
    "What is the square footage of the property?": "The property is 2,500 square feet.",
    "Is there a pool in this property?": "Yes, the property has an outdoor swimming pool.",
    "Can I schedule a viewing for this property?": "You can schedule a viewing by calling our office at 555-1234 or booking online.",
    "Where is the property located?": "The property is located at 123 Main St, Springfield, IL.",
    "Is this house for rent or sale?": "This property is for sale."
}

# Define request schema using Pydantic
class UserQuery(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Real Estate Chatbot! Ask me anything related to real estate."}

@app.post("/chat/")
def chat(query: UserQuery):
    user_input = query.message.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Input cannot be empty.")
    
    # Check if the user's question matches the FAQ
    for question, answer in real_estate_faq.items():
        if user_input.lower() == question.lower():
            return {"response": answer}
    
    # Fallback: Generate a response using the chatbot
    try:
        response = chatbot(user_input, max_length=1000, pad_token_id=50256)
        return {"response": response[0]["generated_text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

