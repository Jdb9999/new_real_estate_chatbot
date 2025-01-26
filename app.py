import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()

# Load pre-trained model for chatbot
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-large")

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

# Define request body using Pydantic
class UserInput(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Real Estate Chatbot! Ask me anything related to real estate."}

@app.post("/chat/")
def chat(user_input: UserInput):
    question = user_input.question

    # Check if the user's question is in the FAQ
    if question in real_estate_faq:
        return {"response": real_estate_faq[question]}
    
    # If not, generate a response using the chatbot
    try:
        response = chatbot(question, max_length=1000, pad_token_id=50256)
        return {"response": response[0]["generated_text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {e}")

# Make sure app runs on deployment services like Railway
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Use PORT from environment or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)

