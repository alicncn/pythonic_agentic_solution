# Test script for DirectPromptAgent class

from workflow_agents.base_agents import DirectPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY") 

prompt = "What is the Capital of France?"

direct_agent = DirectPromptAgent(openai_api_key=openai_api_key)
direct_agent_response = direct_agent.respond(prompt) 

# Print the response from the agent
print(direct_agent_response)

print("\nThe agent used its pre-trained knowledge from the GPT-3.5-turbo model to answer this question. The model was trained on a large corpus of text data that includes geographical information about world capitals.")
