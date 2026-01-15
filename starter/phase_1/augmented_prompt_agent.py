from workflow_agents.base_agents import AugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

agent = AugmentedPromptAgent(openai_api_key=openai_api_key, persona=persona)

augmented_agent_response = agent.respond(prompt)

print(augmented_agent_response)

# Print analysis of knowledge source and persona impact
print("\n Analysis:")
print("The agent likely used its pre-trained knowledge about world geography to answer the prompt.")
print("The capital of France (Paris) is part of the model's general knowledge base.")
print("The system prompt with the persona instruction forced the agent to start its response with")
print('"Dear students," giving it a formal, educational tone consistent with a college professor\'s')
print("style of addressing students, rather than providing a plain direct answer.")
