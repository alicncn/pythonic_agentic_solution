from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"

persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"
agent = KnowledgeAugmentedPromptAgent(openai_api_key=openai_api_key, persona=persona, knowledge=knowledge)

response = agent.respond(prompt)
print("Agent response using provided knowledge (not its own):")
print(response)
print("\nNotice: The agent uses the incorrect knowledge provided ('London') instead of its inherent knowledge ('Paris'), demonstrating that it prioritizes the given knowledge over its pre-trained understanding.")
