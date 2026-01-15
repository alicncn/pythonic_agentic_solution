# agentic_workflow.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from phase_1.workflow_agents.base_agents import ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent

from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# load the product spec
script_dir = os.path.dirname(__file__)
product_spec_path = os.path.join(script_dir, "Product-Spec-Email-Router.txt")
with open(product_spec_path, "r") as file:
    product_spec = file.read()

# Instantiate all the agents

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)
action_planning_agent = ActionPlanningAgent(openai_api_key=openai_api_key, knowledge=knowledge_action_planning)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    f"{product_spec}"
)
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key=openai_api_key, persona=persona_product_manager, knowledge=knowledge_product_manager)

# Product Manager - Evaluation Agent
persona_product_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_product_manager = "The answer should be user stories following the structure: As a [type of user], I want [an action or feature] so that [benefit/value]."
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager_eval,
    evaluation_criteria=evaluation_criteria_product_manager,
    worker_agent=product_manager_knowledge_agent,
    max_interactions=10
)

# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager,
    knowledge=knowledge_program_manager
)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."

evaluation_criteria_program_manager = (
    "The answer should be product features that follow the following structure: "
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user"
)
program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager_eval,
    evaluation_criteria=evaluation_criteria_program_manager,
    worker_agent=program_manager_knowledge_agent,
    max_interactions=10
)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer,
    knowledge=knowledge_dev_engineer
)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."

evaluation_criteria_dev_engineer = (
    "The answer should be tasks following this exact structure: "
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first"
)
development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer_eval,
    evaluation_criteria=evaluation_criteria_dev_engineer,
    worker_agent=development_engineer_knowledge_agent,
    max_interactions=10
)


# Job function persona support functions
def product_manager_support_function(query):
    """Support function for product manager that generates and evaluates user stories."""
    result = product_manager_evaluation_agent.evaluate(query)
    return result['final_response']

def program_manager_support_function(query):
    """Support function for program manager that generates and evaluates features."""
    result = program_manager_evaluation_agent.evaluate(query)
    return result['final_response']

def development_engineer_support_function(query):
    """Support function for development engineer that generates and evaluates tasks."""
    result = development_engineer_evaluation_agent.evaluate(query)
    return result['final_response']

# Routing Agent
routing_agent = RoutingAgent(openai_api_key=openai_api_key, agents=[])

routes = [
    {
        "name": "product_manager",
        "description": "Define user stories for a product based on personas, actions, and desired outcomes",
        "func": product_manager_support_function
    },
    {
        "name": "program_manager",
        "description": "Define features by organizing similar user stories into cohesive groups",
        "func": program_manager_support_function
    },
    {
        "name": "development_engineer",
        "description": "Define development tasks needed to implement user stories",
        "func": development_engineer_support_function
    }
]

routing_agent.agents = routes

# Run the workflow

print("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ****
workflow_prompt = "What would the development tasks for this product be?"
# ****
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")

# Extract workflow steps using the action planning agent
workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
print(f"Workflow steps extracted: {workflow_steps}\n")

# Execute workflow by routing each step to appropriate agent
completed_steps = []

for idx, step in enumerate(workflow_steps, 1):
    print(f"\n{'='*80}")
    print(f"Executing Step {idx}: {step}")
    print('='*80)
    
    # Route the step to the appropriate agent
    result = routing_agent.route(step)
    completed_steps.append(result)
    
    print(f"\nStep {idx} completed.")
    print(f"Result preview: {result[:200]}..." if len(result) > 200 else f"Result: {result}")

# Print final workflow output
print(f"\n\n{'='*80}")
print("WORKFLOW COMPLETED")
print('='*80)

# Save the output to a file
output_file = os.path.join(os.path.dirname(__file__), "workflow_output.txt")
with open(output_file, "w", encoding="utf-8") as f:
    f.write("WORKFLOW EXECUTION RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Workflow Prompt: {workflow_prompt}\n\n")
    f.write(f"Workflow Steps Extracted:\n")
    for i, step in enumerate(workflow_steps, 1):
        f.write(f"  {i}. {step}\n")
    f.write("\n" + "="*80 + "\n\n")
    
    for idx, (step, result) in enumerate(zip(workflow_steps, completed_steps), 1):
        f.write(f"Step {idx}: {step}\n")
        f.write("-"*80 + "\n")
        f.write(f"{result}\n\n")
    
    f.write("="*80 + "\n")
    f.write("WORKFLOW COMPLETED\n")

print(f"\nOutput saved to: {output_file}")
print("\nFinal Output (Last Completed Step):")
print(completed_steps[-1] if completed_steps else "No steps completed")