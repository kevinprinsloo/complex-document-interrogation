#!/usr/bin/env python3

import os
from azure.search.documents.indexes.models import (
    KnowledgeAgent, 
    KnowledgeAgentAzureOpenAIModel, 
    KnowledgeSourceReference, 
    AzureOpenAIVectorizerParameters, 
    KnowledgeAgentOutputConfiguration, 
    KnowledgeAgentOutputConfigurationModality
)
from azure.search.documents.indexes import SearchIndexClient
from azure.identity import DefaultAzureCredential

def load_env_from_file(env_file_path):
    """Load environment variables from .env file"""
    env_vars = {}
    try:
        with open(env_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip('"\'')
                    env_vars[key] = value
                    os.environ[key] = value
    except FileNotFoundError:
        print(f"Environment file {env_file_path} not found")
    return env_vars

def fix_knowledge_agent():
    # Load environment variables from .env file
    env_file = ".azure/legal-ai-sky-v1/.env"
    env_vars = load_env_from_file(env_file)
    
    # Configuration
    search_service = os.environ.get("AZURE_SEARCH_SERVICE")
    search_endpoint = f"https://{search_service}.search.windows.net"
    index_name = os.environ.get("AZURE_SEARCH_INDEX", "legal-multi-small")
    agent_name = os.environ.get("AZURE_SEARCH_AGENT", f"{index_name}-agent-upgrade")
    
    # Use the CORRECT deployment name from environment
    aoai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://genai-special-projects.openai.azure.com/")
    aoai_deployment = os.environ.get("AZURE_OPENAI_SEARCHAGENT_DEPLOYMENT", "gpt-5")  # Use env var
    aoai_model = os.environ.get("AZURE_OPENAI_SEARCHAGENT_MODEL", "gpt-5")  # Use env var
    
    # API version for preview features
    search_api_version = "2025-08-01-preview"
    
    print(f"Updating Knowledge Agent with CORRECT deployment:")
    print(f"  Agent Name: {agent_name}")
    print(f"  Azure OpenAI Endpoint: {aoai_endpoint}")
    print(f"  Azure OpenAI Deployment: {aoai_deployment}")
    print(f"  Azure OpenAI Model: {aoai_model}")
    
    # Initialize credential and client
    credential = DefaultAzureCredential()
    index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)
    
    try:
        # Update Knowledge Agent with correct deployment
        knowledge_source_name = index_name
        
        print(f"\nUpdating Knowledge Agent '{agent_name}' with correct deployment...")
        
        aoai_params = AzureOpenAIVectorizerParameters(
            resource_url=aoai_endpoint,
            deployment_name=aoai_deployment,  # This is the key fix
            model_name=aoai_model,
        )
        
        output_cfg = KnowledgeAgentOutputConfiguration(
            modality=KnowledgeAgentOutputConfigurationModality.ANSWER_SYNTHESIS,
            include_activity=True,
        )
        
        agent = KnowledgeAgent(
            name=agent_name,
            models=[KnowledgeAgentAzureOpenAIModel(azure_open_ai_parameters=aoai_params)],
            knowledge_sources=[
                KnowledgeSourceReference(
                    name=knowledge_source_name,
                    reranker_threshold=2.5,
                    include_references=True,
                    include_reference_source_data=True
                )
            ],
            output_configuration=output_cfg,
        )
        
        index_client.create_or_update_agent(agent, api_version=search_api_version)
        print(f"✅ Knowledge agent '{agent_name}' updated successfully with correct deployment!")
        
        print(f"\n🎉 Your application should now work with agentic retrieval!")
        
    except Exception as e:
        print(f"❌ Error updating knowledge agent: {str(e)}")
        raise

if __name__ == "__main__":
    fix_knowledge_agent()
