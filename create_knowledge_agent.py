#!/usr/bin/env python3

import os
from azure.search.documents.indexes.models import (
    SearchIndexKnowledgeSource, 
    SearchIndexKnowledgeSourceParameters,
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

def create_knowledge_agent():
    # Load environment variables from .env file
    env_file = ".azure/legal-ai-sky-v1/.env"
    env_vars = load_env_from_file(env_file)
    
    # Configuration from environment variables
    search_service = os.environ.get("AZURE_SEARCH_SERVICE") or env_vars.get("AZURE_SEARCH_SERVICE")
    search_endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT") or env_vars.get("AZURE_SEARCH_ENDPOINT") or f"https://{search_service}.search.windows.net"
    index_name = os.environ.get("AZURE_SEARCH_INDEX") or env_vars.get("AZURE_SEARCH_INDEX", "legal-multi-small")
    agent_name = os.environ.get("AZURE_SEARCH_AGENT") or env_vars.get("AZURE_SEARCH_AGENT", f"{index_name}-agent-upgrade")
    
    # Azure OpenAI configuration
    aoai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") or env_vars.get("AZURE_OPENAI_ENDPOINT", "https://genai-special-projects.openai.azure.com/")
    aoai_deployment = os.environ.get("AZURE_OPENAI_SEARCHAGENT_DEPLOYMENT") or env_vars.get("AZURE_OPENAI_SEARCHAGENT_DEPLOYMENT", "gpt-4.1")
    aoai_model = os.environ.get("AZURE_OPENAI_SEARCHAGENT_MODEL") or env_vars.get("AZURE_OPENAI_SEARCHAGENT_MODEL", "gpt-4.1")
    
    # API version for preview features
    search_api_version = "2025-08-01-preview"
    
    print(f"Creating Knowledge Agent with:")
    print(f"  Search Service: {search_service}")
    print(f"  Search Endpoint: {search_endpoint}")
    print(f"  Search Index: {index_name}")
    print(f"  Agent Name: {agent_name}")
    print(f"  Azure OpenAI Endpoint: {aoai_endpoint}")
    print(f"  Azure OpenAI Deployment: {aoai_deployment}")
    print(f"  Azure OpenAI Model: {aoai_model}")
    
    # Initialize credential and client
    credential = DefaultAzureCredential()
    index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)
    
    try:
        # Step 1: Create Knowledge Source
        knowledge_source_name = index_name  # Use same name as index for simplicity
        
        print(f"\nStep 1: Creating Knowledge Source '{knowledge_source_name}'...")
        
        ks = SearchIndexKnowledgeSource(
            name=knowledge_source_name,
            description=f"Knowledge source for {index_name} search index",
            search_index_parameters=SearchIndexKnowledgeSourceParameters(
                search_index_name=index_name,
                source_data_select="id,sourcepage,sourcefile,content,category,document_type,year,vendor,title,description",  # Include all metadata fields for filtering
            ),
        )
        
        index_client.create_or_update_knowledge_source(
            knowledge_source=ks, 
            api_version=search_api_version
        )
        print(f"✓ Knowledge source '{knowledge_source_name}' created successfully.")
        
        # Step 2: Create Knowledge Agent
        print(f"\nStep 2: Creating Knowledge Agent '{agent_name}'...")
        
        aoai_params = AzureOpenAIVectorizerParameters(
            resource_url=aoai_endpoint,
            deployment_name=aoai_deployment,
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
                    reranker_threshold=2.5,  # Adjust as needed
                    include_references=True,
                    include_reference_source_data=True
                )
            ],
            output_configuration=output_cfg,
        )
        
        index_client.create_or_update_agent(agent, api_version=search_api_version)
        print(f"✓ Knowledge agent '{agent_name}' created successfully.")
        
        print(f"\n🎉 Knowledge Agent setup complete!")
        print(f"Your application should now be able to use agentic retrieval with agent: {agent_name}")
        
    except Exception as e:
        print(f"❌ Error creating knowledge agent: {str(e)}")
        print(f"Make sure you have:")
        print(f"  1. Proper Azure credentials (run 'az login')")
        print(f"  2. Search Contributor role on the Azure AI Search service")
        print(f"  3. Cognitive Services OpenAI User role on the Azure OpenAI service")
        raise

if __name__ == "__main__":
    create_knowledge_agent()
