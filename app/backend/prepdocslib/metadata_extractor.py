import json
import logging
from typing import Dict, Optional, Any
from openai import AsyncAzureOpenAI, AsyncOpenAI

logger = logging.getLogger("scripts")


class MetadataExtractor:
    """
    Service for extracting metadata from document filenames and content using Azure OpenAI
    """
    
    def __init__(self, openai_client: AsyncAzureOpenAI | AsyncOpenAI, model_name: str, deployment_name: Optional[str] = None):
        self.openai_client = openai_client
        self.model_name = model_name
        self.deployment_name = deployment_name
        
    async def extract_metadata_from_filename(self, filename: str) -> Dict[str, Any]:
        """
        Extract metadata from a filename using AI
        
        Args:
            filename: The filename to analyze
            
        Returns:
            Dictionary containing extracted metadata fields
        """
        system_prompt = """Extract metadata from document filenames. Return only valid JSON with these fields:
        {"title": "document title", "description": "brief description", "category": "business category", "document_type": "document type", "year": "year if found", "vendor": "company name if found"}
        
        Use null for unknown fields. Example:
        {"title": "Service Agreement", "description": "Software subscription agreement", "category": "Legal", "document_type": "Contract", "year": "2016", "vendor": "Anaplan"}"""
        
        user_prompt = f"Extract metadata from this filename: {filename}"
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=200,
                top_p=0.95,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if content:
                # Try to parse JSON response
                try:
                    metadata = json.loads(content)
                    # Ensure all expected fields are present
                    expected_fields = ["title", "description", "category", "document_type", "year", "vendor"]
                    for field in expected_fields:
                        if field not in metadata:
                            metadata[field] = None
                    return metadata
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON response for filename {filename}: {content}")
                    return self._get_default_metadata(filename)
            else:
                logger.warning(f"Empty response from OpenAI for filename {filename}")
                return self._get_default_metadata(filename)
                
        except Exception as e:
            logger.error(f"Error extracting metadata from filename {filename}: {e}")
            return self._get_default_metadata(filename)
    
    async def extract_metadata_from_content(self, filename: str, content_preview: str) -> Dict[str, Any]:
        """
        Extract metadata from document content using AI
        
        Args:
            filename: The filename for context
            content_preview: First few paragraphs of document content
            
        Returns:
            Dictionary containing extracted metadata fields
        """
        system_prompt = """Extract metadata from document content. Return only valid JSON with these fields:
        {"title": "document title", "description": "brief description", "category": "business category", "document_type": "document type", "year": "year if found", "vendor": "company name if found"}
        
        Use null for unknown fields. Focus on content, not just filename."""
        
        user_prompt = f"""Filename: {filename}

Content preview:
{content_preview[:2000]}  # Limit content to avoid token limits

Extract metadata from this document."""
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=300,
                top_p=0.95,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if content:
                try:
                    metadata = json.loads(content)
                    # Ensure all expected fields are present
                    expected_fields = ["title", "description", "category", "document_type", "year", "vendor"]
                    for field in expected_fields:
                        if field not in metadata:
                            metadata[field] = None
                    return metadata
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON response for content analysis of {filename}: {content}")
                    return await self.extract_metadata_from_filename(filename)
            else:
                logger.warning(f"Empty response from OpenAI for content analysis of {filename}")
                return await self.extract_metadata_from_filename(filename)
                
        except Exception as e:
            logger.error(f"Error extracting metadata from content of {filename}: {e}")
            return await self.extract_metadata_from_filename(filename)
    
    def _get_default_metadata(self, filename: str) -> Dict[str, Any]:
        """
        Generate default metadata when AI extraction fails
        """
        # Remove file extension and clean up filename for title
        title = filename.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').title()
        
        return {
            "title": title,
            "description": None,
            "category": None,
            "document_type": None,
            "year": None,
            "vendor": None
        }
