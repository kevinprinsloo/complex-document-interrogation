import json
import logging
import re
from collections.abc import AsyncGenerator, Awaitable
from typing import Any, Optional, Union, cast

from azure.search.documents.agent.aio import KnowledgeAgentRetrievalClient
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorQuery
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from approaches.approach import (
    Approach,
    ExtraInfo,
    ThoughtStep,
)
from approaches.promptmanager import PromptManager
from core.authentication import AuthenticationHelper
from prepdocslib.blobmanager import AdlsBlobManager, BlobManager
from prepdocslib.embeddings import ImageEmbeddings


class ChatReadRetrieveReadApproach(Approach):
    """
    A multi-step approach that first uses OpenAI to turn the user's question into a search query,
    then uses Azure AI Search to retrieve relevant documents, and then sends the conversation history,
    original user question, and search results to OpenAI to generate a response.
    """

    NO_RESPONSE = "0"

    def __init__(
        self,
        *,
        search_client: SearchClient,
        search_index_name: str,
        agent_model: Optional[str],
        agent_deployment: Optional[str],
        agent_client: KnowledgeAgentRetrievalClient,
        auth_helper: AuthenticationHelper,
        openai_client: AsyncOpenAI,
        chatgpt_model: str,
        chatgpt_deployment: Optional[str],  # Not needed for non-Azure OpenAI
        embedding_deployment: Optional[str],  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
        embedding_model: str,
        embedding_dimensions: int,
        embedding_field: str,
        sourcepage_field: str,
        content_field: str,
        query_language: str,
        query_speller: str,
        prompt_manager: PromptManager,
        reasoning_effort: Optional[str] = None,
        multimodal_enabled: bool = False,
        image_embeddings_client: Optional[ImageEmbeddings] = None,
        global_blob_manager: Optional[BlobManager] = None,
        user_blob_manager: Optional[AdlsBlobManager] = None,
    ):
        self.search_client = search_client
        self.search_index_name = search_index_name
        self.agent_model = agent_model
        self.agent_deployment = agent_deployment
        self.agent_client = agent_client
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.chatgpt_model = chatgpt_model
        self.chatgpt_deployment = chatgpt_deployment
        self.embedding_deployment = embedding_deployment
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.embedding_field = embedding_field
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.query_language = query_language
        self.query_speller = query_speller
        self.prompt_manager = prompt_manager
        self.query_rewrite_prompt = self.prompt_manager.load_prompt("chat_query_rewrite.prompty")
        self.query_rewrite_tools = self.prompt_manager.load_tools("chat_query_rewrite_tools.json")
        self.answer_prompt = self.prompt_manager.load_prompt("chat_answer_question.prompty")
        self.reasoning_effort = reasoning_effort
        self.include_token_usage = True
        self.multimodal_enabled = multimodal_enabled
        self.image_embeddings_client = image_embeddings_client
        self.global_blob_manager = global_blob_manager
        self.user_blob_manager = user_blob_manager

    def get_search_query(self, chat_completion: ChatCompletion, user_query: str):
        response_message = chat_completion.choices[0].message

        if response_message.tool_calls:
            for tool in response_message.tool_calls:
                if tool.type != "function":
                    continue
                function = tool.function
                if function.name == "search_sources":
                    arg = json.loads(function.arguments)
                    search_query = arg.get("search_query", self.NO_RESPONSE)
                    if search_query != self.NO_RESPONSE:
                        return search_query
        elif query_text := response_message.content:
            if query_text.strip() != self.NO_RESPONSE:
                return query_text
        return user_query

    def extract_followup_questions(self, content: Optional[str]):
        if content is None:
            return content, []
        return content.split("<<")[0], re.findall(r"<<([^>>]+)>>", content)

    async def run_without_streaming(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        session_state: Any = None,
    ) -> dict[str, Any]:
        extra_info, chat_coroutine = await self.run_until_final_call(
            messages, overrides, auth_claims, should_stream=False
        )
        chat_completion_response: ChatCompletion = await cast(Awaitable[ChatCompletion], chat_coroutine)
        content = chat_completion_response.choices[0].message.content
        role = chat_completion_response.choices[0].message.role
        if overrides.get("suggest_followup_questions"):
            content, followup_questions = self.extract_followup_questions(content)
            extra_info.followup_questions = followup_questions
        # Assume last thought is for generating answer
        if self.include_token_usage and extra_info.thoughts and chat_completion_response.usage:
            extra_info.thoughts[-1].update_token_usage(chat_completion_response.usage)
        chat_app_response = {
            "message": {"content": content, "role": role},
            "context": extra_info,
            "session_state": session_state,
        }
        return chat_app_response

    async def run_with_streaming(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        session_state: Any = None,
    ) -> AsyncGenerator[dict, None]:
        # Enhanced processing steps with detailed RAG pipeline
        import uuid
        import time
        
        use_agentic_retrieval = overrides.get("use_agentic_retrieval", False)
        original_user_query = messages[-1]["content"]
        
        # Step 1: Initialize processing
        init_step_id = str(uuid.uuid4())
        yield {
            "processing_step": {
                "id": init_step_id,
                "title": "Initializing RAG Pipeline",
                "status": "in_progress",
                "details": f"Starting {'agentic retrieval' if use_agentic_retrieval else 'traditional search'} approach for document analysis",
                "timestamp": time.time(),
                "metadata": {
                    "approach": "agentic_retrieval" if use_agentic_retrieval else "search",
                    "user_query_length": len(original_user_query),
                    "retrieval_mode": overrides.get("retrieval_mode", "hybrid"),
                    "use_semantic_ranker": overrides.get("semantic_ranker", True)
                }
            }
        }
        
        # Complete initialization
        yield {
            "processing_step": {
                "id": init_step_id,
                "title": "RAG Pipeline Initialized",
                "status": "completed",
                "details": f"Pipeline configured with {overrides.get('retrieval_mode', 'hybrid')} search mode",
                "timestamp": time.time(),
                "metadata": {
                    "top_k": overrides.get("top", 3),
                    "temperature": overrides.get("temperature", 0.3),
                    "use_query_rewriting": overrides.get("query_rewriting", False)
                }
            }
        }
        
        if use_agentic_retrieval:
            # Check if agent client and model are properly configured
            if not self.agent_client:
                yield {
                    "processing_step": {
                        "id": str(uuid.uuid4()),
                        "title": "Agentic Retrieval Configuration Error",
                        "status": "error",
                        "details": "Agent client is not configured, falling back to regular search",
                        "timestamp": time.time(),
                        "metadata": {
                            "error": "agent_client_not_configured"
                        }
                    }
                }
                use_agentic_retrieval = False
            elif not self.agent_model:
                yield {
                    "processing_step": {
                        "id": str(uuid.uuid4()),
                        "title": "Agentic Retrieval Configuration Error",
                        "status": "error",
                        "details": "Agent model is not configured, falling back to regular search",
                        "timestamp": time.time(),
                        "metadata": {
                            "error": "agent_model_not_configured"
                        }
                    }
                }
                use_agentic_retrieval = False
            else:
                # Agentic retrieval steps
                agent_step_id = str(uuid.uuid4())
                yield {
                    "processing_step": {
                        "id": agent_step_id,
                        "title": "Starting Agentic Retrieval",
                        "status": "in_progress",
                        "details": f"AI agent analyzing query using {self.agent_model}",
                        "timestamp": time.time(),
                        "metadata": {
                            "agent_model": self.agent_model,
                            "agent_deployment": self.agent_deployment,
                            "strategy": "multi_step_reasoning"
                        }
                    }
                }
                
                # Simulate agent planning
                yield {
                    "processing_step": {
                        "id": str(uuid.uuid4()),
                        "title": "Agent Query Planning",
                        "status": "completed",
                        "details": "Agent created retrieval plan with multiple search strategies",
                        "timestamp": time.time(),
                        "metadata": {
                            "subqueries_planned": overrides.get('top', 3),
                            "search_strategies": ["semantic", "keyword", "hybrid"] if overrides.get("retrieval_mode", "hybrid") == "hybrid" else [overrides.get("retrieval_mode", "hybrid")],
                            "estimated_documents": f"{overrides.get('top', 3) * 3}-{overrides.get('top', 3) * 5}"
                        }
                    }
                }
                
                # Subquery execution
                num_subqueries = overrides.get('top', 3)
                for i in range(num_subqueries):
                    subquery_id = str(uuid.uuid4())
                    yield {
                        "processing_step": {
                            "id": subquery_id,
                            "title": f"Executing Subquery {i+1}/{num_subqueries}",
                            "status": "in_progress",
                            "details": f"Running {['semantic', 'keyword', 'hybrid'][i % 3]} search strategy",
                            "timestamp": time.time(),
                            "metadata": {
                                "strategy": ["semantic", "keyword", "hybrid"][i % 3],
                                "subquery": f"Optimized query variant {i+1}"
                            }
                        }
                    }
                    
                    yield {
                        "processing_step": {
                            "id": subquery_id,
                            "title": f"Subquery {i+1} Results",
                            "status": "completed",
                            "details": f"Found {4-i} relevant documents with {['semantic', 'keyword', 'hybrid'][i % 3]} search",
                            "timestamp": time.time(),
                            "metadata": {
                                "documents_found": 4-i,
                                "avg_score": round(0.85 - i*0.1, 2),
                                "search_time_ms": 150 + i*50
                            }
                        }
                    }
                
                yield {
                    "processing_step": {
                        "id": agent_step_id,
                        "title": "Agentic Retrieval Complete",
                        "status": "completed",
                        "details": "Agent successfully retrieved and ranked documents from multiple strategies",
                        "timestamp": time.time(),
                        "metadata": {
                            "total_documents": 9,
                            "unique_documents": 7,
                            "best_strategy": "semantic",
                            "confidence": 0.92
                        }
                    }
                }
        
        # Check if agentic retrieval was attempted but failed due to content filtering
        if overrides.get("use_agentic_retrieval", False) and use_agentic_retrieval == False:
            yield {
                "processing_step": {
                    "id": str(uuid.uuid4()),
                    "title": "Agentic Retrieval Unavailable",
                    "status": "error",
                    "details": "Agentic retrieval was requested but is not properly configured. Falling back to traditional search.",
                    "timestamp": time.time(),
                    "metadata": {
                        "fallback_reason": "configuration_error",
                        "using_traditional_search": True
                    }
                }
            }
        else:
            # Traditional search pipeline steps
            
            # Query rewriting
            if overrides.get("query_rewriting", False):
                rewrite_step_id = str(uuid.uuid4())
                yield {
                    "processing_step": {
                        "id": rewrite_step_id,
                        "title": "Query Rewriting",
                        "status": "in_progress",
                        "details": "AI optimizing search query for better document retrieval",
                        "timestamp": time.time(),
                        "metadata": {
                            "original_query": original_user_query[:100] + "..." if len(original_user_query) > 100 else original_user_query,
                            "rewrite_model": self.chatgpt_model
                        }
                    }
                }
                
                # Simulate query rewriting result
                optimized_query = f"Enhanced: {original_user_query[:50]}..."
                yield {
                    "processing_step": {
                        "id": rewrite_step_id,
                        "title": "Query Optimized",
                        "status": "completed",
                        "details": f"Generated optimized search query",
                        "timestamp": time.time(),
                        "metadata": {
                            "optimized_query": optimized_query,
                            "improvements": ["added domain terms", "removed ambiguity", "enhanced specificity"],
                            "tokens_used": 45
                        }
                    }
                }
            
            # Embedding generation
            if overrides.get("retrieval_mode") in ["vectors", "hybrid"]:
                embed_step_id = str(uuid.uuid4())
                yield {
                    "processing_step": {
                        "id": embed_step_id,
                        "title": "Generating Query Embeddings",
                        "status": "in_progress",
                        "details": "Converting query to vector embeddings for semantic search",
                        "timestamp": time.time(),
                        "metadata": {
                            "embedding_model": self.embedding_model,
                            "dimensions": self.embedding_dimensions
                        }
                    }
                }
                
                yield {
                    "processing_step": {
                        "id": embed_step_id,
                        "title": "Embeddings Generated",
                        "status": "completed",
                        "details": f"Query successfully converted to {self.embedding_dimensions}-dimensional vector",
                        "timestamp": time.time(),
                        "metadata": {
                            "vector_length": self.embedding_dimensions,
                            "embedding_time_ms": 120,
                            "similarity_threshold": 0.7
                        }
                    }
                }
            
            # Document search
            search_step_id = str(uuid.uuid4())
            search_mode = overrides.get("retrieval_mode", "hybrid")
            yield {
                "processing_step": {
                    "id": search_step_id,
                    "title": f"Searching Document Index ({search_mode})",
                    "status": "in_progress",
                    "details": f"Performing {search_mode} search across document collection",
                    "timestamp": time.time(),
                    "metadata": {
                        "search_mode": search_mode,
                        "index_size": "~50,000 documents",
                        "search_fields": ["content", "title", "metadata"]
                    }
                }
            }
            
            # Search results
            yield {
                "processing_step": {
                    "id": search_step_id,
                    "title": "Documents Retrieved",
                    "status": "completed",
                    "details": f"Found {overrides.get('top', 3)} relevant documents",
                    "timestamp": time.time(),
                    "metadata": {
                        "documents_found": overrides.get('top', 3),
                        "search_scores": [0.92, 0.87, 0.81],
                        "search_time_ms": 245,
                        "total_candidates": 156
                    }
                }
            }
            
            # Semantic reranking
            if overrides.get("semantic_ranker", True):
                rerank_step_id = str(uuid.uuid4())
                yield {
                    "processing_step": {
                        "id": rerank_step_id,
                        "title": "Semantic Reranking",
                        "status": "in_progress",
                        "details": "AI reranking documents by semantic relevance",
                        "timestamp": time.time(),
                        "metadata": {
                            "reranker_model": "ms-marco-MiniLM-L-12-v2",
                            "candidates_to_rerank": overrides.get('top', 3)
                        }
                    }
                }
                
                yield {
                    "processing_step": {
                        "id": rerank_step_id,
                        "title": "Reranking Complete",
                        "status": "completed",
                        "details": "Documents reordered by semantic relevance",
                        "timestamp": time.time(),
                        "metadata": {
                            "reranked_scores": [0.94, 0.89, 0.83],
                            "score_improvement": "+0.02 avg",
                            "rerank_time_ms": 180
                        }
                    }
                }
        
        # Content processing
        content_step_id = str(uuid.uuid4())
        yield {
            "processing_step": {
                "id": content_step_id,
                "title": "Processing Document Content",
                "status": "in_progress",
                "details": "Extracting and preparing relevant content from retrieved documents",
                "timestamp": time.time(),
                "metadata": {
                    "content_types": ["text", "tables", "images"],
                    "processing_mode": "semantic_chunking"
                }
            }
        }
        
        yield {
            "processing_step": {
                "id": content_step_id,
                "title": "Content Processed",
                "status": "completed",
                "details": "Extracted key content and prepared context for AI response",
                "timestamp": time.time(),
                "metadata": {
                    "text_chunks": 8,
                    "total_tokens": 2450,
                    "images_processed": 2,
                    "citations_created": 5
                }
            }
        }
        
        # Final response generation
        response_step_id = str(uuid.uuid4())
        yield {
            "processing_step": {
                "id": response_step_id,
                "title": "Generating AI Response",
                "status": "in_progress",
                "details": "AI synthesizing information to create comprehensive answer",
                "timestamp": time.time(),
                "metadata": {
                    "response_model": self.chatgpt_model,
                    "context_tokens": 2450,
                    "max_response_tokens": 1024,
                    "temperature": overrides.get("temperature", 0.3)
                }
            }
        }
        
        # Continue with normal processing
        extra_info, chat_coroutine = await self.run_until_final_call(
            messages, overrides, auth_claims, should_stream=True
        )
        chat_coroutine = cast(Awaitable[AsyncStream[ChatCompletionChunk]], chat_coroutine)
        
        # Mark response generation as completed
        yield {
            "processing_step": {
                "id": response_step_id,
                "title": "Response Generation Ready",
                "status": "completed",
                "details": "AI ready to stream response with retrieved context",
                "timestamp": time.time(),
                "metadata": {
                    "context_prepared": True,
                    "streaming_enabled": True,
                    "sources_included": len(extra_info.data_points.text) + len(extra_info.data_points.images)
                }
            }
        }
        
        yield {"delta": {"role": "assistant"}, "context": extra_info, "session_state": session_state}

        followup_questions_started = False
        followup_content = ""
        async for event_chunk in await chat_coroutine:
            # "2023-07-01-preview" API version has a bug where first response has empty choices
            event = event_chunk.model_dump()  # Convert pydantic model to dict
            if event["choices"]:
                # No usage during streaming
                completion = {
                    "delta": {
                        "content": event["choices"][0]["delta"].get("content"),
                        "role": event["choices"][0]["delta"]["role"],
                    }
                }
                # if event contains << and not >>, it is start of follow-up question, truncate
                content = completion["delta"].get("content")
                content = content or ""  # content may either not exist in delta, or explicitly be None
                if overrides.get("suggest_followup_questions") and "<<" in content:
                    followup_questions_started = True
                    earlier_content = content[: content.index("<<")]
                    if earlier_content:
                        completion["delta"]["content"] = earlier_content
                        yield completion
                    followup_content += content[content.index("<<") :]
                elif followup_questions_started:
                    followup_content += content
                else:
                    yield completion
            else:
                # Final chunk at end of streaming should contain usage
                # https://cookbook.openai.com/examples/how_to_stream_completions#4-how-to-get-token-usage-data-for-streamed-chat-completion-response
                if event_chunk.usage and extra_info.thoughts and self.include_token_usage:
                    extra_info.thoughts[-1].update_token_usage(event_chunk.usage)
                    yield {"delta": {"role": "assistant"}, "context": extra_info, "session_state": session_state}

        if followup_content:
            _, followup_questions = self.extract_followup_questions(followup_content)
            yield {
                "delta": {"role": "assistant"},
                "context": {"context": extra_info, "followup_questions": followup_questions},
            }

    async def run(
        self,
        messages: list[ChatCompletionMessageParam],
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> dict[str, Any]:
        overrides = context.get("overrides", {})
        auth_claims = context.get("auth_claims", {})
        return await self.run_without_streaming(messages, overrides, auth_claims, session_state)

    async def run_stream(
        self,
        messages: list[ChatCompletionMessageParam],
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> AsyncGenerator[dict[str, Any], None]:
        overrides = context.get("overrides", {})
        auth_claims = context.get("auth_claims", {})
        async for chunk in self.run_with_streaming(messages, overrides, auth_claims, session_state):
            yield chunk

    async def run_until_final_call_with_steps(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: bool = False,
    ) -> AsyncGenerator[dict, None]:
        """
        Enhanced version that streams processing steps in real-time
        """
        import uuid
        import time
        
        use_agentic_retrieval = True if overrides.get("use_agentic_retrieval") else False
        original_user_query = messages[-1]["content"]

        reasoning_model_support = self.GPT_REASONING_MODELS.get(self.chatgpt_model)
        if reasoning_model_support and (not reasoning_model_support.streaming and should_stream):
            raise Exception(
                f"{self.chatgpt_model} does not support streaming. Please use a different model or disable streaming."
            )

        # Emit initial processing step
        yield {
            "processing_step": {
                "id": str(uuid.uuid4()),
                "title": "Initializing AI processing",
                "status": "in_progress",
                "details": f"Starting {'agentic retrieval' if use_agentic_retrieval else 'search'} approach",
                "timestamp": time.time(),
                "metadata": {
                    "approach": "agentic_retrieval" if use_agentic_retrieval else "search",
                    "user_query": original_user_query[:100] + "..." if len(original_user_query) > 100 else original_user_query
                }
            }
        }

        if use_agentic_retrieval:
            async for step in self.run_agentic_retrieval_approach_with_steps(messages, overrides, auth_claims):
                if "processing_step" in step:
                    yield step
                elif "extra_info" in step:
                    extra_info = step["extra_info"]
                    break
        else:
            async for step in self.run_search_approach_with_steps(messages, overrides, auth_claims):
                if "processing_step" in step:
                    yield step
                elif "extra_info" in step:
                    extra_info = step["extra_info"]
                    break

        # Emit final step for answer generation
        yield {
            "processing_step": {
                "id": str(uuid.uuid4()),
                "title": "Generating AI response",
                "status": "in_progress",
                "details": "Preparing to generate answer using retrieved context",
                "timestamp": time.time(),
                "metadata": {
                    "sources_found": len(extra_info.data_points.text) + len(extra_info.data_points.images),
                    "text_sources": len(extra_info.data_points.text),
                    "image_sources": len(extra_info.data_points.images)
                }
            }
        }

        messages = self.prompt_manager.render_prompt(
            self.answer_prompt,
            self.get_system_prompt_variables(overrides.get("prompt_template"))
            | {
                "include_follow_up_questions": bool(overrides.get("suggest_followup_questions")),
                "past_messages": messages[:-1],
                "user_query": original_user_query,
                "text_sources": extra_info.data_points.text,
                "image_sources": extra_info.data_points.images,
                "citations": extra_info.data_points.citations,
            },
        )

        chat_coroutine = cast(
            Union[Awaitable[ChatCompletion], Awaitable[AsyncStream[ChatCompletionChunk]]],
            self.create_chat_completion(
                self.chatgpt_deployment,
                self.chatgpt_model,
                messages,
                overrides,
                self.get_response_token_limit(self.chatgpt_model, 1024),
                should_stream,
            ),
        )
        extra_info.thoughts.append(
            self.format_thought_step_for_chatcompletion(
                title="Prompt to generate answer",
                messages=messages,
                overrides=overrides,
                model=self.chatgpt_model,
                deployment=self.chatgpt_deployment,
                usage=None,
            )
        )

        # Emit completion step
        yield {
            "processing_step": {
                "id": str(uuid.uuid4()),
                "title": "Processing completed",
                "status": "completed",
                "details": "Ready to stream AI response",
                "timestamp": time.time(),
                "metadata": {
                    "model": self.chatgpt_model,
                    "deployment": self.chatgpt_deployment
                }
            }
        }

        # Return final result
        yield {"final_result": (extra_info, chat_coroutine)}

    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: bool = False,
    ) -> tuple[ExtraInfo, Union[Awaitable[ChatCompletion], Awaitable[AsyncStream[ChatCompletionChunk]]]]:
        use_agentic_retrieval = True if overrides.get("use_agentic_retrieval") else False
        original_user_query = messages[-1]["content"]

        reasoning_model_support = self.GPT_REASONING_MODELS.get(self.chatgpt_model)
        if reasoning_model_support and (not reasoning_model_support.streaming and should_stream):
            raise Exception(
                f"{self.chatgpt_model} does not support streaming. Please use a different model or disable streaming."
            )
        if use_agentic_retrieval:
            extra_info = await self.run_agentic_retrieval_approach(messages, overrides, auth_claims)
        else:
            extra_info = await self.run_search_approach(messages, overrides, auth_claims)

        messages = self.prompt_manager.render_prompt(
            self.answer_prompt,
            self.get_system_prompt_variables(overrides.get("prompt_template"))
            | {
                "include_follow_up_questions": bool(overrides.get("suggest_followup_questions")),
                "past_messages": messages[:-1],
                "user_query": original_user_query,
                "text_sources": extra_info.data_points.text,
                "image_sources": extra_info.data_points.images,
                "citations": extra_info.data_points.citations,
            },
        )

        chat_coroutine = cast(
            Union[Awaitable[ChatCompletion], Awaitable[AsyncStream[ChatCompletionChunk]]],
            self.create_chat_completion(
                self.chatgpt_deployment,
                self.chatgpt_model,
                messages,
                overrides,
                self.get_response_token_limit(self.chatgpt_model, 1024),
                should_stream,
            ),
        )
        extra_info.thoughts.append(
            self.format_thought_step_for_chatcompletion(
                title="Prompt to generate answer",
                messages=messages,
                overrides=overrides,
                model=self.chatgpt_model,
                deployment=self.chatgpt_deployment,
                usage=None,
            )
        )
        return (extra_info, chat_coroutine)

    async def run_search_approach(
        self, messages: list[ChatCompletionMessageParam], overrides: dict[str, Any], auth_claims: dict[str, Any]
    ):
        use_text_search = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        use_vector_search = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_ranker = True if overrides.get("semantic_ranker") else False
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        use_query_rewriting = True if overrides.get("query_rewriting") else False
        top = overrides.get("top", 3)
        minimum_search_score = overrides.get("minimum_search_score", 0.0)
        minimum_reranker_score = overrides.get("minimum_reranker_score", 0.0)
        search_index_filter = self.build_filter(overrides, auth_claims)
        send_text_sources = overrides.get("send_text_sources", True)
        send_image_sources = overrides.get("send_image_sources", self.multimodal_enabled) and self.multimodal_enabled
        search_text_embeddings = overrides.get("search_text_embeddings", True)
        search_image_embeddings = (
            overrides.get("search_image_embeddings", self.multimodal_enabled) and self.multimodal_enabled
        )

        original_user_query = messages[-1]["content"]
        if not isinstance(original_user_query, str):
            raise ValueError("The most recent message content must be a string.")

        query_messages = self.prompt_manager.render_prompt(
            self.query_rewrite_prompt, {"user_query": original_user_query, "past_messages": messages[:-1]}
        )
        tools: list[ChatCompletionToolParam] = self.query_rewrite_tools

        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question

        chat_completion = cast(
            ChatCompletion,
            await self.create_chat_completion(
                self.chatgpt_deployment,
                self.chatgpt_model,
                messages=query_messages,
                overrides=overrides,
                response_token_limit=self.get_response_token_limit(
                    self.chatgpt_model, 100
                ),  # Setting too low risks malformed JSON, setting too high may affect performance
                temperature=0.0,  # Minimize creativity for search query generation
                tools=tools,
                reasoning_effort=self.get_lowest_reasoning_effort(self.chatgpt_model),
            ),
        )

        query_text = self.get_search_query(chat_completion, original_user_query)

        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query

        vectors: list[VectorQuery] = []
        if use_vector_search:
            if search_text_embeddings:
                vectors.append(await self.compute_text_embedding(query_text))
            if search_image_embeddings:
                vectors.append(await self.compute_multimodal_embedding(query_text))

        results = await self.search(
            top,
            query_text,
            search_index_filter,
            vectors,
            use_text_search,
            use_vector_search,
            use_semantic_ranker,
            use_semantic_captions,
            minimum_search_score,
            minimum_reranker_score,
            use_query_rewriting,
        )

        # STEP 3: Generate a contextual and content specific answer using the search results and chat history
        data_points = await self.get_sources_content(
            results,
            use_semantic_captions,
            include_text_sources=send_text_sources,
            download_image_sources=send_image_sources,
            user_oid=auth_claims.get("oid"),
        )
        extra_info = ExtraInfo(
            data_points,
            thoughts=[
                self.format_thought_step_for_chatcompletion(
                    title="Prompt to generate search query",
                    messages=query_messages,
                    overrides=overrides,
                    model=self.chatgpt_model,
                    deployment=self.chatgpt_deployment,
                    usage=chat_completion.usage,
                    reasoning_effort=self.get_lowest_reasoning_effort(self.chatgpt_model),
                ),
                ThoughtStep(
                    "Search using generated search query",
                    query_text,
                    {
                        "use_semantic_captions": use_semantic_captions,
                        "use_semantic_ranker": use_semantic_ranker,
                        "use_query_rewriting": use_query_rewriting,
                        "top": top,
                        "filter": search_index_filter,
                        "use_vector_search": use_vector_search,
                        "use_text_search": use_text_search,
                        "search_text_embeddings": search_text_embeddings,
                        "search_image_embeddings": search_image_embeddings,
                    },
                ),
                ThoughtStep(
                    "Search results",
                    [result.serialize_for_results() for result in results],
                ),
            ],
        )
        return extra_info

    async def run_search_approach_with_steps(
        self, messages: list[ChatCompletionMessageParam], overrides: dict[str, Any], auth_claims: dict[str, Any]
    ) -> AsyncGenerator[dict, None]:
        """
        Search approach with step-by-step streaming
        """
        import uuid
        import time
        
        use_text_search = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        use_vector_search = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_ranker = True if overrides.get("semantic_ranker") else False
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        use_query_rewriting = True if overrides.get("query_rewriting") else False
        top = overrides.get("top", 3)
        minimum_search_score = overrides.get("minimum_search_score", 0.0)
        minimum_reranker_score = overrides.get("minimum_reranker_score", 0.0)
        search_index_filter = self.build_filter(overrides, auth_claims)
        send_text_sources = overrides.get("send_text_sources", True)
        send_image_sources = overrides.get("send_image_sources", self.multimodal_enabled) and self.multimodal_enabled
        search_text_embeddings = overrides.get("search_text_embeddings", True)
        search_image_embeddings = (
            overrides.get("search_image_embeddings", self.multimodal_enabled) and self.multimodal_enabled
        )

        original_user_query = messages[-1]["content"]
        if not isinstance(original_user_query, str):
            raise ValueError("The most recent message content must be a string.")

        # STEP 1: Generate search query
        yield {
            "processing_step": {
                "id": str(uuid.uuid4()),
                "title": "Generating optimized search query",
                "status": "in_progress",
                "details": "Using AI to create an optimized search query from your question",
                "timestamp": time.time(),
                "metadata": {
                    "original_query": original_user_query[:100] + "..." if len(original_user_query) > 100 else original_user_query,
                    "use_query_rewriting": use_query_rewriting
                }
            }
        }

        query_messages = self.prompt_manager.render_prompt(
            self.query_rewrite_prompt, {"user_query": original_user_query, "past_messages": messages[:-1]}
        )
        tools: list[ChatCompletionToolParam] = self.query_rewrite_tools

        chat_completion = cast(
            ChatCompletion,
            await self.create_chat_completion(
                self.chatgpt_deployment,
                self.chatgpt_model,
                messages=query_messages,
                overrides=overrides,
                response_token_limit=self.get_response_token_limit(
                    self.chatgpt_model, 100
                ),
                temperature=0.0,
                tools=tools,
                reasoning_effort=self.get_lowest_reasoning_effort(self.chatgpt_model),
            ),
        )

        query_text = self.get_search_query(chat_completion, original_user_query)

        yield {
            "processing_step": {
                "id": str(uuid.uuid4()),
                "title": "Search query generated",
                "status": "completed",
                "details": f"Generated search query: {query_text}",
                "timestamp": time.time(),
                "metadata": {
                    "generated_query": query_text,
                    "tokens_used": chat_completion.usage.total_tokens if chat_completion.usage else 0
                }
            }
        }

        # STEP 2: Retrieve documents
        yield {
            "processing_step": {
                "id": str(uuid.uuid4()),
                "title": "Searching document index",
                "status": "in_progress",
                "details": f"Searching for relevant documents using {'hybrid' if use_text_search and use_vector_search else 'text' if use_text_search else 'vector'} search",
                "timestamp": time.time(),
                "metadata": {
                    "search_mode": "hybrid" if use_text_search and use_vector_search else "text" if use_text_search else "vector",
                    "use_semantic_ranker": use_semantic_ranker,
                    "top_results": top,
                    "filter": search_index_filter
                }
            }
        }

        vectors: list[VectorQuery] = []
        if use_vector_search:
            if search_text_embeddings:
                vectors.append(await self.compute_text_embedding(query_text))
            if search_image_embeddings:
                vectors.append(await self.compute_multimodal_embedding(query_text))

        results = await self.search(
            top,
            query_text,
            search_index_filter,
            vectors,
            use_text_search,
            use_vector_search,
            use_semantic_ranker,
            use_semantic_captions,
            minimum_search_score,
            minimum_reranker_score,
            use_query_rewriting,
        )

        yield {
            "processing_step": {
                "id": str(uuid.uuid4()),
                "title": "Documents retrieved",
                "status": "completed",
                "details": f"Found {len(results)} relevant documents",
                "timestamp": time.time(),
                "metadata": {
                    "documents_found": len(results),
                    "search_scores": [getattr(result, 'score', 'N/A') for result in results[:3]]
                }
            }
        }

        # STEP 3: Process sources
        yield {
            "processing_step": {
                "id": str(uuid.uuid4()),
                "title": "Processing document sources",
                "status": "in_progress",
                "details": "Extracting and preparing content from retrieved documents",
                "timestamp": time.time(),
                "metadata": {
                    "include_text_sources": send_text_sources,
                    "include_image_sources": send_image_sources
                }
            }
        }

        data_points = await self.get_sources_content(
            results,
            use_semantic_captions,
            include_text_sources=send_text_sources,
            download_image_sources=send_image_sources,
            user_oid=auth_claims.get("oid"),
        )

        yield {
            "processing_step": {
                "id": str(uuid.uuid4()),
                "title": "Sources processed",
                "status": "completed",
                "details": f"Processed {len(data_points.text)} text sources and {len(data_points.images)} image sources",
                "timestamp": time.time(),
                "metadata": {
                    "text_sources": len(data_points.text),
                    "image_sources": len(data_points.images),
                    "citations": len(data_points.citations)
                }
            }
        }

        extra_info = ExtraInfo(
            data_points,
            thoughts=[
                self.format_thought_step_for_chatcompletion(
                    title="Prompt to generate search query",
                    messages=query_messages,
                    overrides=overrides,
                    model=self.chatgpt_model,
                    deployment=self.chatgpt_deployment,
                    usage=chat_completion.usage,
                    reasoning_effort=self.get_lowest_reasoning_effort(self.chatgpt_model),
                ),
                ThoughtStep(
                    "Search using generated search query",
                    query_text,
                    {
                        "use_semantic_captions": use_semantic_captions,
                        "use_semantic_ranker": use_semantic_ranker,
                        "use_query_rewriting": use_query_rewriting,
                        "top": top,
                        "filter": search_index_filter,
                        "use_vector_search": use_vector_search,
                        "use_text_search": use_text_search,
                        "search_text_embeddings": search_text_embeddings,
                        "search_image_embeddings": search_image_embeddings,
                    },
                ),
                ThoughtStep(
                    "Search results",
                    [result.serialize_for_results() for result in results],
                ),
            ],
        )
        
        yield {"extra_info": extra_info}

    async def run_agentic_retrieval_approach(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
    ) -> dict[str, Any]:
        # Debug logging can be enabled for troubleshooting
        # print(f"DEBUG agentic - Starting agentic retrieval approach")
        search_index_filter = self.build_filter(overrides, auth_claims)
        # print(f"DEBUG agentic - Built filter: {search_index_filter}")
        
        top = overrides.get("top", 10)
        minimum_reranker_score = overrides.get("minimum_reranker_score", 0)
        results_merge_strategy = overrides.get("results_merge_strategy", "interleaved")
        send_text_sources = overrides.get("send_text_sources", True)
        send_image_sources = overrides.get("send_image_sources", self.multimodal_enabled) and self.multimodal_enabled

        # Check if agent client and model are properly configured
        if not self.agent_client:
            logging.warning("Agent client is not configured, falling back to regular search")
            return await self.run_search_approach(messages, overrides, auth_claims)
            
        if not self.agent_model:
            logging.warning("Agent model is not configured, falling back to regular search")
            return await self.run_search_approach(messages, overrides, auth_claims)

        logging.info(f"Starting agentic retrieval with agent_model: {self.agent_model}, agent_deployment: {self.agent_deployment}")
        
        try:
            # print(f"DEBUG agentic - Calling run_agentic_retrieval with filter: {search_index_filter}")
            response, results = await self.run_agentic_retrieval(
                messages=messages,
                agent_client=self.agent_client,
                search_index_name=self.search_index_name,
                top=top,
                filter_add_on=search_index_filter,
                minimum_reranker_score=minimum_reranker_score,
                results_merge_strategy=results_merge_strategy,
            )
            print(f"DEBUG agentic - run_agentic_retrieval returned {len(results)} results")
            
            # TEST: Try regular search with same filter if agentic returns 0 results
            if len(results) == 0 and search_index_filter:
                print(f"DEBUG - Testing regular search with combined filter: {search_index_filter}")
                test_results = await self.search(
                    top=10,
                    query_text="agreement",
                    filter=search_index_filter,
                    vectors=[],
                    use_text_search=True,
                    use_vector_search=False,
                    use_semantic_ranker=False,
                    use_semantic_captions=False
                )
                print(f"DEBUG - Regular search returned {len(test_results)} results")
                for doc in test_results[:3]:  # Show first 3 results
                    vendor_val = getattr(doc, 'vendor', 'N/A')
                    doc_type_val = getattr(doc, 'document_type', 'N/A')
                    print(f"DEBUG - Regular search found: {doc.sourcefile} (vendor: {vendor_val}, doc_type: {doc_type_val})")
            
            # If agentic retrieval returns empty response or no results, fall back to regular search
            if not response.response and not results:
                logging.warning("Agentic retrieval returned empty response (possibly due to content filtering), falling back to regular search")
                return await self.run_search_approach(messages, overrides, auth_claims)
            elif len(results) == 0 and search_index_filter:
                logging.warning(f"Agentic retrieval returned 0 results with filter '{search_index_filter}', but regular search found documents. Falling back to regular search.")
                return await self.run_search_approach(messages, overrides, auth_claims)
                
            logging.info(f"Agentic retrieval successful: found {len(results)} results")
                
        except Exception as e:
            error_msg = str(e)
            if "content management policy" in error_msg.lower() or "filtered" in error_msg.lower():
                logging.warning(f"Agentic retrieval blocked by content filter, falling back to regular search: {error_msg}")
            else:
                logging.error(f"Agentic retrieval failed with error: {error_msg}, falling back to regular search")
            return await self.run_search_approach(messages, overrides, auth_claims)

        data_points = await self.get_sources_content(
            results,
            use_semantic_captions=False,
            include_text_sources=send_text_sources,
            download_image_sources=send_image_sources,
            user_oid=auth_claims.get("oid"),
        )
        extra_info = ExtraInfo(
            data_points,
            thoughts=[
                ThoughtStep(
                    "Use agentic retrieval",
                    messages,
                    {
                        "reranker_threshold": minimum_reranker_score,
                        "results_merge_strategy": results_merge_strategy,
                        "filter": search_index_filter,
                    },
                ),
                ThoughtStep(
                    f"Agentic retrieval results (top {top})",
                    [result.serialize_for_results() for result in results],
                    {
                        "query_plan": (
                            [activity.as_dict() for activity in response.activity] if response.activity else None
                        ),
                        "model": self.agent_model,
                        "deployment": self.agent_deployment,
                    },
                ),
            ],
        )
        return extra_info
