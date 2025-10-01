"""
Additional methods for streaming processing steps in agentic retrieval
"""
import uuid
import time
from typing import Any, AsyncGenerator
from openai.types.chat import ChatCompletionMessageParam


async def run_agentic_retrieval_approach_with_steps(
    self,
    messages: list[ChatCompletionMessageParam],
    overrides: dict[str, Any],
    auth_claims: dict[str, Any],
) -> AsyncGenerator[dict, None]:
    """
    Agentic retrieval approach with step-by-step streaming
    """
    from approaches.approach import ExtraInfo, ThoughtStep
    
    search_index_filter = self.build_filter(overrides, auth_claims)
    minimum_reranker_score = overrides.get("minimum_reranker_score", 0)
    top = overrides.get("top", 3)
    results_merge_strategy = overrides.get("results_merge_strategy", "interleaved")
    send_text_sources = overrides.get("send_text_sources", True)
    send_image_sources = overrides.get("send_image_sources", self.multimodal_enabled) and self.multimodal_enabled

    yield {
        "processing_step": {
            "id": str(uuid.uuid4()),
            "title": "Initializing agentic retrieval",
            "status": "in_progress",
            "details": "Starting intelligent agent-based document retrieval",
            "timestamp": time.time(),
            "metadata": {
                "top_results": top,
                "merge_strategy": results_merge_strategy,
                "minimum_reranker_score": minimum_reranker_score
            }
        }
    }

    try:
        yield {
            "processing_step": {
                "id": str(uuid.uuid4()),
                "title": "Agent analyzing query",
                "status": "in_progress",
                "details": "AI agent is analyzing your query to determine optimal retrieval strategy",
                "timestamp": time.time(),
                "metadata": {
                    "agent_model": self.agent_model,
                    "search_index": self.search_index_name
                }
            }
        }

        response, results = await self.run_agentic_retrieval(
            messages=messages,
            agent_client=self.agent_client,
            search_index_name=self.search_index_name,
            top=top,
            filter_add_on=search_index_filter,
            minimum_reranker_score=minimum_reranker_score,
            results_merge_strategy=results_merge_strategy,
        )
        
        # If agentic retrieval returns empty response (due to content filtering), fall back to regular search
        if not response.response and not results:
            yield {
                "processing_step": {
                    "id": str(uuid.uuid4()),
                    "title": "Falling back to regular search",
                    "status": "in_progress",
                    "details": "Agentic retrieval returned empty response, switching to regular search approach",
                    "timestamp": time.time(),
                    "metadata": {
                        "fallback_reason": "empty_agentic_response"
                    }
                }
            }
            
            # Use the regular search approach
            async for step in self.run_search_approach_with_steps(messages, overrides, auth_claims):
                yield step
            return
            
        yield {
            "processing_step": {
                "id": str(uuid.uuid4()),
                "title": "Agentic retrieval completed",
                "status": "completed",
                "details": f"Agent successfully retrieved {len(results)} relevant documents",
                "timestamp": time.time(),
                "metadata": {
                    "documents_retrieved": len(results),
                    "agent_activities": len(response.activity) if response.activity else 0,
                    "query_plan": [activity.as_dict() for activity in response.activity] if response.activity else None
                }
            }
        }
            
    except Exception as e:
        yield {
            "processing_step": {
                "id": str(uuid.uuid4()),
                "title": "Agentic retrieval failed",
                "status": "error",
                "details": f"Agentic retrieval encountered an error: {str(e)}, falling back to regular search",
                "timestamp": time.time(),
                "metadata": {
                    "error": str(e),
                    "fallback_reason": "agentic_error"
                }
            }
        }
        
        # Fall back to regular search
        async for step in self.run_search_approach_with_steps(messages, overrides, auth_claims):
            yield step
        return

    yield {
        "processing_step": {
            "id": str(uuid.uuid4()),
            "title": "Processing retrieved sources",
            "status": "in_progress",
            "details": "Extracting and preparing content from agent-retrieved documents",
            "timestamp": time.time(),
            "metadata": {
                "include_text_sources": send_text_sources,
                "include_image_sources": send_image_sources
            }
        }
    }

    data_points = await self.get_sources_content(
        results,
        use_semantic_captions=False,
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
    
    yield {"extra_info": extra_info}
