from langchain_google_genai import ChatGoogleGenerativeAI
import os
from typing import Annotated, Dict, Any, List, Union, Optional
from typing_extensions import TypedDict, Annotated, Literal
import operator
from dataclasses import dataclass, field
from re import T
import json
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import time
from tavily import TavilyClient
import re
from model import get_gemma

from api_key import TAVILY_API_KEY

gemma_model = get_gemma()

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

max_web_research_loops: int = 4


@dataclass(kw_only=True)
class SummaryState:
    research_topic: str = field(default=None)
    search_query: str = field(default=None)
    web_search_results: Annotated[list, operator.add] = field(default_factory=list)
    sources_gathered: Annotated[list, operator.add] = field(default=list)
    research_loop_count: int = 0
    running_summary: str = None


@dataclass(kw_only=True)
class SummaryStateInput:
    research_topic: str = None

@dataclass(kw_only=True)
class SummaryStateOutput:
    running_summary: str = None


# Query Writer
query_writer_instructions = """Your gola is to generate web search query.
The query will gather information about specific topic.

Topic: {research_topic}

Return your query as JSON object:
{{
    "query": "string",
    "aspect" : "string",
    "rationale" : "string"
}}
"""

# Summerizer Instructions
summerizer_instructions = """Your goal is to generate high-quality summary of the web search results.
when EXTENDING an existing summary:
1. Seamlessly integrate new information without repeating what's already covered.
2. Maintain consistancy with existing content's style.
3. Only add new and non-redudant information.
4. Ensure smooth transition between existing and new content.

when creating a NEW summary:
1. Highlight the most relevant information from each source.
2. Provide concise overview of the key points related to each report topic.
3. Emphasize on significant findings or insights.
4. Ensure coherent flow of information.

In both cases:
1. Focus on factual & objective information
2. Maintain consistat technical depth
3. Avoid repetition & redundancy
4. DON'T use phrases like "based on new results"
5. DON'T add preamble like "Here is an extended summary ...", instead just provide summary directly
6. DON'T add references or works cited section.
7. You will generate tables using markdown when user asks you to do.
"""

# Reflection Instructions
reflection_summary = """You are an expert research assistant analyzing summary about {research_topic}.

Your Tasks :
1. Identify knowledge gaps or areas the need further exploration.
2. Generate a follow-up question that would help in expanding the understanding.
3. Focus on technical details, implementation specifics.

Ensure follow-up question is self-contained and includes necessary context for web search.

Return response as JSON object:
{{
    "knowledge_gap" : "string",
    "follow_up_query" : "string"
}}
"""


def generate_query(state: SummaryState):
    # To generate query for web search

    system_message_for_query_writer = query_writer_instructions.format(research_topic=state.research_topic)

    result = gemma_model.invoke(
        [
            HumanMessage(content=f"IMPORTANT INSTRUCTIONS:\n{system_message_for_query_writer}\n\nGenerate a query for web search")
        ]
    )

    # print(f"[FUN] GENERATE_QUERY:\nType: {type(result)}\nContent: {result}")
    raw_content = result.content.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_content, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        raise ValueError(f"Failed to extract JSON from model response: {raw_content}")

    query = json.loads(json_str)
    return {"search_query" : query["query"]}


def deduplicate_and_format_sources(
    search_response: Union[Dict[str, Any], List[Dict[str, Any]]],
    max_tokens_per_source: int,
    fetch_full_page: bool = False
) -> str:
    """
    Format and deduplicate search responses from various search APIs.

    Takes either a single search response or list of responses from search APIs,
    deduplicates them by URL, and formats them into a structured string.

    Args:
        search_response (Union[Dict[str, Any], List[Dict[str, Any]]]): Either:
            - A dict with a 'results' key containing a list of search results
            - A list of dicts, each containing search results
        max_tokens_per_source (int): Maximum number of tokens to include for each source's content
        fetch_full_page (bool, optional): Whether to include the full page content. Defaults to False.

    Returns:
        str: Formatted string with deduplicated sources

    Raises:
        ValueError: If input is neither a dict with 'results' key nor a list of search results
    """
    # Convert input to list of results
    if isinstance(search_response, dict):
        sources_list = search_response['results']
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and 'results' in response:
                sources_list.extend(response['results'])
            else:
                sources_list.extend(response)
    else:
        raise ValueError("Input must be either a dict with 'results' or a list of search results")

    # Deduplicate by URL
    unique_sources = {}
    for source in sources_list:
        if source['url'] not in unique_sources:
            unique_sources[source['url']] = source

    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source: {source['title']}\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if fetch_full_page:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"

    return formatted_text.strip()


def format_sources(search_results: Dict[str, Any]) -> str:
    """
    Format search results into a bullet-point list of sources with URLs.

    Creates a simple bulleted list of search results with title and URL for each source.

    Args:
        search_results (Dict[str, Any]): Search response containing a 'results' key with
                                        a list of search result objects

    Returns:
        str: Formatted string with sources as bullet points in the format "* title : url"
    """
    return '\n'.join(
        f"* {source['title']} : {source['url']}"
        for source in search_results['results']
    )


def web_research(state: SummaryState):
    search_results = tavily_client.search(state.search_query, include_raw_content=True, max_results=1)
    search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000)
    return {
        "sources_gathered" : [format_sources(search_results)],
        "research_loop_count" : state.research_loop_count + 1,
        "web_search_results" : [search_str]
    }


def summarize_sources(state: SummaryState):
    existing_summary = state.running_summary
    print(state.web_search_results)
    most_recent_web_search = state.web_search_results[-1]

    if existing_summary:
        human_message = (
            f"IMPORTANT INSTRUCTIONS:\n{summerizer_instructions}\n\n"
            f"Extend the existing summary: {existing_summary}\n\n"
            f"Include new search results: {most_recent_web_search}"
            f"That addresses the following topic: {state.research_topic}"
        )
    else:
        human_message = (
            f"IMPORTANT INSTRUCTIONS:\n{summerizer_instructions}\n\n"
            f"Generate summary of these search results: {most_recent_web_search}"
            f"That addresses the following topic: {state.research_topic}"
        )

    result = gemma_model.invoke([HumanMessage(content=human_message)])

    return {"running_summary" : result.content}


def reflect_on_summary(state: SummaryState):
    result = gemma_model([
        HumanMessage(content=f"IMPORTANT INSTRUCTIONS:\n{reflection_summary.format(research_topic=state.research_topic)}\n\nIdentify a knowledge gap and generate a follow-up web search query based on existing knowledge: {state.running_summary}")
    ])
#    print(f">> [FUN] REFLECT ON SUMMARY:\n{result.content}")
    raw_content = result.content.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_content, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        raise ValueError(f"Failed to extract JSON from model response: {raw_content}")

    query = json.loads(json_str)
    print(query)
    return {"search_query" : query["follow_up_query"]}


def finalize_summary(state: SummaryState):
    all_sources = "\n".join(source for source in state.sources_gathered)
    print(f"All Sources: {all_sources}")
    running_summary = f"## Summary\n\n{state.running_summary}\n\nSources:\n{all_sources}"
    return {"running_summary" : running_summary}

def route_research(state: SummaryState):
    if state.research_loop_count <= max_web_research_loops:
        return "web_research"
    else:
        return "finalize_summary"
    

def perform_deep_research(query):
    
    builder = StateGraph(SummaryState, input_schema=SummaryStateInput, output_schema=SummaryStateOutput)

    builder.add_node("generate_query", generate_query)
    builder.add_node("web_research", web_research)
    builder.add_node("summarize_sources", summarize_sources)
    builder.add_node("reflect_on_summary", reflect_on_summary)
    builder.add_node("finalize_summary", finalize_summary)

    # Add edges
    builder.add_edge(START, "generate_query")
    builder.add_edge("generate_query", "web_research")
    builder.add_edge("web_research", "summarize_sources")
    builder.add_edge("summarize_sources", "reflect_on_summary")
    builder.add_conditional_edges("reflect_on_summary", route_research)
    builder.add_edge("finalize_summary", END)


    graph = builder.compile()

    research_input = SummaryStateInput(research_topic=query)

    research_output = graph.invoke(research_input)

    return research_output["running_summary"]