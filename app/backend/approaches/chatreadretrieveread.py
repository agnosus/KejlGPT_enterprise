from typing import Any, Coroutine, List, Literal, Optional, Union, overload
import aiohttp
import asyncio
import re
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorQuery
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai_messages_token_helper import build_messages, get_token_limit

from approaches.approach import ThoughtStep
from approaches.chatapproach import ChatApproach
from core.authentication import AuthenticationHelper

class ChatReadRetrieveReadApproach(ChatApproach):
    """
    A multi-step approach that first uses OpenAI to turn the user's question into a search query,
    then uses Azure AI Search to retrieve relevant documents, and then sends the conversation history,
    original user question, and search results to OpenAI to generate a response.
    """

    def __init__(
        self,
        *,
        search_client: SearchClient,
        auth_helper: AuthenticationHelper,
        openai_client: AsyncOpenAI,
        chatgpt_model: str,
        chatgpt_deployment: Optional[str],  # Not needed for non-Azure OpenAI
        embedding_deployment: Optional[str],  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
        embedding_model: str,
        embedding_dimensions: int,
        sourcepage_field: str,
        content_field: str,
        query_language: str,
        query_speller: str,
    ):
        self.search_client = search_client
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.chatgpt_model = chatgpt_model
        self.chatgpt_deployment = chatgpt_deployment
        self.embedding_deployment = embedding_deployment
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.query_language = query_language
        self.query_speller = query_speller
        self.chatgpt_token_limit = get_token_limit(chatgpt_model)

    @property
    def system_message_chat_conversation(self):
        return """You are a helpful assistant that answers technical questions about Buchi Extraction solutions. Be brief in your answers.
    Concepts to remember:\\n - Application: A method used on an instrument to determine the amount of a given analyte or to describe how to use the instrument on a given sample type with specific parameters. Methods, procedures, and results of such applications are explained in application notes.\\n - Configuration/Instrument Configuration: An instrument with a particular article number that includes a set of features, components, or accessories. A bundle refers to an instrument sold with another instrument, usually with a specific article number.\\n – Hydrolysis Unit: Instrument such as HydrolEx H-506. An instrument for sample preparation for fat determination.\\n – Extraction Units: instruments divided into classical extraction and pressurized solvent extraction.\\n – Classical extraction consists of FatExtractor E-500 and UniversalExtractor E-800.\\n – FatExtractor E-500 is used for fat determination of food and feed samples.\\n - FatExtractor E-500 is divided into configuration Soxhlet (SOX), hot extraction (HE) and economic continuous extraction (ECE).\\n – UniversalExtractor E-800 is used for extraction of environmental, food and feed, chemical, polymer and textile samples.\\n – UniversalExtractor E-800 is divided into UniversalExtractor E-800 standard and UniversalExtractor E-800 HE.\\n – Pressurized solvent extraction: SpeedExtractor is divided into configuration E-916, E-916 XL and E-914.
      - High-End Kjeldahl: Includes the KjelMaster K-375 (a distillation unit with integrated titration for nitrogen-containing analytes) which can be coupled with KjelSampler K-376 / K-377 (an autosampler instrument that can transfer samples to the KjelMaster K-375).
Answer ONLY with the facts listed in the list of sources below. If there isn't enough information, just say "I was not able to find any information in the provided resources. If your question is considered relevant and there should be an answer available, I will receive training and updates in the coming weeks." Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question.
   For tabular information, return it as an HTML table. Do not return markdown format. Always use plain text for equations. If the question is not in English, answer in the language used in the question.
   Each source has a name followed by the actual information. Always include the source name for each fact you use in the response. Use square brackets to reference the source, for example [example1.txt]. Don't combine sources, list each source separately, for example [example1.txt][example2.pdf].
{follow_up_questions_prompt}
{injected_prompt}

        """

    async def get_api_response(self, query: str) -> dict:
        """
        Make an API call to Perplexity AI when search results are empty.
        """
        url = "https://api.perplexity.ai/chat/completions"
        payload = {
            "return_citations": False,
            "messages": [
                {
            "role": "system",
            "content": "Be precise and concise."
        },
                {
                    "role": "user",
                    "content": f"Answer the following question in the context of Extraction: {query}"
                }
            ],
            "model": "sonar",
            "temperature": 0.2
        }
        headers = {
            "Authorization": "Bearer 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",
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"API call failed with status {response.status}"}

    @overload
    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: Literal[False],
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, ChatCompletion]]: ...

    @overload
    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: Literal[True],
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, AsyncStream[ChatCompletionChunk]]]: ...

    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: bool = False,
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]]]:
        has_text = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        has_vector = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]

        use_semantic_captions = True if overrides.get("retrieval_mode") == "text" else False
        if use_semantic_captions:
            top = 3
            minimum_search_score = 0
            minimum_reranker_score = 0
        else:
            top = overrides.get("top", 3)
            minimum_search_score = overrides.get("minimum_search_score", 0.0)
            minimum_reranker_score = overrides.get("minimum_reranker_score", 0.0)

        filter = self.build_filter(overrides, auth_claims)
        use_semantic_ranker = True if overrides.get("semantic_ranker") and has_text else False

        original_user_query = messages[-1]["content"]
        if not isinstance(original_user_query, str):
            raise ValueError("The most recent message content must be a string.")
        user_query_request = "Generate search query for: " + original_user_query

        tools: List[ChatCompletionToolParam] = [
            {
                "type": "function",
                "function": {
                    "name": "search_sources",
                    "description": "Retrieve sources from the Azure AI Search index",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_query": {
                                "type": "string",
                                "description": "Query string to retrieve documents from azure search eg: 'Health care plan'",
                            }
                        },
                        "required": ["search_query"],
                    },
                },
            }
        ]

        query_response_token_limit = 100
        query_messages = build_messages(
            model=self.chatgpt_model,
            system_prompt=self.query_prompt_template,
            tools=tools,
            few_shots=self.query_prompt_few_shots,
            past_messages=messages[:-1],
            new_user_content=user_query_request,
            max_tokens=self.chatgpt_token_limit - query_response_token_limit,
        )

        chat_completion: ChatCompletion = await self.openai_client.chat.completions.create(
            messages=query_messages,  # type: ignore
            model="gpt4" if overrides.get('use_gpt4') else "chat",
            temperature=0.0,
            max_tokens=query_response_token_limit,
            n=1,
            tools=tools,
        )

        query_text = self.get_search_query(chat_completion, original_user_query)

        vectors: list[VectorQuery] = []
        if has_vector:
            vectors.append(await self.compute_text_embedding(query_text))

        if not has_text:
            query_text = None

        results = await self.search(
            top,
            query_text,
            filter,
            vectors,
            use_semantic_ranker,
            use_semantic_captions,
            minimum_search_score,
            minimum_reranker_score,
        )

        # GAHAintervention: each result is checked using gpt to see if it can answer the question 
        async def generate_response(user_query, DOC, i):
            doc = DOC[i].content
            response = await self.openai_client.chat.completions.create(
                model="gpt4" if overrides.get('use_gpt4') else "chat",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that is expert in technical text understanding. You pay attention to the smallest detail in the text. Keep it concise and to the point."
                    },
                    {
                        "role": "user",
                        "content": f"Answer the user query based on the following context. If the context does not contain enough information to answer the query, return NONE : \nQUERY: {user_query} \n CONTEXT: {doc}"
                    }
                ],
                temperature=0)
            response_str = str(response.choices[0].message.content).strip()
            return response_str

        async def generate_responses_async(user_query, DOC):
            tasks = [generate_response(user_query, DOC, i) for i in range(len(DOC))]
            responses = await asyncio.gather(*tasks)

            RESPONSE = []
            INDICES = []
            for index, response_str in enumerate(responses):
                if response_str.lower() != 'none':
                    RESPONSE.append(response_str)
                    INDICES.append(index)

            return RESPONSE, INDICES

        RESPONSE, INDICES = await generate_responses_async(query_text, results)

        
        for i in INDICES:
            idx = INDICES.index(i)
            results[i].content = RESPONSE[idx]
        final_result = [results[i] for i in INDICES]
        results = final_result

        # end of intervention
        sources_content = self.get_sources_content(results, use_semantic_captions, use_image_citation=False)
        content = "\n".join(sources_content)

        system_message = self.get_system_prompt(
            overrides.get("prompt_template"),
            self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else "",
        )

        response_token_limit = 1024
        messages_for_completion = build_messages(
            model=self.chatgpt_model,
            system_prompt=system_message,
            past_messages=messages[:-1],
            new_user_content=original_user_query + "\n\nSources:\n" + content,
            max_tokens=self.chatgpt_token_limit - response_token_limit,
        )

        data_points = {"text": sources_content}

        extra_info = {
            "data_points": data_points,
            "thoughts": [
                ThoughtStep(
                    "Prompt to generate search query",
                    [str(message) for message in query_messages],
                    {"model": "gpt4o"}
                ),
                ThoughtStep(
                    "Search using generated search query",
                    query_text,
                    {
                        "use_semantic_captions": use_semantic_captions,
                        "use_semantic_ranker": use_semantic_ranker,
                        "top": top,
                        "filter": filter,
                        "has_vector": has_vector,
                    },
                ),
                ThoughtStep(
                    "Search results",
                    [result.serialize_for_results() for result in results],
                ),
                ThoughtStep(
                    "Prompt to generate answer",
                    [str(message) for message in messages_for_completion],
                    {"model": "gpt4o"},
                ),
            ],
        }

        # Generate the initial response
        chat_completion = await self.openai_client.chat.completions.create(
            model="gpt4" if overrides.get('use_gpt4') else "chat",
            messages=messages_for_completion,
            temperature=overrides.get("temperature", 0.3),
            max_tokens=response_token_limit,
            n=1,
            stream=False,
        )

        initial_response = chat_completion.choices[0].message.content

        # Check if the response indicates no information was found
        if 'any information' in initial_response.lower():
            # If no information was found, use Perplexity AI API
            api_response = await self.get_api_response(query_text)
            
            if "error" not in api_response:
                prefix = "Sorry! No information was found in our documents. Meanwhile here is an answer from perplexity.ai that you may find useful:\n\n"
                api_content = api_response['choices'][0]['message']['content']
                
                # Transform numbered citations in the main text
                if 'citations' in api_response:
                    transformed_content = api_content
                    for idx, citation in enumerate(api_response['citations'], 1):
                        transformed_content = transformed_content.replace(
                            f'[{idx}]', 
                            f'[{citation}]'
                        )
                    prefixed_api_content = prefix + transformed_content
                else:
                    prefixed_api_content = prefix + api_content

                # Update messages with the API response
                new_messages = messages[:-1]  # All messages except the last user message
                new_messages.append({"role": "user", "content": original_user_query})
                new_messages.append({"role": "assistant", "content": prefixed_api_content})

                # Update extra_info to reflect the API call
                extra_info["thoughts"].extend([
                    ThoughtStep(
                        "No information in documents",
                        "Initial response indicated no information was found. Using Perplexity AI API response.",
                        {}
                    ),
                    ThoughtStep(
                        "API Response",
                        prefixed_api_content,
                        {"api_call": "Perplexity AI", "model": api_response['model']}
                    )
                ])

                # Create new chat completion with API response
                chat_coroutine = self.openai_client.chat.completions.create(
                model="gpt4" if overrides.get('use_gpt4') else "chat",
                messages=[
                    {
                        "role": "user",
                        "content": f""" repeat the ANSWER verbatim (begin with 'Sorry! No information was found in our documents. Meanwhile here is an answer from perplexity.ai that you may find useful:\n\n').  ANSWER: {prefixed_api_content}"""
                    }
                ],
                temperature=overrides.get("temperature", 0.3),
                max_tokens=1024,
                n=1,
                stream=should_stream,
            )
            else:
                # Handle API error
                error_message = "Sorry, I couldn't find an answer to your question and encountered an issue with the backup system."
                new_messages = messages[:-1]  # All messages except the last user message
                new_messages.append({"role": "user", "content": original_user_query})
                new_messages.append({"role": "assistant", "content": error_message})
                
                extra_info["thoughts"].append(
                    ThoughtStep(
                        "API Error",
                        api_response["error"],
                        {"api_call": "Perplexity AI"}
                    )
                )

                chat_coroutine = self.openai_client.chat.completions.create(
                    model="gpt4" if overrides.get('use_gpt4') else "chat",
                    messages=new_messages,
                    temperature=overrides.get("temperature", 0.3),
                    max_tokens=1024,
                    n=1,
                    stream=should_stream,
                )
        else:
            # If information was found, use the original response
            chat_coroutine = self.openai_client.chat.completions.create(
                model="gpt4" if overrides.get('use_gpt4') else "chat",
                messages=messages_for_completion,
                temperature=overrides.get("temperature", 0.3),
                max_tokens=response_token_limit,
                n=1,
                stream=should_stream,
            )

        return (extra_info, chat_coroutine)

    # Other methods (build_filter, search, compute_text_embedding, get_search_query, get_sources_content)
    # should be implemented here, but they're not provided in the original snippet.
    # Make sure to include these methods in your actual implementation.