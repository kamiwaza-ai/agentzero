"""
    ChatRetriever implements some basic retrieval functions against the Kamiwaza catalog on behalf of ChatProcessor
    
    todo:
        - make the call to this less kamiwaza specific to make CP's inbound more flexible
        - make our behaviors more flexible
        
    
"""
import logging
import re
import json
import asyncio
from typing import Optional, List, Any, Dict
from kamiwaza.middleware.embedding.embedding_sentence_transformers import SentenceTransformerEmbedding
from kamiwaza.middleware.vectordb.milvus import MilvusImplementation
from kamiwaza.services.catalog.plugin_loader import load_plugin
from kamiwaza.services.retrieval.services import RetrievalService
from .ChatPrompts import chat_prompts

class ChatRetriever():

    MODEL = 'model'  # Default model identifier

    def __init__(self, chat_processor: Any, **kwargs):
        """_summary_

        Args:
            chat_processor (Any): This is an instance of ChatProcessor; we will not import
                because that would be circular. Any because we can't import the definition.
                Will reconsider this in the future

            Passthrough:
                'query' - dataset filter
                'platform' - platform filter (eg, "file", "s3")
                'environment' - env filter (eg, "PROD")
                'owners' - owners filter (list of owner URNs)
                'container' - container parent - generally should not use

            We don't permit other arguments because, because we are passing a class instance,
            which is a reference, so mutations here will affect the passed copy. We therefore
            expect callers to DISPOSE of this.
        """

        self.dataset_kwargs = {}
        for key in ['query', 'platform', 'environment', 'owners', 'container']:
            if key in kwargs:
                self.dataset_kwargs[key] = kwargs.pop(key)
        # only datahub for now
        self.catalog = load_plugin('data_datahub')()

        # only default vectordb for now
        self.vectordb = MilvusImplementation()
        self.chat_prompts = chat_prompts

        self.defaultmessages = [
                {"role": "system", "content": "You are an AI assistant that carefully follows instructions to aid retrieval of possibly relevant documents."},
            ]

        self.chat_processor = chat_processor

        # we never retrieve inside retrieval unless explciit
        self.chat_processor.retrieval = False # don't loop
        
        # we don't want streamed responses here because we aren't doing a callback atm
        self.chat_processor.STREAM = False

        # we are not doing a chat
        self.chat_processor.CREATE_TITLES = False

        self.logger = logging.getLogger(__name__)

    async def relevance(self, prompt: str) -> dict:
        """
        Asynchronously determines the relevance of datasets to the given prompt and extracts collections from custom properties.

        Args:
            prompt (str): The user's prompt to determine relevance for.

        Returns:
            dict: A dictionary containing two keys: 'sources' with relevant sources and 'collections' with extracted collections.
        """
        self.logger.debug(f"Setting {self.chat_processor.messages} to {self.defaultmessages}")
        self.chat_processor.messages = [item for item in self.defaultmessages]

        datasets = self.catalog.list_datasets(self.dataset_kwargs)
        dataset_urns = [dataset['urn'] for dataset in datasets]
        self.logger.debug(f"Datasets to include: {datasets}")

        sources = []
        collections = set()  # Using a set to avoid duplicate collections

        # Get dataset URNs and descriptions
        # TODO: after 0.2.0 use catalog.custom_properties 
        dataset_details_list = self.catalog.get_dataset_details(dataset_urns)
        if dataset_details_list:
            for dataset_details in dataset_details_list:
                # Safely extracting 'description' from 'datasetProperties' if available
                dataset_properties = dataset_details.get('datasetProperties', {})
                custom_properties = dataset_properties.get('customProperties', {})
                description = custom_properties.get('description', "No Description")
                if not description:
                    description = "No Description (evaluate path)"
                
                # Extracting 'platform', 'name' (as id), and 'origin' (as environment) from 'datasetKey' safely
                dataset_key = dataset_details.get('datasetKey', {})
                platform = dataset_key.get('platform', '').split(':')[-1]  # Extracting platform name after 'urn:li:dataPlatform:'
                dataset_id = dataset_key.get('name', '')  # Using 'name' as 'id'
                environment = dataset_key.get('origin', '')  # Using 'origin' as 'environment'
                
                # Constructing URN with safe extraction
                urn = f"urn:li:dataset:(urn:li:dataPlatform:{platform},{dataset_id},{environment})"
                sources.append((description, urn))

                # Extracting 'collection' from custom properties if available
                collection = custom_properties.get('collection')
                if collection:
                    collections.add(collection)

        relevance_prompt = self.chat_prompts.relevance_prompt 
        relevance_prompt = relevance_prompt.format(sources='\n'.join(str(source) for source in sources), prompt=prompt)
        
        # Directly await the asynchronous generate_response call; lower rep penalty because this tends ot look redundant and cut off
        response = await self.chat_processor.generate_response(relevance_prompt, chat_params={'repetition_penality': 1.01})
        self.logger.debug(f"full model response: [[[{response}]]]")
        response = response['last_response']  # Assuming generate_response returns a dict with 'last_response'
        self.logger.debug(f"Relevance response from LLM: {response}")

        relevant_sources = []  # Initialize an empty list to hold relevant sources
        # Process response to extract relevant sources and collections as before

        # First, try to find and safely parse a list representation in the response
        list_match = re.search(r'\[.*?\]', response)
        if list_match:
            # Safely parse the matched list string without using eval()
            try:
                # Convert the matched list string into a list using json.loads
                self.logger.debug(f"Matched list: {list_match.group(0)}")
                matched_list_str = list_match.group(0).replace("'", '"')  # Ensure double quotes for JSON compatibility
                evaluated_list = json.loads(matched_list_str)
                if isinstance(evaluated_list, list):
                    # Check if the evaluated list items match any of the source keys
                    source_keys = [key for _, key in sources]
                    relevant_sources = [item for item in evaluated_list if item in source_keys]
                    self.logger.debug(f"Matched sources: {relevant_sources}")
            except json.JSONDecodeError as e:
                self.logger.debug(f"Error parsing list from response: {e}")

        # If no valid list was found or evaluated, look for quoted strings
        if not relevant_sources:
            self.logger.debug(f"No relevant sources found in response: {response} - checking for quoted string")
            quoted_strings = re.findall(r'"(.*?)"', response)
            for string in quoted_strings:
                # Check if the stripped string matches any source key
                for _, key in sources:
                    if string == key:
                        relevant_sources.append(key)
            self.logger.debug(f"Matched sources: {relevant_sources}")

        return {"sources": relevant_sources, "collections": list(collections)}

    async def define_query(self, prompt: str, sources: List[tuple]) -> str:
        """
        Asynchronously defines a query by formatting a prompt with the given sources and sending it to the chat processor for a response.

        Args:
            prompt (str): The base prompt to which the sources will be appended.
            sources (List[tuple]): A list of tuples containing source information.

        Returns:
            str: The response from the chat processor.
        """
        retrieve_prompt = self.chat_prompts.retrieve_prompt

        self.logger.debug(f"Resetting chat processor messages to default.")
        self.chat_processor.messages = [item for item in self.defaultmessages]  # Reset messages to default
        formatted_sources = '\n'.join([source[0] for source in sources])  # Format sources for inclusion in the prompt
        
        # Directly await the asynchronous generate_response call
        response = await self.chat_processor.generate_response(retrieve_prompt.format(prompt=prompt, relevant_sources=formatted_sources))
        response = response['last_response']  # Assuming generate_response returns a dict with 'last_response'
        return response


    def retrieve_chunks(self, collections: List[str], query: str, dataset_urns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Retrieves relevant chunks based on the given query, collections, and optional dataset URNs.

        Args:
            collections (List[str]): List of collection names to search in.
            query (str): The query text for retrieval.
            dataset_urns (Optional[List[str]]): Optional list of dataset URNs to filter the search results.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing relevant chunk data.
        """
        rs = RetrievalService()
        # Leverage the updated retrieve_relevant_chunks method with additional parameters
        chunks = rs.retrieve_relevant_chunks(collections=collections, query=query, catalog_urns=dataset_urns, max_results=10)
        return chunks

    async def get_relevant_chunks(self, prompt: str) -> List[Dict[str, Any]]:
        """
        Asynchronously retrieves relevant chunks based on the given prompt.

        Args:
            prompt (str): The user's prompt to determine relevance for.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing relevant chunk data.
        """
        # Extract sources and collections from the relevance determination
        relevance_result = await self.relevance(prompt)
        if not relevance_result['sources']:
            return []

        collections = relevance_result['collections']
        dataset_urns = [source for source in relevance_result['sources']]  # Extract dataset URNs from sources

        # Generate a query based on the prompt and sources
        query = await self.define_query(prompt, relevance_result['sources'])

        # Retrieve chunks based on the generated query, collections, and dataset URNs
        try:
            chunks = self.retrieve_chunks(collections=collections, query=query, dataset_urns=dataset_urns)
        except Exception as e:
            self.logger.error(f"Error retrieving chunks: {e}")
            return []

        return chunks

    # One more function to go to get our model to sift the retrieved chunks, but we can hold off briefly
    # we could also use reranking here, but I'm more inclined to push a reranker call into the
    # kamiwaza retriever since we've now offloaded the embedding of the query/etc

    # def validate_chunks(self, prompt, chunks) -> list:
    #     self.chat_processor.messages = self.defaultmessages
    #     self.chat_processor.messages.append({"role": "user", "content": prompt})
    #     for chunk in chunks:
    #         self.chat_processor.messages.append({"role": "assistant", "content": chunk})
    #     response = self.chat_processor.generate_response()['last_response']
    #     return response
