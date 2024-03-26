import logging
from pathlib import Path

class Config:

    def __init__(self):

        # default for private models if desired; can set to none to
        # force host declarations
        self.default_host_name = "localhost"
        self.default_port = 8000

        
        # Kamiwaza integration - will pull model deployments, hosts, ports, populate selector
        self.kamiwaza_api_endpoint = "http://localhost:7777"
        
        # Setting this false for release 0 because this isn't really ready
        # for public consumption yet and the debug is noisy
        self.enable_events = False

        # if using event bus
        self.kafka_host = "localhost"
        self.kafka_port = 9093
        self.default_kafka_topic = "agent-events"
        self.default_kafka_group_id = "agent-events-consumer"
        self.weaviate_url = "http://weaviate:8088"
        self.BASE_DIR = Path(__file__).parent

settings = Config()

