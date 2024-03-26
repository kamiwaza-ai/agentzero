import json
from kafka import KafkaProducer, KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic
from datetime import datetime
import time
from agentzero.config import settings

TOPICS = ["agent-logs"]

class AgentEvents:
    def __init__(self, bootstrap_servers = None):
        self.bootstrap_servers = bootstrap_servers if bootstrap_servers else f"{settings.kafka_host}:{settings.kafka_port}"
        self.producer = None
        self.consumer = None
        self.admin_client = KafkaAdminClient(bootstrap_servers=self.bootstrap_servers)
        self.create_topics()

    def create_topics(self):
        """
        Create Kafka topics from the TOPICS list if they do not exist.
        """
        existing_topics = self.admin_client.list_topics()
        new_topics = [NewTopic(name=topic, num_partitions=1, replication_factor=1) for topic in TOPICS if topic not in existing_topics]
        if new_topics:
            try:
                self.admin_client.create_topics(new_topics)
            except Exception as e:
                print(f"Failed to create topics. Error: {str(e)}")

    def create_producer(self):
        if not self.producer:
            self.producer = KafkaProducer(bootstrap_servers=self.bootstrap_servers,
                                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))
        return self.producer

    def create_consumer(self, group_id):
        self.consumer = KafkaConsumer(bootstrap_servers=self.bootstrap_servers,
                                      group_id=group_id,
                                      auto_offset_reset='earliest',
                                      value_deserializer=lambda x: json.loads(x.decode('utf-8')))
        return self.consumer

    def emit_event(self, topic, message):
        producer = self.create_producer()
        try:
            producer.send(topic, value=message).get(timeout=10)
        except Exception as e:
            print(f"Failed to send message to topic {topic}. Error: {str(e)}")

    # 300000 - 5m
    def listen_for_events(self, topic, group_id, timeout_ms=300000):
        consumer = self.create_consumer(group_id)
        consumer.subscribe([topic])
        start_time = time.time()
        while (time.time() - start_time) * 1000 < timeout_ms:
            messages = consumer.poll(timeout_ms=timeout_ms)
            for _, records in messages.items():
                for record in records:
                    yield record.value
        consumer.close()

    def close(self):
        if self.producer:
            self.producer.close()
        if self.consumer:
            self.consumer.close()
