import pytest
import time
import uuid
from AgentEvents.AgentEvents import AgentEvents

# Test data
TEST_TOPIC = 'test-topic'
TEST_GROUP_ID = 'agent-group-'+str(uuid.uuid4())

# Test event structure
TEST_EVENT = {'message': 'test event'}

def test_list_topics(agent_events_setup):
    """
    Test to connect to Kafka and list topics
    """
    # Create a Kafka consumer
    consumer = agent_events_setup.create_consumer(TEST_GROUP_ID)
    # Get the list of topics
    topics = consumer.topics()
    # Check if TEST_TOPIC is in the list of topics
    assert TEST_TOPIC in topics


@pytest.fixture
def agent_events_setup():
    return AgentEvents(bootstrap_servers='localhost:9093')

@pytest.mark.parametrize('event', [TEST_EVENT])
def test_emit_event(agent_events_setup, event):
    agent_events_setup.emit_event(TEST_TOPIC, event)
    # Assertions to check if event is emitted correctly

@pytest.mark.parametrize('event', [TEST_EVENT])
def test_listen_for_events(agent_events_setup, event):
    agent_events_setup.emit_event(TEST_TOPIC, event)
    time.sleep(5) # Allow time for the event to be processed
    consumer = agent_events_setup.create_consumer(TEST_GROUP_ID)
    consumer.subscribe([TEST_TOPIC])
    messages = consumer.poll(timeout_ms=10000)
    assert len(messages) > 0 # Check if messages were received

def test_close(agent_events_setup):
    agent_events_setup.close()
    assert True