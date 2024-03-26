# AgentZero

AgentZero is a package that acts as a chat interface, with classes for interacting with OpenAI compatible APIs for inference,
both locally and elsewhere.

## Installing/Running

Below this installing/running is a quickstart for with & without Kamiwaza community edition.

Largely untested in fresh envs as of release 0 here; but you'll want to:

```
git clone https://kamiwaza-ai/agentzero
cd agentzero
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

And optionally if you are using it

`pip install ./kamiwaza*whl`

Or similar. Don't install the kamiwaza libs via pip unless you are also running Kamiwaza community edition (or other, but then you are probably
talking to us); we wrapper the imports for Kamiwaza to autodetect.

## Quickstart With Kamiwaza Community Edition

1. Follow the install instructions
2. Launch Kamiwaza
3. Edit the config file ina gentzero (config.py)

You should be done

## Quickstart without Kamiwaza

1. Be sure to set your `OPENAІ_API_KEY` env variable
2. Follow the instructions to install
3. for now (sorry!) edit LLM/ChatProcessor.py and set the `MODEL = 'model'` line to your preferred model. First PR moves to config?

## Kamiwaza

AgentZero has hooks to work with Kamiwaza, leveraging the **Distributed Data Engines** and **Inference Mesh** from Kamiwaza.AI - it can leverage Kamiwaza to:

* Check for and automatically leverage deployed Kamiwaza models
* Leverage the Kamiwaza catalog for list available sources for automated RAG
* Use Kamiwaza to search vector databases and do byte-range retrieval on relevant results

## AgentZero vs AgentZero Chat

This release is largely focused on AgentZero chat, which is in `Modules/Chat`; however, you'll notice there's a bunch of scaffolding here. You can consider
the rest of that as experimentation of sorts; 

## State/Roadmap

This is a very early release; it's a usable MVP, and it should work out of the box with Kamiwaza Community Edition 0.2.0+; if you change default ports/etc, modify `config.py`

While this was basically hacked together in ~2 days, here's some stuff we have planned:

- [ ] Visual controls for the user to have more (priority #1!)
- [ ] Switch to React & Material UI for front-end (initial release is vanilla html)
- [ ] Move history out of text files
- [ ] Authentication
- [ ] Easier support for different endpoints for primary model responder & RAG functions





