#Step 1:Configuration

import json
import os
from typing import Annotated, Dict
from spider_env import SpiderEnv
from autogen import ConversableAgent, UserProxyAgent, config_list_from_json

from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

os.environ["AUTOGEN_USE_DOCKER"]="False"

llm_config = {
    "cache_seed": 48,
    "config_list": [{
        "model": os.environ.get("OPENAI_MODEL_NAME", "llama3-70b-8192"),
        "api_key": os.environ["GROQ_API_KEY"],
        "base_url": os.environ.get("OPENAI_API_BASE", "https://api.groq.com/openai/v1")
    }],
}

# Step 2: Import data
gym = SpiderEnv()

#Randomly select a question from spider
observation, info = gym.reset()

#The Natural Language Question
question = observation["instruction"]

print(question)

#The schema of the corresponding database
schema = info["schema"]
print(schema)

