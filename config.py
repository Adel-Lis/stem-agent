
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "o4-mini"
MODEL_EVAL = "gpt-4.1"

MAX_GROWTH_CYCLES = 20 # I want to prevent infinite loops
MIN_GROWTH_CYCLES = 3

BASE_OUTPUT_DIR = "outputs"

MAX_SEARCH_RESULTS = 5 # I set a limit to how many the builder reads from the web query