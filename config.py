import os
import boto3
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse

load_dotenv()

AWS_REGION   = os.environ.get("AWS_DEFAULT_REGION", "ap-southeast-1")
BEARER_TOKEN = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "")
MODEL_ID     = "apac.anthropic.claude-3-5-sonnet-20240620-v1:0"


BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'assistant.db')}"
CHROMA_DIR         = os.path.join(BASE_DIR, "chroma_db")
PDF_DIR            = os.path.join(BASE_DIR, "uploaded_pdfs")
CASBIN_MODEL_PATH  = os.path.join(BASE_DIR, "rbac", "model.conf")   # ← fixed
CASBIN_POLICY_PATH = os.path.join(BASE_DIR, "rbac", "policy.csv")   # ← fixed

os.environ["LANGCHAIN_TRACING_V2"] = os.environ.get("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_API_KEY"]    = os.environ.get("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]    = os.environ.get("LANGCHAIN_PROJECT", "multi-agent")

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        if not BEARER_TOKEN:
            raise RuntimeError("AWS_BEARER_TOKEN_BEDROCK is not set in your .env file.")

        client = boto3.client(
            service_name          = "bedrock-runtime",
            region_name           = AWS_REGION,
            aws_session_token     = BEARER_TOKEN,
            aws_access_key_id     = "dummy",
            aws_secret_access_key = "dummy"
        )

        _llm = ChatBedrockConverse(
            model  = MODEL_ID,
            client = client
        )

    return _llm