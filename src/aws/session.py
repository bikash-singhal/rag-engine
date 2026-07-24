import os

import boto3
from dotenv import load_dotenv

load_dotenv()


def get_bedrock_runtime_client():

    session = boto3.Session(
        profile_name=os.getenv("AWS_PROFILE"),
        region_name=os.getenv("AWS_REGION"),
    )

    return session.client("bedrock-runtime")
