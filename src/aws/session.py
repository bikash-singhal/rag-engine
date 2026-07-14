import os

import boto3
from dotenv import load_dotenv

load_dotenv()


def get_bedrock_runtime_client():

    print("AWS_PROFILE =", os.getenv("AWS_PROFILE"))
    print("AWS_REGION  =", os.getenv("AWS_REGION"))
    
    session = boto3.Session(
        profile_name=os.getenv("AWS_PROFILE"),
        region_name=os.getenv("AWS_REGION"),
    )

    return session.client("bedrock-runtime")