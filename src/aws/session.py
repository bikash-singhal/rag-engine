from boto3 import Session

from src.config.settings import AWS_PROFILE, AWS_REGION


def get_bedrock_runtime_client():

    session = Session(
        profile_name=AWS_PROFILE,
        region_name=AWS_REGION,
    )

    return session.client("bedrock-runtime")
