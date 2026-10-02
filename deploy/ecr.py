"""Build the gateway image and push it to AWS ECR, where Runpod pulls it from.

    python deploy/ecr.py [tag]     # default tag: the date and time, e.g. 20261002-2130

Creates the repository if needed, gives Runpod's deployment roles pull access (the repository policy from
https://docs.runpod.io/tutorials/pods/use-private-ecr-images), pushes <tag> and `latest`, and registers the repository
with Runpod (an ECR delegation) once. Needs AWS credentials, RUNPOD_API_KEY and Docker. Behind a TLS-inspecting proxy,
set BUILD_CA to its CA bundle (passed to the build as a secret, not kept in the image).
"""
import base64
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import boto3
import httpx

REGION = os.environ.get("AWS_REGION", "us-east-1")
REPO = "model-gateway"
RUNPOD_ROLES = ["arn:aws:iam::550005742258:role/prod-us-east-1-deployment-role",
                "arn:aws:iam::550005742258:role/prod-us-west-2-deployment-role"]
POLICY = {"Version": "2012-10-17", "Statement": [{
    "Sid": "AllowRunpodPull", "Effect": "Allow", "Principal": "*",
    "Action": ["ecr:GetAuthorizationToken", "ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer",
               "ecr:BatchGetImage"],
    "Condition": {"StringEquals": {"aws:PrincipalArn": RUNPOD_ROLES}}}]}


def run(*cmd, **kw):
    print("+", " ".join(cmd[:6]), "..." if len(cmd) > 6 else "")
    subprocess.run(cmd, check=True, **kw)


def main(tag: str) -> None:
    ecr = boto3.client("ecr", region_name=REGION)
    try:
        repo = ecr.describe_repositories(repositoryNames=[REPO])["repositories"][0]
    except ecr.exceptions.RepositoryNotFoundException:
        repo = ecr.create_repository(repositoryName=REPO, imageScanningConfiguration={"scanOnPush": False})["repository"]
        print(f"created repository {repo['repositoryUri']}")
    ecr.set_repository_policy(repositoryName=REPO, policyText=json.dumps(POLICY))
    uri = repo["repositoryUri"]

    auth = ecr.get_authorization_token()["authorizationData"][0]
    user, password = base64.b64decode(auth["authorizationToken"]).decode().split(":", 1)
    run("docker", "login", "--username", user, "--password-stdin", auth["proxyEndpoint"], input=password.encode())
    root = Path(__file__).resolve().parent.parent / "gateway"
    secret = ["--secret", f"id=ca,src={os.environ['BUILD_CA']}"] if os.environ.get("BUILD_CA") else []
    run("docker", "build", *secret, "-t", f"{uri}:{tag}", "-t", f"{uri}:latest", str(root))
    run("docker", "push", f"{uri}:{tag}")
    run("docker", "push", f"{uri}:latest")

    headers = {"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"}
    have = httpx.get("https://api.runpod.io/v2/registries/delegations", headers=headers, timeout=30).json()
    have = have.get("delegations", have.get("data", [])) if isinstance(have, dict) else have
    if not any(d.get("repository") == REPO for d in have):
        # The API calls it an ARN, but takes the image URI with a tag (any tag then works for the repository)
        r = httpx.post("https://api.runpod.io/v2/registries/delegations", headers=headers, timeout=60,
                       json={"resource": f"{uri}:{tag}", "name": REPO})
        print("runpod delegation:", r.status_code, r.text[:300])
    print(f"image: {uri}:{tag}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M"))
