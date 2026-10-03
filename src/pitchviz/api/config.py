import os

ALLOWED_ORIGIN   = os.getenv("ALLOWED_ORIGIN", "*")
GITHUB_TOKEN     = os.environ["GITHUB_TOKEN"]
GITHUB_OWNER     = os.environ["GITHUB_OWNER"]
GITHUB_REPO      = os.environ["GITHUB_REPO"]
DO_SPACES_REGION = os.environ["DO_SPACES_REGION"]
DO_SPACES_BUCKET = os.environ["DO_SPACES_BUCKET"]

GH_HEADERS = {
    "Authorization":        f"Bearer {GITHUB_TOKEN}",
    "Accept":               "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
