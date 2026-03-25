import sys
import boto3
import json
import os


def get_review(diff_text: str) -> str:
    bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    region = os.getenv("AWS_REGION", "us-east-2")

    try:
        if bearer_token:
            # Use bearer token for authentication
            client = boto3.client(
                "bedrock-runtime",
                region_name=region,
                aws_access_key_id="",
                aws_secret_access_key="",
                aws_session_token=bearer_token
            )
        else:
            # Fall back to default credentials (for local development)
            client = boto3.client("bedrock-runtime", region_name=region)

        prompt = f"""You are a senior software engineer performing a code review.
Review the following PR diff and provide actionable feedback on:
- Bugs or logic errors
- Security vulnerabilities
- Performance issues
- Code style and readability concerns

Be concise. Only comment on things that matter. If the code looks good, say so briefly.

```diff
{diff_text}
```"""

        response = client.invoke_model(
            modelId="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
            }),
        )

        result = json.loads(response["body"].read())
        return result["content"][0]["text"]
    except Exception as e:
        raise RuntimeError(f"Failed to get code review from Bedrock: {e}")


def truncate_diff_by_lines(diff_text: str, max_lines: int = 1000) -> str:
    """Truncate diff by line count to maintain valid format."""
    lines = diff_text.split("\n")
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + "\n\n... (diff truncated due to size)"
    return diff_text


def main():
    if len(sys.argv) < 2:
        print("Usage: review.py <diff_file>", file=sys.stderr)
        sys.exit(1)

    diff_path = sys.argv[1]

    try:
        with open(diff_path, "r") as f:
            diff_text = f.read()
    except FileNotFoundError:
        print(f"Error: Diff file not found: {diff_path}", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"Error reading diff file: {e}", file=sys.stderr)
        sys.exit(1)

    if not diff_text.strip():
        print("No changes detected.")
        return

    # Truncate very large diffs by line count to maintain valid format
    diff_text = truncate_diff_by_lines(diff_text)

    try:
        review = get_review(diff_text)
        # Truncate review if it exceeds GitHub's comment size limit (~65KB)
        max_comment_size = 65000
        if len(review) > max_comment_size:
            review = review[:max_comment_size] + "\n\n... (review truncated due to size)"
        print(review)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
