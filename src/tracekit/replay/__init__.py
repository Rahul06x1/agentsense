"""Replay engine — fast-follow (Weeks 7-10). Placeholder in v0.

Mocked, deterministic-redaction-aware replay + trajectory diff. The model client
is pluggable behind one adapter: AWS Bedrock (Converse API, boto3) for dev;
OpenAI-compatible (Gemini/Ollama/OpenAI) for the OSS release. Capture Bedrock
usage (incl. cache tokens) + metrics.latencyMs into span fields.
"""
