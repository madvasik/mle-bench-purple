# MLE-Bench purple (AgentBeats)

Purple agent for [MLE-bench](https://agentbeats.dev/agentbeater/mle-bench): accepts the competition `tar.gz` and benchmark instructions from the green agent ([mle-bench-green](https://github.com/RDI-Foundation/mle-bench-green)), runs a **tabular sklearn baseline** (RandomForest), returns **`submission.csv`** as an A2A file artifact.

No LLM API keys are required in this image. Kaggle credentials are only used by the **green** side when you Quick Submit.

## Quick Submit (AgentBeats)

- **Green secrets:** `KAGGLE_USERNAME`, `KAGGLE_KEY`
- **Config JSON:** `{"competition_id": "spaceship-titanic"}`

Register this repo + `ghcr.io/<you>/mle-bench-purple:latest` as a purple agent; set **Amber manifest URL** to the raw `amber-manifest.json5` on `main`.

## Local

```bash
uv sync
uv run src/server.py --host 127.0.0.1 --port 9009
```

## Docker / GHCR

```bash
docker build -t mle-bench-purple .
```

CI: push to `main` with `GHCR_WRITE_TOKEN` secret if `GITHUB_TOKEN` cannot push packages (see your τ² repo workflow).

## Tests

```bash
uv sync --extra test
uv run pytest --agent-url http://127.0.0.1:9009
```
