## Purpose

Lets a reviewer start the service on a clean machine with Docker in one command, using the same configuration variables as the local setup.

## ADDED Requirements

### Requirement: One-command container start
The repository SHALL include a Dockerfile and a compose file so that `docker compose up` builds and starts the service on port 8000, with configuration read from `.env`. The image MUST NOT contain the `.env` file or the API key.

#### Scenario: Start with compose
- **WHEN** a reviewer copies `.env.example` to `.env`, fills in the API key and runs `docker compose up`
- **THEN** `GET http://localhost:8000/health` returns 200 with `{"status": "ok"}`

#### Scenario: Secrets not baked into the image
- **WHEN** the built image's filesystem is inspected
- **THEN** it contains no `.env` file
