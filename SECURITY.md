# Security

Don't post API tokens, credentials, private data, or proprietary imagery in issues or pull requests.

Only load `joblib` models you trust, because loading one can execute code. The SHA-256 sidecar detects accidental changes to a model file, but it isn't a signature.

Server integrations should restrict paths, URL hosts, allowed source IDs, job concurrency, and output retention, and should run with minimal privileges.

Report suspected vulnerabilities privately to the repository owner rather than publishing exploit details.
