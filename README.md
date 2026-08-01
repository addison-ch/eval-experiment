## An educational LLM evaluation system

Potential improvements (TODOS):
- Runner coordinates retries, have client surface whether or not exception is retriable
- Capture raw HTTP responses from API somewhere
- Fix duplicate cache misses in a run leading to wasted API calls