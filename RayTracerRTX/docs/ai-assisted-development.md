# AI-Assisted Development

This document records the use of AI tools as a development process artifact.
It is not a replacement for testing or manual review; every accepted change is
validated through builds, tests, Docker checks, or direct application runs.

## Where AI Assistance Was Used

| Area | AI Role | Human/Tool Verification |
|---|---|---|
| Repository analysis | Inspect project structure and identify missing coursework artifacts | `git status`, source review |
| Build fixes | Diagnose Visual Studio, CUDA, OptiX and linker configuration issues | Native MSBuild run |
| Test planning | Propose CPU unit tests and GPU smoke-test criteria | Test executable output |
| Docker support | Add reproducible CPU-only check and document RTX GUI limitation | `docker compose build`, `docker compose run --rm coursework-check` |
| Static analysis evidence | Prepare SAST report structure and interpret warnings | `docs/static-analysis.md` |
| Performance evidence | Prepare benchmark documentation and table format | Native benchmark output |
| Documentation | Generate architecture/API/security notes and Mermaid diagrams | Manual review and repository diff |

## Validation Principle

AI-generated suggestions are accepted only after one of the following checks:

- native Visual Studio/MSBuild build succeeds;
- CPU unit tests pass;
- GPU smoke test passes on the RTX-capable Windows host;
- Docker coursework check passes;
- the user validates visual behavior in the interactive application.

## Current Tooling Boundary

The AI assistant can modify source files, documentation, tests, Docker files,
and Git history. It cannot replace final visual acceptance of the renderer,
because material appearance and scene composition are evaluated by the project
author in the running application.

