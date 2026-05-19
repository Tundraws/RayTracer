# Coursework Completion Checklist (Variant 42)

## What is mandatory

- GPU-based real-time ray tracing implemented (`C++ + CUDA + OptiX`)
- Functional scene: geometry, materials, light, shadows, camera controls
- Git-based development history with clear commits
- Tests (unit + at least one GPU smoke/integration check)
- Static analysis and quality report
- Documentation (`README`, architecture notes, test evidence)

## Current "ideal" test package

- Unit tests for scene logic:
  - default scene values
  - material toggling
  - movement and clamping bounds
  - invalid index behavior
- Unit tests for camera logic:
  - normalized and orthogonal basis vectors
  - aspect ratio fallback
  - FOV influence on camera scale
- GPU smoke test:
  - renderer initialization
  - one-frame render
  - non-empty pixel buffer
  - valid GPU frame time

## Evidence to include in report

- Test run log (console output with PASS/FAIL lines)
- Screenshot of successful test run
- Performance table (FPS / Frame ms / GPU ms) for 3 scenarios
- Screenshot(s) of running renderer with HUD
- Static analysis summary (tool + key warnings + fixes)

## Suggested next implementation steps

1. Add GitHub Actions workflow:
   - build tests
   - run tests
   - run static analysis (`cppcheck` or `clang-tidy`)
2. Add `docs/performance-results.md` and record measurements
3. Add architecture diagram and module interaction notes
4. Finalize report sections 4 and 5 using generated artifacts
