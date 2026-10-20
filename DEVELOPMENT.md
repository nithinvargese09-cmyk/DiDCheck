# Development and provenance

This repository was prepared on 2026-10-18 as companion software to the finalized October 2026 research paper. It does not replace or modify that paper.

1. Recovered the final paper, technical supplement, scientific implementation and saved simulation outputs.
2. Copied the numerical core without modifying its bytes; its SHA-256 is recorded in `validation/CORE_PROVENANCE.json`.
3. Added explicit input validation, a typed result interface, documented sampling designs, centered bootstrap components, simulation utilities and plots.
4. Added independent scalar-loop covariance and resampling checks, and selected compact references extracted from the original saved simulations.
5. Replayed bounded prefixes and three complete analytic scenarios with the original random streams and batch size. These are partial reproductions, not a full rerun of the study.
6. Applied BSD-3-Clause as selected by the project owner, prepared citation metadata, documentation, CI and build artifacts.

See `VALIDATION_REPORT.md` and the accompanying machine-readable records for executed results and limitations. No fabricated commit history, external review, publication, downloads, or adoption is represented. The initial public commit records the prepared release; it does not reconstruct historical development commits.

OpenAI Codex assisted with package design, code adaptation/generation, documentation, tests, execution and review, in addition to the substantive research assistance disclosed in the companion manuscript. The authors retain responsibility for the research and software. AI assistance is not evidence of independent peer review.
