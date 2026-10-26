# GPU session

- commit: 26e9b4d
- machine: Apple M3
- memory: 8 GB
- os: macOS 26.7.1
- python: 3.12.2
- jax: 0.12.0.1
- jaxlib: 0.12.0.1
- jax-metal-shim: 0.1.0

| Step | Status | Minutes | Finished | Note | Log |
|---|---|---|---|---|---|
| setup | passed | 0.0 | 2026-10-26T18:42 |  | `logs/setup.log` |
| install | passed | 0.1 | 2026-10-26T18:42 |  | `logs/install.log` |
| cxx | passed | 0.1 | 2026-10-26T18:42 |  | `logs/cxx.log` |
| e2e | passed | 0.0 | 2026-10-26T18:42 |  | `logs/e2e.log` |
| phase3 | passed | 0.1 | 2026-10-26T18:42 |  | `logs/phase3.log` |
| complex | passed | 0.1 | 2026-10-26T18:42 |  | `logs/complex.log` |
| narrow | failed | 0.5 | 2026-10-26T18:42 | exit 1 | `logs/narrow.log` |
| probe | passed | 1.4 | 2026-10-26T18:44 |  | `logs/probe.log` |
| accuracy | passed | 0.0 | 2026-10-26T18:44 |  | `logs/accuracy.log` |
| accuracy-extended | passed | 0.1 | 2026-10-26T18:44 |  | `logs/accuracy-extended.log` |
| jax-suite | done | 74.8 | 2026-10-26T19:59 |  | `logs/jax-suite.log` |
| jax-heavy | done | 13.1 | 2026-10-26T20:12 |  | `logs/jax-heavy.log` |

JAX's tests so far:

| File | passed | failed | skipped | crashed |
|---|---|---|---|---|
| **total** | 18376 | 1080 | 3181 | 186 |

Details: [jax-suite/summary.md](jax-suite/summary.md) (each failure labeled METAL only or also on CPU); op probe: `re/findings/jax-op-probe.md`; extended accuracy: `re/findings/accuracy-extended.md`.
