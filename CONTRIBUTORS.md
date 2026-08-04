# Contributors & Toolchain

Every person, AI, and plugin that touched this repository — an honest ledger of how NOESIS was built.

---

## 👤 Author

| | Contributor | Role |
|---|---|---|
| 🧠 | **[Chandandeep Sharma](https://github.com/infinitule)** (`@infinitule`) | Creator & architect — concept, direction, PRISM foundation, all design decisions |

## 🤖 AI Pair

| | Contributor | Role |
|---|---|---|
| ⚡ | **[Claude Code](https://claude.com/claude-code)** (Anthropic) | Implementation partner — wrote the NOESIS engine, AutoMode, tests, docs, and diagrams across live pair-programming sessions |

## 🔌 Session Plugin Toolchain

The MCP plugins available in the build sessions. Marked by whether they actively contributed to this repository or stood by.

| Plugin | Status | Contribution |
|---|---|---|
| **GitHub MCP** | ✅ used | Branch pushes, pull requests #1–#6, squash merges, repo automation |
| **Malwarebytes ScamGuard** | 🟡 available | Link / email / phone threat-intelligence (not needed for this build) |
| **Figma MCP** | 🟡 available | Design-to-code and diagram tooling (not needed — diagrams are Mermaid + hand-built SVG) |
| **Gmail MCP** | 🟡 available | Email search & drafting (not needed for this build) |
| **Higgsfield MCP** | 🟡 available | Image / video / audio generation (not needed for this build) |
| **Hugging Face MCP** | 🟡 available | Model & dataset hub access (not needed for this build) |

## 🏗 Foundations

| Dependency | Role |
|---|---|
| **[PRISM](https://github.com/infinitule/PRISM)** | The photonic recursive substrate — `MetaMetaPrompt` P₀ seed, φ₁ optical memory rule, `SynapticEmbedder` MI entropy. Vendored as a git submodule. |
| **[anthropic](https://pypi.org/project/anthropic/)** | Claude Messages API SDK — the reasoning engine underneath every turn |
| **[NumPy](https://numpy.org)** | The attention vector's home — 64 floats, tanh-bounded |
| **[Rich](https://github.com/Textualize/rich)** | Terminal rendering for the AutoMode demo |
| **[pytest](https://pytest.org)** | All 59 tests, every merge gated on green |

---

*Want to contribute? Open an issue or PR at [infinitule/_nsn](https://github.com/infinitule/_nsn) — the blueprint in [`docs/BLUEPRINT.md`](docs/BLUEPRINT.md) maps every function you'd be touching.*
