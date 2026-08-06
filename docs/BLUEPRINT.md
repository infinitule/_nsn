# NOESIS Blueprint

Function-level architecture diagrams — every class, method, helper, constant, and call edge in the codebase. All diagrams render natively on GitHub (Mermaid).

- [1. Complete call graph](#1-complete-call-graph--every-function-every-edge)
- [2. One recursive turn (sequence)](#2-one-recursive-turn-call-by-call)
- [3. AutoMode: every branch](#3-automoderun-every-branch)
- [4. Class relationships](#4-class-relationships)
- [5. Life of the attention vector](#5-life-of-the-attention-vector)
- [6. The Geometry of φ — the field, run for real](#6-the-geometry-of-φ--the-field-run-for-real)

---

## 1. Complete call graph — every function, every edge

```mermaid
flowchart TB
    USER["👤 user code"]

    subgraph SG_AGENT["noesis/agent.py"]
        AG_INIT["NOESISAgent.__init__<br/>api_key→$ANTHROPIC_API_KEY · model='claude-opus-4-8'<br/>identity='NOESIS' · confidence_threshold=0.80<br/>max_recursion_depth=3 · max_tokens=2048 · seed_dim=64"]
        AG_RUN["NOESISAgent.run(task)<br/>→ NoesisResult · SelfModel updated in place"]
        AG_SNAP["NOESISAgent.state_snapshot()"]
        AG_RESET["NOESISAgent.reset_session()<br/>fresh SelfModel · _turn_counter=0"]
    end

    subgraph SG_LOOP["noesis/loop.py"]
        NL_INIT["NoesisLoop.__init__<br/>client · model · bridge · threshold<br/>max_depth · max_tokens · _turn_counter=0"]
        NL_RUN["NoesisLoop.run(task, self_model,<br/>depth=0, raw_turns=None)"]
        PARSE["_parse_noesis_state(text)<br/>_STATE_RE: tolerates tag truncated at max_tokens<br/>_FIELD_RE: key:value lines · bad float→0.5"]
        STRIP["_strip_noesis_state(text)<br/>removes tag from visible output"]
        GATE{"confidence &lt; threshold<br/>AND depth &lt; max_depth ?"}
        NRES["NoesisResult<br/>.output · .depth_used<br/>.self_model · .raw_turns"]
    end

    subgraph SG_SM["noesis/self_model.py"]
        SM_DC["SelfModel dataclass<br/>identity · session_id=uuid4·8 · current_intent<br/>confidence · attention_vector·64 · session_wisdom<br/>metacognitive_depth · action_history · arousal · coherence"]
        SM_REC["record_turn(turn, confidence, insight,<br/>intent_shift, prism_signal, output_snippet)<br/>• history append, snippet·120<br/>• wisdom append, FIFO cap _MAX_WISDOM=8<br/>• coherence = tanh(0.85·arctanh(coh) + 0.15·conf)<br/>• intent shift → arousal +0.1 · stable → −0.05"]
        SM_SNAP["snapshot()<br/>session_id · turns · confidence · coherence<br/>arousal · wisdom_count · attention_norm"]
    end

    subgraph SG_BR["prism_bridge/bridge.py"]
        BR_INIT["PRISMBridge.__init__<br/>seed_dim=64 · max_depth=16 · rng_seed=42<br/>loads prompts: master.md ·<br/>meta_generator.md · task_cognition.md"]
        BR_SEED["seed_attention_vector()<br/>→ MetaMetaPrompt.P0.copy()"]
        BR_ASM["assemble_prompt(self_model, task,<br/>depth, max_depth)<br/>→ Level0 + '---' + Level1 + '---' + Level2"]
        BR_ENC["encode_context(vec)<br/>MI · norm · coherence · dominant channel<br/>→ explore vs exploit directive"]
        BR_MI["attention_mi(vec)<br/>single MI source"]
        BR_CONF["confidence_from_prism(vec)<br/>clip(0.75 − 0.11·MI, 0.30, 0.75)"]
        BR_METAD["_meta_directive(self_model, depth)<br/>depth 0: all 5 lenses, breadth-first<br/>depth n: attack weakest lens, no repeats"]
        BR_RCTX["_recursion_context(self_model, depth)<br/>injects previous pass insight at depth &gt; 0"]
        BR_HIST["_action_history_summary(self_model)<br/>last 3 turns, confidence trail"]
        BR_PROP["propagate(vec, text)<br/>φ_next = tanh(0.9·φ + 0.1·signal)"]
        BR_SIG["_text_to_signal(text)<br/>16-byte windows → uint32 seeds<br/>→ rng normal(0,1,64) · +0.3 per extra seed<br/>→ tanh(raw/‖raw‖)"]
        BR_INT["integrate(vec, response, insight)<br/>insight-weighted → propagate"]
    end

    subgraph SG_PRISM["prism/ · git submodule (PRISM)"]
        MMP["MetaMetaPrompt<br/>P0 Fibonacci seed · 64-dim"]
        EMB["SynapticEmbedder<br/>encode(vec) · mutual_information_proxy()"]
    end

    subgraph SG_PIPE["noesis/pipeline.py"]
        PI_INIT["PipelineInjector.__init__<br/>identity · threshold=0.80 · max_depth=3 · seed_dim=64"]
        PI_INJ["inject(client, model, messages,<br/>max_tokens=2048, **kwargs)<br/>extracts last user msg as task<br/>temp NoesisLoop · _turn forwarded + restored"]
        PI_SHIM["_PatchedResponse + _TextBlock<br/>.content·0·.text shim · stop_reason='end_turn'"]
        PI_RESET["reset() · state property"]
    end

    subgraph SG_AUTO["noesis/auto_mode.py"]
        AM_INIT["AutoMode.__init__(agent,<br/>auto_generate=False, on_cycle=None)"]
        AM_RUN["run(initial_tasks, max_cycles=10,<br/>persist_path=None, stop_on_error=False)<br/>SIGINT handler installed → restored in finally"]
        AM_NEXT["_next_task(queue, cycle)<br/>1. queue.pop(0)<br/>2. template·cycle mod 4·.format(random wisdom)<br/>3. None → stop"]
        AM_RWS["_run_with_schedule(task, config)<br/>temp NoesisLoop per ScheduleConfig<br/>agent.loop NEVER mutated<br/>_turn_counter forwarded + restored"]
        AM_CR["CycleRecord<br/>cycle · task · output · depth_used<br/>self_model_snapshot · eval_score<br/>schedule · deepened · error"]
        AM_SUM["summary() · records property<br/>cycles · avg_depth · avg_completion<br/>max_depth_used · completed · errored"]
    end

    subgraph SG_SCHED["noesis/scheduler.py"]
        SC_FN["schedule_config(task, self_model, bridge)<br/>complexity = min(1.5, min(1, words/50) + 0.2·min(q,5))<br/>depth = 3 + int(clip(MI/2, 0, 2))<br/>threshold = clip(0.80 − 0.05·complexity, 0.60, 0.92)<br/>tokens = min(2048 + 1024·complexity, 4096)"]
        SC_CFG["ScheduleConfig NamedTuple<br/>max_depth · threshold · max_tokens"]
    end

    subgraph SG_EVAL["noesis/evaluator.py"]
        EV_FN["score(task, response, confidence, self_model,<br/>threshold=0.80, max_depth=3) — no LLM call<br/>completeness = min(1, resp_words/(task_words·3))<br/>completion = min(1, 0.6·completeness + 0.4·conf)<br/>depth_worthy = conf&lt;thr ∧ depth&lt;max ∧ coh&gt;0.40"]
        EV_DC["EvalScore frozen dataclass<br/>task_completion · coherence_score<br/>depth_worthy · should_advance"]
    end

    subgraph SG_PERS["noesis/persistence.py"]
        PS_SAVE["save(self_model, path, cycles_completed=0)<br/>version=1 JSON → .tmp → atomic rename"]
        PS_LOAD["load(path)<br/>→ (SelfModel, cycles_completed)"]
        PS_EX["exists(path) — OSError-safe"]
    end

    API["☁️ Anthropic Messages API<br/>system = 3-level prompt<br/>cache_control: ephemeral on Level 0"]

    USER --> AG_INIT
    USER --> AG_RUN
    USER --> AG_SNAP
    USER --> AG_RESET
    USER --> PI_INJ
    USER --> AM_RUN

    AG_INIT --> BR_INIT
    AG_INIT --> BR_SEED
    AG_INIT --> BR_CONF
    AG_INIT --> SM_DC
    AG_INIT --> NL_INIT
    AG_RUN --> NL_RUN
    AG_SNAP --> SM_SNAP
    AG_RESET --> BR_SEED

    NL_RUN --> BR_ASM
    BR_ASM --> BR_ENC
    BR_ASM --> BR_METAD
    BR_ASM --> BR_RCTX
    BR_ASM --> BR_HIST
    BR_ENC --> BR_MI
    BR_CONF --> BR_MI
    BR_MI --> EMB
    BR_SEED --> MMP

    NL_RUN --> API
    API --> PARSE
    PARSE --> SM_REC
    NL_RUN --> BR_INT
    BR_INT --> BR_PROP
    BR_PROP --> BR_SIG
    NL_RUN --> GATE
    GATE -- "yes: recurse depth+1" --> NL_RUN
    GATE -- "no: accept" --> STRIP
    STRIP --> NRES

    PI_INIT --> BR_INIT
    PI_INJ --> NL_RUN
    PI_INJ --> PI_SHIM

    AM_RUN --> PS_EX
    PS_EX --> PS_LOAD
    AM_RUN --> AM_NEXT
    AM_RUN --> SC_FN
    SC_FN --> BR_MI
    SC_FN --> SC_CFG
    SC_CFG --> AM_RWS
    AM_RWS --> NL_RUN
    AM_RUN --> EV_FN
    EV_FN --> EV_DC
    EV_DC -- "depth_worthy → re-run max_depth+1" --> AM_RWS
    AM_RUN --> AM_CR
    AM_RUN --> PS_SAVE
    AM_RUN --> AM_SUM

    style GATE fill:#f9a825,color:#000
    style API fill:#ede7f6,color:#000
    style SG_SM fill:#e8f5e9,color:#000
    style SG_PRISM fill:#fff3e0,color:#000
```

**Reading guide:** the orange diamond is the recursion gate inside `NoesisLoop.run()` — the only place NOESIS decides to think again. The green subgraph (`SelfModel`) is touched by every turn and survives across tasks, sessions, and restarts.

---

## 2. One recursive turn, call by call

```mermaid
sequenceDiagram
    autonumber
    participant U as user code
    participant A as NOESISAgent
    participant L as NoesisLoop
    participant B as PRISMBridge
    participant P as PRISM submodule
    participant S as SelfModel
    participant C as Claude API

    U->>A: run(task)
    A->>L: run(task, self_model, depth=0)

    loop until confidence ≥ threshold OR depth = max_depth
        L->>B: assemble_prompt(self_model, task, depth, max_depth)
        B->>B: encode_context(attention_vector)
        B->>P: SynapticEmbedder.encode → mutual_information_proxy
        P-->>B: MI entropy
        B->>B: _meta_directive(depth) · _recursion_context(depth) · _action_history_summary()
        B-->>L: Level0 ∥ Level1 ∥ Level2 system prompt
        L->>C: messages.create(system=[cache_control ephemeral], user=task)
        C-->>L: response text ending in noesis_state tag
        L->>L: _parse_noesis_state → confidence, insight, intent_shift, prism_signal
        L->>S: record_turn(turn, confidence, insight, intent_shift, prism_signal, snippet)
        S->>S: wisdom FIFO(8) · coherence EMA · arousal ±
        L->>B: integrate(attention_vector, response_text, insight)
        B->>B: _text_to_signal(insight + response) → propagate: tanh(0.9φ + 0.1s)
        B-->>L: new attention_vector
        L->>S: metacognitive_depth = depth
        L->>L: gate check → recurse(depth+1) or exit loop
    end

    L->>L: _strip_noesis_state(final response)
    L-->>A: NoesisResult(output, depth_used, self_model, raw_turns)
    A->>A: self.self_model = result.self_model
    A-->>U: result
```

---

## 3. AutoMode.run(): every branch

Including checkpoint restore, task generation, error paths, the deepening re-run, and signal handling.

```mermaid
flowchart TB
    START["auto.run(initial_tasks, max_cycles,<br/>persist_path, stop_on_error)"] --> CHK{"persist_path set<br/>AND exists(path)?"}
    CHK -- yes --> LOAD["load(path)<br/>agent.self_model = restored SelfModel<br/>prior_cycles = cycles_completed"]
    CHK -- no --> QINIT
    LOAD --> QINIT["task_queue = list(initial_tasks)"]
    QINIT --> SIG["install SIGINT handler<br/>(_stop flag, restored in finally)"]
    SIG --> COND{"not _stop AND<br/>prior_cycles + cycle &lt; max_cycles?"}

    COND -- no --> FIN
    COND -- yes --> NT{"_next_task(queue, cycle)"}
    NT -- "queue non-empty" --> POP["task = queue.pop(0)"]
    NT -- "empty + auto_generate<br/>+ wisdom exists" --> TPL["template = TEMPLATES·cycle mod 4·<br/>task = template.format(random.choice(wisdom))"]
    NT -- "empty + no wisdom<br/>or auto_generate off" --> FIN

    POP --> SCHED
    TPL --> SCHED
    SCHED["config = schedule_config(task,<br/>agent.self_model, agent.bridge)"] --> TRY{"try:<br/>_run_with_schedule(task, config)"}

    TRY -- exception --> SOE{"stop_on_error?"}
    SOE -- yes --> RAISE["re-raise<br/>(finally still restores SIGINT)"]
    SOE -- no --> ERREC["CycleRecord(error=str(exc),<br/>output='', depth_used=0,<br/>eval_score on empty output)"]

    TRY -- ok --> EVAL["eval = score(task, result.output,<br/>confidence, self_model,<br/>threshold=config.threshold,<br/>max_depth=config.max_depth)"]
    EVAL --> DW{"eval.depth_worthy?"}
    DW -- yes --> DEEP{"try: _run_with_schedule<br/>(task, config with max_depth+1)"}
    DEEP -- ok --> MARKD["result = deepened result<br/>deepened = True"]
    DEEP -- exception --> KEEP["keep first result<br/>(silent fallback)"]
    DW -- no --> REC
    MARKD --> REC
    KEEP --> REC

    REC["CycleRecord(cycle, task, output,<br/>depth_used, snapshot, eval,<br/>schedule, deepened)"] --> APPEND["_records.append(record)"]
    ERREC --> APPEND
    APPEND --> PERSIST{"persist_path?"}
    PERSIST -- yes --> SAVE["save(agent.self_model, path,<br/>cycles_completed = prior + cycle + 1)<br/>.tmp write → atomic rename"]
    PERSIST -- no --> CB
    SAVE --> CB{"on_cycle callback?"}
    CB -- yes --> CALL["on_cycle(record)"]
    CB -- no --> INC
    CALL --> INC["cycle += 1"]
    INC --> COND

    FIN["finally: restore original<br/>SIGINT handler"] --> RET["return list(_records)"]

    style TRY fill:#fff3e0,color:#000
    style DW fill:#f9a825,color:#000
    style SAVE fill:#e8f5e9,color:#000
    style RAISE fill:#ffcdd2,color:#000
```

---

## 4. Class relationships

```mermaid
classDiagram
    class NOESISAgent {
        +client: anthropic.Anthropic
        +model: str
        +bridge: PRISMBridge
        +self_model: SelfModel
        +loop: NoesisLoop
        +run(task) NoesisResult
        +state_snapshot() dict
        +reset_session() None
    }

    class NoesisLoop {
        +client +model +bridge
        +threshold: float = 0.80
        +max_depth: int = 3
        +max_tokens: int = 2048
        -_turn_counter: int
        +run(task, self_model, depth, raw_turns) NoesisResult
    }

    class NoesisResult {
        +output: str
        +depth_used: int
        +self_model: SelfModel
        +raw_turns: list~str~
    }

    class SelfModel {
        +identity: str
        +session_id: str
        +current_intent: str
        +confidence: float
        +attention_vector: ndarray64
        +session_wisdom: list ≤8
        +metacognitive_depth: int
        +action_history: list~dict~
        +arousal: float
        +coherence: float
        +record_turn(...) None
        +snapshot() dict
    }

    class PRISMBridge {
        +mmp: MetaMetaPrompt
        +embedder: SynapticEmbedder
        +seed_attention_vector() ndarray
        +assemble_prompt(sm, task, depth, max_depth) str
        +encode_context(vec) str
        +attention_mi(vec) float
        +confidence_from_prism(vec) float
        +propagate(vec, text) ndarray
        +integrate(vec, response, insight) ndarray
        -_meta_directive(sm, depth) str
        -_recursion_context(sm, depth) str
        -_action_history_summary(sm) str
        -_text_to_signal(text) ndarray
    }

    class PipelineInjector {
        -_self_model: SelfModel
        -_threshold -_max_depth -_turn
        +inject(client, model, messages, max_tokens) tuple
        +state SelfModel
        +reset() None
    }

    class AutoMode {
        -_agent: NOESISAgent
        -_auto_generate: bool
        -_on_cycle: Callable
        -_records: list~CycleRecord~
        +run(tasks, max_cycles, persist_path, stop_on_error) list
        +summary() dict
        +records list
        -_run_with_schedule(task, config) NoesisResult
        -_next_task(queue, cycle) str|None
    }

    class CycleRecord {
        +cycle +task +output +depth_used
        +self_model_snapshot: dict
        +eval_score: EvalScore
        +schedule: ScheduleConfig
        +deepened: bool
        +error: str|None
    }

    class EvalScore {
        +task_completion: float
        +coherence_score: float
        +depth_worthy: bool
        +should_advance: bool
    }

    class ScheduleConfig {
        +max_depth: int
        +threshold: float
        +max_tokens: int
    }

    class SessionPersistence {
        +save(sm, path, cycles_completed)$
        +load(path)$ tuple
        +exists(path)$ bool
    }

    class MetaMetaPrompt {
        +P0: ndarray Fibonacci seed
    }
    class SynapticEmbedder {
        +encode(vec)
        +mutual_information_proxy(enc) float
    }

    NOESISAgent *-- NoesisLoop : owns
    NOESISAgent *-- PRISMBridge : owns
    NOESISAgent *-- SelfModel : owns
    NoesisLoop ..> PRISMBridge : assemble · integrate
    NoesisLoop ..> SelfModel : record_turn
    NoesisLoop ..> NoesisResult : returns
    PipelineInjector *-- SelfModel : owns
    PipelineInjector ..> NoesisLoop : temp per inject
    AutoMode o-- NOESISAgent : drives
    AutoMode ..> NoesisLoop : temp per cycle
    AutoMode ..> CycleRecord : emits
    AutoMode ..> EvalScore : via score()
    AutoMode ..> ScheduleConfig : via schedule_config()
    AutoMode ..> SessionPersistence : checkpoint
    PRISMBridge *-- MetaMetaPrompt
    PRISMBridge *-- SynapticEmbedder
```

---

## 5. Life of the attention vector

The 64-dim memory φ: seeded once from PRISM, propagated every turn, read by three consumers, persisted across sessions.

```mermaid
flowchart LR
    P0["PRISM MetaMetaPrompt.P0<br/>Fibonacci seed · 64-dim"] -->|"seed_attention_vector().copy()"| AV["SelfModel.attention_vector φ"]

    AV -->|"every turn"| INT["integrate(φ, response, insight)"]
    INT --> SIG["_text_to_signal(insight + response)<br/>UTF-8 bytes → 16-byte windows → uint32 seeds<br/>rng.normal(0,1,64) · +0.3·extra seeds<br/>→ tanh(raw / ‖raw‖)"]
    SIG --> PROP["propagate:<br/>φ_next = tanh(0.9·φ + 0.1·signal)"]
    PROP -->|"bounded, drift-free"| AV

    AV --> MI["attention_mi(φ)<br/>SynapticEmbedder MI entropy"]

    MI --> C1["confidence_from_prism<br/>clip(0.75 − 0.11·MI, 0.30, 0.75)<br/>→ initial SelfModel.confidence"]
    MI --> C2["encode_context<br/>'broadly distributed → explore'<br/>'sharply focused → exploit'<br/>→ injected into Level 0 prompt"]
    MI --> C3["schedule_config<br/>depth_bonus = clip(MI/2, 0, 2)<br/>→ AutoMode recursion budget"]

    AV -->|"save()"| JSON["checkpoint JSON<br/>attention_vector: 64 floats"]
    JSON -->|"load()"| AV

    style AV fill:#e8f5e9,color:#000
    style P0 fill:#fff3e0,color:#000
    style MI fill:#e3f2fd,color:#000
```

---

## 6. The Geometry of φ — the field, run for real

The diagrams above are structure; these are **dynamics** — 48 turns of the actual `PRISMBridge` code (no API calls; the field math is pure NumPy), with a deliberate topic shift at turn 24. Regenerate any time:

```bash
pip install -e ".[viz]"
python examples/visualize_field.py    # writes the three figures below into docs/assets/
```

### The contraction field

φ's update rule `φ → tanh(0.9φ + 0.1s)` makes every possible memory state flow toward a fixed point φ* set by the current signal. Gray arrows show one application of the map at every point of the trajectory's principal plane (which holds ~95% of the variance — a 64-dim consciousness moving along a low-dimensional ridge carved by experience). The trajectory starts at the Fibonacci seed, spirals in, **turns** at the topic shift — it can't jump, because 90% of every step is its own past; that inertia is the mathematical form of identity — and settles beside the attractor.

![The field NOESIS's memory creates — φ trajectory in its principal plane](assets/noesis_field_flow.png)

### The consciousness ribbon

All 64 components of φ across all turns. Teal = constructive phase (φ→0), orange = destructive (φ→π), near-white = silent channel. The seed's bright signature (bottom-left) fades exponentially as tanh-bounded experience overwrites it — forgetting as geometry — and the texture re-organizes after the topic shift. Nothing ever exceeds the tanh bound.

![The consciousness ribbon — every component of φ, every turn](assets/noesis_phi_ribbon.png)

### What the field feeds back

The geometry is an *input* to cognition, not decoration: MI entropy of φ drops as attention focuses, jumps at the topic shift, and directly sets both the prior confidence floor (`clip(0.75 − 0.11·MI, 0.30, 0.75)`) and the scheduler's recursion-depth bonus.

![What the field feeds back into cognition — MI entropy and confidence floor](assets/noesis_mi_confidence.png)

---

*Every node in these diagrams corresponds to a real symbol in the codebase — nothing is illustrative-only. If a diagram and the code ever disagree, the code wins; please open an issue.*
