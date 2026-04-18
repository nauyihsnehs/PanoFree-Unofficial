## Cost Dimension for LLM-Assisted Coding

## 1 Cost-control objective

The practical objective is:

> **Use the smallest model and the smallest context that can reliably solve the current subtask, and reserve the strongest model only for architecture, ambiguity resolution, and difficult debugging.**

This is especially important because official API pricing for strong reasoning/coding models is materially higher on output tokens than on input tokens, and both OpenAI and Anthropic explicitly provide discounted cached-input mechanisms that reward stable reusable prefixes. :contentReference[oaicite:1]{index=1}

---

## 2 Core principles for reducing token usage

### A. Prefer **task decomposition over long-context conversation**

Do not keep one endlessly growing coding chat.

Instead, split the work into:

- geometry utilities
- scheduler
- warping
- mask logic
- risky-area module
- guidance module
- vertical expansion
- stitching / debugging

Each subtask should have its own compact prompt state.

**Why:** repeated re-sending of old context is one of the fastest ways to waste tokens.

---

### B. Send **diffs and interfaces**, not full files

When asking the model to modify code, prefer sending:

- function signatures
- relevant class definitions
- local call sites
- minimal failing snippet
- specific diff target

Avoid sending:

- the whole repository
- large unchanged files
- duplicate logs
- repeated background already reflected in code comments

A good default rule is:

> **Only send the smallest slice that is sufficient to make a correct local edit.**

---

### C. Convert long history into a **state summary**

After a module becomes stable, compress the conversation into:

- current module goal
- decisions already fixed
- APIs already frozen
- unresolved bugs
- known assumptions
- next action

Then use that summary as the new context root.

This reduces both token cost and inconsistency caused by the model rereading old exploratory reasoning.

---

### D. Freeze stable specs early

Once a decision is made, turn it into a short canonical spec:

- camera convention
- coordinate system
- file layout
- naming convention
- tensor/image shape convention
- projection formulas
- scheduler assumptions

Then reuse only that spec, not the whole discussion that led to it.

This improves cacheability and reduces re-explanation overhead.

---

## 3 Core principles for reducing call count

### A. Ask for **one complete local patch**, not many micro-edits

Bad pattern:

- “write function header”
- “now write loop”
- “now add mask”
- “now add comments”

Better pattern:

- “Implement this entire module with these constraints, these inputs/outputs, and these tests.”

One well-scoped request is often cheaper than 5–10 tiny interactive rounds.

---

### B. Ask for **self-check before returning**

A useful prompt pattern is:

> “Before finalizing, verify shape consistency, variable naming consistency, boundary conditions, and obvious runtime errors. Then return one final patch.”

This slightly increases one-call reasoning, but often reduces total retries.

---

### C. Request **assumption listing up front**

For under-specified modules, ask the model to first output:

1. assumptions
2. implementation plan
3. final code

This avoids repeated correction loops caused by hidden assumptions.

---

### D. Separate **design calls** from **implementation calls**

Use strong models for:

- architecture
- ambiguous paper interpretation
- debugging hard failures
- reconciling inconsistent modules

Use cheaper/faster models for:

- boilerplate implementation
- refactoring
- docstrings
- test generation
- argument renaming
- small utility functions

A cost-efficient coding pipeline is usually **hierarchical**, not single-model.

---

## 4 Recommended model allocation strategy

### Tier 1 — cheap / fast model

Use for:

- boilerplate
- wrappers
- config files
- repetitive transforms
- logging
- test scaffolding
- documentation formatting

### Tier 2 — strong general coding model

Use for:

- module implementation
- medium debugging
- refactoring across a few files
- turning pseudocode into code

### Tier 3 — highest-end model

Use only for:

- paper-to-code interpretation gaps
- hard bug isolation
- architecture decisions
- resolving conflicting evidence
- final review of critical modules

In other words:

> **Do not spend Opus / top-tier GPT tokens on work that Sonnet / mid-tier GPT can already do reliably.**

---

## 5 Prompt-shaping tactics that materially save tokens

### Tactic 1: Start with a strict output contract

Example:

- “Return only Python code.”
- “Return unified diff only.”
- “Return one markdown table only.”
- “Do not restate the paper.”
- “Do not explain unless necessary.”

This directly suppresses unnecessary output-token burn.

---

### Tactic 2: Force local scope

Example:

- “Only modify these three functions.”
- “Do not rename public APIs.”
- “Do not touch unrelated files.”
- “Preserve current scheduler behavior.”

This reduces both output size and follow-up repair calls.

---

### Tactic 3: Provide acceptance tests in the same prompt

Include:

- expected shapes
- 1–2 sample inputs
- edge cases
- expected invariants

This often reduces iteration count more than it increases prompt size.

---

### Tactic 4: Ask for a failure-oriented review

Instead of:

- “Is this good?”

Use:

- “Find the top 5 likely correctness failures.”
- “Find hidden shape mismatches.”
- “Check for coordinate convention inconsistencies.”

Targeted review prompts are much more token-efficient than generic review prompts.

---

## 6 Repository handling strategy

### Avoid full-repo ingestion by default

A coding LLM does **not** need the entire repo for most tasks.

Use a staged repository policy:

#### Level 1 — local context only

- target file
- imported helper file
- relevant test

#### Level 2 — module context

- all files in the same subsystem

#### Level 3 — architecture context

- only when cross-module design is necessary

This prevents the common failure mode of paying for massive context that is not actually used.

---

### Maintain a machine-readable project memo

Keep a compact file such as:

- `project_state.md`
- `module_specs.md`
- `camera_conventions.md`
- `repro_assumptions.md`

Then pass only the relevant memo section instead of re-explaining decisions each time.

This is especially beneficial when the provider supports cached prompt prefixes, because stable repeated context is exactly what caching is designed for. Anthropic explicitly exposes prompt caching and cache-hit pricing, and OpenAI also prices cached input separately at a discount. :contentReference[oaicite:2]{index=2}

---

## 7 Cost-aware workflow for this specific reproduction project

For the **PanoFree full-spherical reproduction**, a cost-efficient LLM workflow would be:

### Step 1 — one strong design pass

Use a strong model once to define:

- module boundaries
- data flow
- camera conventions
- scheduler conventions
- known paper ambiguities
- reproduction assumptions

Output:

- one frozen implementation spec

### Step 2 — cheap implementation passes per module

For each module:

- pass only the frozen spec excerpt
- pass only the target file(s)
- request one complete local patch
- request minimal comments

### Step 3 — targeted debug calls

When a bug appears, provide:

- exact error
- minimal stack trace
- minimal reproducer
- relevant code slice only

Do **not** reopen the whole project context.

### Step 4 — sparse strong-model audits

Only after 2–3 modules are completed:

- ask a strong model to check consistency across modules

This is much cheaper than using the strongest model for every coding turn.

---

## 8 Specific ways to reduce output-token cost

Because output tokens are often expensive relative to input tokens in current frontier APIs, the following are especially valuable: :contentReference[oaicite:3]{index=3}

- request **patches instead of full rewritten files**
- request **concise justifications**
- suppress repeated restatement of requirements
- ask for **bulleted bug lists**, not essays
- ask for **only changed functions**
- ask for **tests only when the code is stable enough to deserve them**
- avoid “teach me every line” mode during active implementation

---

## 9 Specific ways to reduce retries

Retries are often more expensive than a slightly better initial prompt.

To reduce retries:

- define naming conventions before coding
- define tensor/image shape conventions before coding
- define coordinate system before coding
- define acceptable shortcuts before coding
- define what may be approximated from the paper before coding

A large fraction of wasted calls in paper reproduction comes from hidden disagreement on assumptions, not from raw coding difficulty.

---

## 10 Practical call-budget policy

A useful default budget per module is:

- **1 design call**
- **1 implementation call**
- **1 repair/debug call**
- **optional 1 review call**

If a module exceeds this routinely, the prompt spec is probably under-constrained.

That is often a better diagnosis than “the model is bad”.

---

## 11 When to pay for a stronger model

Use a stronger model only when at least one of the following is true:

- the paper is ambiguous and multiple interpretations are plausible
- multiple modules conflict and require reconciliation
- the bug is conceptual rather than syntactic
- the failure requires long-chain reasoning across geometry + scheduling + generation
- previous cheaper-model attempts already converged to the wrong pattern

Otherwise, stronger models may improve elegance, but not cost efficiency.

---

## 12 Final recommendation

For LLM-assisted coding, the cheapest reliable workflow is usually:

> **Strong model once for spec, cheaper model repeatedly for local implementation, strong model again only for hard debugging or cross-module review.**

The main savings usually come from:

1. smaller context windows per call,
2. fewer retries,
3. reuse of cached stable prefixes,
4. patch-oriented outputs,
5. disciplined module boundaries.

In practice, this matters more than small differences in raw per-token price between frontier coding models.