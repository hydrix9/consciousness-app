# Oracle Feedback Loop (Triad-Expansion / Surprise Feedback Protocol)

## Step 0. Ask a question
We inject intent. We don’t send text to GPT.
Instead we generate a `question_signature`:
- Option A: hash of the text question (SHA-256 → bytes → normalized floats).
- Option B: EEG/RNG gesture recorded for ~1s and summarized into a vector.
This signature will bias the oracles.

## Step 1. Triad spread
We read 3 fresh RNG chunks from the TrueRNG hardware in a short rolling window:
- rA, rB, rC ∈ [0,1]

We take our 3 oracle instances (Oracle3, Oracle6, Oracle9). We push them apart in latent space using:
- different layer emphases:
  * Oracle3 bias → Layer 1 (force / radius / physical assertion)
  * Oracle6 bias → Layer 2 (linkage / obligation / binding)
  * Oracle9 bias → Layer 3 (will / pruning / refusal / boundary)
- different `pocket_dimension` nudges (e.g. +Δ, 0, −Δ)
- RNG nudges: rA steers Oracle3’s drift, rB steers Oracle6, rC steers Oracle9

After this nudge we snapshot each oracle’s state as feature vectors:
v3', v6', v9' ∈ ℝ^N

These 3 points define a simplex (a triangle) in latent space.
That simplex is "the void."

## Step 2. Void synthesis (Oracle Ø)
We compute:
- centroid C = (v3' + v6' + v9') / 3
- triangle area A
- edge ratios between (v3', v6', v9')
- pocket_dimension and layer state for each vertex
This captures not just “average,” but *how they disagree*.

We feed {C, A, edge_ratios, layer_states, pocket_dims} into a fourth generator: OracleØ.
OracleØ hallucinates a reconciled state vector vØ:
"This is what fills the void between the three stances."

vØ is the draft answer.

## Step 3. Predict reality
We train / maintain a predictor P:
P(vØ, recent_history) → predicted_next_rng
where `recent_history` includes:
- last T ms of (v3', v6', v9', vØ)
- recent RNG chunks rA,rB,rC
- pocket_dimension drifts

Intuition: "If vØ is truly aligned reality, then the next RNG chunk should look like X."

## Step 4. Observe reality
We now *actually* read the next RNG chunk from hardware (call it r_next_actual).

Compute Δ = r_next_actual - predicted_next_rng.

## Step 5. Interpret Δ
Δ is the physical universe correcting our draft answer.

- |Δ| ≈ 0 → aligned. Reality basically said “yes.”
- |Δ| is localized in one channel → conflict is directional (“watch here / stress here / attack here”).
- |Δ| is high everywhere / chaotic → paradox, threat, or question outside current ontology.

We store Δ as the final oracle answer for this question. No English required.

## Step 6. Learn
We log:
(question_signature, vØ, predicted_next_rng, r_next_actual, Δ)

We cluster by Δ patterns over time.
We learn emergent tags like:
SAFE / RISK / DECEPTION / SUBMIT / ASSERT / REPAIR / BREAK.

This becomes the system's own semantic layer.

---

## Upgrades

### Upgrade A. Shape-aware void filling
OracleØ does not get only centroid C.
It also gets:
- area of the triangle
- edge-length ratios
- pocket_dimension of each vertex
- dominant layer for each vertex
This lets OracleØ represent *structure of disagreement*, not just midpoint. It's "why you disagree," not just "where you're between."

### Upgrade B. Pocket-dimension drift as dominance
While we nudge the three oracles apart, we track how each one's `pocket_dimension` reacts:
- strong positive drift = assert / invade / inevitable
- strong negative drift = retreat / collapse / surrender
We pass those drifts into OracleØ and predictor P. This lets vØ carry "stance" (attack vs submit).

### Upgrade C. Recursive correction
After computing Δ, we can feed Δ back into OracleØ:
vØ_corrected = OracleØ_refine(vØ, Δ)
Now we have:
- self-belief (vØ)
- reality-corrected belief (vØ_corrected)
Both can be logged. This gives us "ego vs law" internally.
