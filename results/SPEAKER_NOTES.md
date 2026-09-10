# ErwinParT — full deck speaker notes

`ErwinParT_full_deck.pdf` · 23 slides · 16:9

**The arc.** Slides 1–7 are the *idea* (ball-tree attention, why it should be cheaper).
Slides 8–11 are *what the partitioning actually looks like* on real jets. Slides 12–23 are
*did it work* — this cycle's measurements.

**Say once, early (slide 1 or 12):** there is **no ParT accuracy baseline** in this deck.
ParT could not be trained on the available hardware. Every accuracy number is ErwinParTv2
against itself or against its own dense-attention control.

**One inconsistency to get ahead of.** The v1 figures (slides 8–11) use a *single* attention
level with ball size m ∈ {16, 32, 64} and jets padded to 128. v2 uses **m = 8 with a
three-level hierarchy at L = 64**. If someone asks why m changed: the sweep on slide 17 is
exactly that question, and m = 8 won.

---

## Part I — the idea (slides 1–7, original deck)

### 1 · Title — ErwinParT
- Erwin is a tree-based hierarchical transformer for large-scale physical systems (ICML 2025).
- The question this deck opens: can it replace ParT's global attention for jet tagging?
- Note the date — this was the proposal; everything from slide 12 is the follow-through.

### 2 · Standard self-attention
- ParT's particle attention block: `softmax(QKᵀ/√d + U)V`, where **U** is the pre-softmax
  pairwise interaction bias — ParT's key contribution.
- No locality constraint: every constituent attends to every other.
- **Cost is O(N²)** — this is the thing we are trying to remove.
- *Flag forward:* slide 19 shows that removing U entirely costs nothing measurable. Do not
  reveal that yet, but be ready if someone asks "isn't U the whole point of ParT?"

### 3 · What a ball is
- A ball is a region bounded by a hypersphere: centre **c**, radius **r**.
- Keep this short — it is a definition slide.

### 4 · What a ball tree is
- A hierarchical sequence of partitions L₀ … L_m; each level covers the point set with
  disjoint balls; at leaf level the nodes are the individual particles.
- **Point to make:** the partition is *geometric*, built from the constituent positions —
  it is not learned. That matters later: the tree is index math under `no_grad`.

### 5 · Ball-tree attention
- Pick a level k; attention runs independently inside each ball at that level.
- Larger balls capture longer-range structure; smaller balls are cheaper.
- **This is the whole trade-off in one sentence** — and slide 17 is the experiment that
  measures where on that trade-off we should sit.

### 6 · Computational cost
- Attention is computed per ball, so cost falls from quadratic to **quadratic in ball size,
  linear in the number of balls**: O(m²·n/m) = O(n·m).
- *Caveat to voice:* this is the cost of the attention *mechanism*. Slide 21 shows the
  whole-model FLOPs, which scale more gently because linear layers dominate at jet sizes.
  Both are true; do not let the audience conflate them.

### 7 · First ErwinParT run
- Ball attention built into our efficient particle transformer; run on 10% of JetClass on a
  4090. This was v1.
- **Bridge:** "That version had a per-jet Python loop and no real ball tree — it sorted by
  ΔR and chunked. Everything from slide 12 is the rebuild."

---

## Part II — what the partitioning looks like (slides 8–11, v1 figures)

### 8 · How the tree groups constituents
- One real top jet, n = 120, in the (Δη, Δφ) plane. Marker size ∝ log pT. Colour = ball.
- Left to right: m = 16 (8 balls) → m = 32 (4) → m = 64 (2).
- **Point out:** the groups are spatially compact and roughly radial around the jet axis —
  the tree is finding angular neighbourhoods, which is what QCD's angular ordering says
  should matter.
- **Do not claim** these colours correspond to physical subjets; they are geometric
  partitions, not a clustering algorithm's output.

### 9 · Each ball is a subjet
- Same jet, same trees, now showing the **pT-weighted centroid** of each ball; grey points
  are the constituents.
- **Point out:** as m grows the centroids collapse toward the jet axis — coarser balls
  average over more of the jet. At m = 16 the centroids trace the substructure; at m = 64
  there are only two and they straddle the core.
- **This is the visual intuition for coarsening**: in v2, each of these centroids carries
  the *summed 4-momentum* of its ball, which makes it a subjet in the physics sense.

### 10 · O(n·m) against O(n²)
- Idealised token-pair count per attention block, log-log, against constituent multiplicity.
- **Point out the shaded band — JetClass lives at n ≤ 128, and inside it the lines have
  barely separated.** The asymptotic argument is real but pays off outside the region our
  jets occupy.
- **This is the honest set-up for slide 21**, which measures the same thing on the real
  model and finds 1.5–1.9×, not orders of magnitude. Saying it here first makes slide 21
  a confirmation rather than a climb-down.
- **Do not claim** these are measured costs — they are counted token pairs, excluding tree
  construction and permutation overhead (the figure's own caption says so).

### 11 · The same partitioning across topologies
- Three jet classes (t→bqq′, W→qq, QCD) × three ball sizes.
- **Point out:** the two-prong W jet and the diffuse QCD jet get visibly different
  partitions — the tree adapts to topology without being told the class.
- **Bridge:** "So the partitioning behaves sensibly. The question is whether it costs
  accuracy — that is what we measured this cycle."

---

## Part III — did it work (slides 12–23, v2 results)

### 12 · Title
- One question: does a jet need every constituent to see every other one?
- **Say the caveat in red at the bottom out loud.** No ParT baseline.

### 13 · The analysis in one line
- Build → verify → train → **ablate** → **cost**. The two highlighted boxes are where the
  science is; build and verify are hygiene.
- Mention 117 verification checks once and move on.

### 14 · What we built
- Attention runs *inside* each grey ball, never across. Pooling halves the nodes and sums
  4-momenta, so a level-1 node is literally a subjet.
- **The bottom level is a single ball** — the only place where everything sees everything.
- **Claim:** locality at fine scales, globality recovered by coarsening.
- **Avoid:** do not claim the coarse levels encode physics we verified — the statistics
  behave like subjet splittings, but slide 19 shows it does not convert into accuracy.

### 15 · Why 64 slots per jet
- Left: the multiplicity distribution and where we cut. Truncation is **pT-ordered** — we
  drop 1.4% of particles but only **0.115% of the jet pT**.
- Right: the cut is class-biased — Hgg 24%, Tbqq 14%, everything else near zero.
- **Claim:** 93.7% of jets fit; ParT's L = 128 removes the last 0.1% and pays 69% padding.
- **Avoid:** do not say truncation is harmless yet — that is the next slide's job.

### 16 · Does truncation cost us anything?
- Retrained at L = 128, which truncates nothing, everything else fixed.
- **Right panel, left group: on the 93.7% of jets L=64 keeps whole, the two models are
  identical to four decimals (p = 0.971).** Middle: on the 6.3% it truncates, L=128 is 3.7
  points better. Right: washes out overall.
- Left panel: Hgg's two dots sit on top of each other despite 24% truncation.
- **Claim:** truncation has a real but strictly confined cost; it is not why Hgg is hard.
- **Avoid:** "truncation is free". It costs 3.7 points on 6% of jets — a trade we made.

### 17 · Is ball locality costing us accuracy?  ← **the core slide**
- Ball-size sweep, everything else fixed, **2.21 M parameters at every m**.
- **m = 64 means one ball containing all constituents — that is full dense attention**
  inside our own hierarchy. The cleanest control available: not ParT, but the same model
  with locality switched off.
- **Claim:** dense attention is **not significantly better** than 8-particle balls
  (paired McNemar, p = 0.295), at 2.1× the cost.
- **Avoid:** "locality is free" without the detection limit — we can exclude a large
  effect, not a small one (±0.006). Error bars are test-set sampling, not run-to-run.

### 18 · Per-class check  *(hold in reserve)*
- Four ball sizes overlaid per class; curves indistinguishable everywhere, including the
  two-prong Hbb/Hcc where you would most expect locality to hurt.
- **Claim:** the null is not an averaging artefact.
- **Avoid:** the plateaus at the top are the ~1,000-jet measurement ceiling, not performance.
- Skip unless challenged.

### 19 · Which parts actually matter?
- Everything except the readout sits inside the detection band.
- **Own the correction:** we predicted ParT's interaction bias would matter — it is the
  mechanism we built `pair_features.py` for — and **removing it changes nothing
  measurable (p = 0.823)**. Two untested explanations: ball-locality excludes the
  wide-angle pairs, or 80k jets is too small a budget for a subtle bias term.
- **Claim:** the hierarchy does the work. Constituents-only readout is significantly worse
  (p = 0.024); bottleneck-only is indistinguishable — so the *coarse* levels carry signal.
- **Avoid:** presenting this as a refutation of ParT's design. Different scale, different
  pair population.

### 20 · Does it reproduce?
- Both architectural nulls hold on a second seed with the same sign.
- **The log-polar arm — the only thing that looked promising at seed 0 — flips sign.**
  That is what noise looks like, and why we are not pursuing it.
- **Avoid:** seed changes the split too, so these are different test jets. The *conclusion*
  reproduces, not the number.

### 21 · What it buys
- FLOPs counted at dispatch level — hardware-independent, unlike our wall-clock numbers,
  which were unusable on this machine's GPU backend.
- Only the shaded band contains real jets; bold ratios are inside it, grey are extrapolation.
- **Claim:** 1.5–1.9× fewer FLOPs at jet multiplicities. Separately: ParT could not train
  at batch 256 on 16 GB — zero epochs in 88 minutes — while ours trained in 41.
- **Avoid: do not quote the L = 512 ratio.** Connect back to slide 10 — we said then that
  the asymptotic win sits outside JetClass, and this is the measurement confirming it.

### 22 · What the data do and do not establish
- The right column is deliberately as long as the left.
- If pushed on "is it better than ParT": we do not know, and this hardware cannot answer it.

### 23 · Where this leaves us
- 1–3 are what we established; 4 is what we did not.
- **Ask for cluster time.** The ablations say run the cheap configuration first.

---

## If you only have ten minutes

Slides **12 → 14 → 17 → 19 → 23**. That is: the question, the architecture, the core
result, what earns its keep, and the ask. Everything else is support.

---

# Glossary — the five things you will be asked

Reference material. The one-line answers are at the end if you need them fast.

## Ablation — what it is, why we do it

**Remove one component, retrain, measure what happens.** The term comes from experimental
biology: lesion a region and see which function breaks.

ErwinParTv2 inherited machinery from two parents — ParT's pairwise interaction bias, and
Erwin's distance bias, tree rotation and coarsening hierarchy. When the assembled model
works you cannot tell *which parts* made it work. It is entirely possible for half the
components to contribute nothing while performance comes from somewhere you did not expect.

That is not hypothetical. We predicted ParT's interaction bias was the crux of the merge.
It contributes nothing measurable.

**What makes an ablation valid:** change exactly one thing and hold everything else fixed —
same data, split, seed, schedule, epochs, and parameter count where possible. Our ball-size
sweep is 2.21 M parameters at every m, so geometry is the only variable.

> *"We removed each component in turn to find out which ones are actually load-bearing."*

## Paired McNemar — what the p-values mean

A test for comparing **two classifiers on the same test set**. For each of the 10,000 test
jets, record whether each model got it right:

|                    | model B right | model B wrong |
|--------------------|---------------|---------------|
| **model A right**  | both right    | n₁₀ (A wins)  |
| **model A wrong**  | n₀₁ (B wins)  | both wrong    |

Jets both models get right — or both get wrong — say nothing about which is better. They
cancel. **Only the disagreements count.** McNemar asks whether n₀₁ and n₁₀ differ by more
than chance:  χ² = (|n₀₁ − n₁₀| − 1)² / (n₀₁ + n₁₀).

Worked example, removing ParT's U bias:

```
baseline wins   485 jets
ablation wins   493 jets
p = 0.823   ->  a near-perfect coin flip
```

**Why paired matters.** Both models see identical jets, so intrinsic jet difficulty is
shared and cancels out. Treating them as independent samples would inflate the error bars
and destroy sensitivity. With ~845 disagreements out of 10,000 the paired test resolves
differences down to **±0.006 accuracy**.

**Keep this caveat attached:** p = 0.823 does not mean "identical". It means "any difference
is below our detection limit of 0.006". We can exclude a large effect, not a small one.

> *"Same jets, both models — we only count the cases where they disagree."*

## Seeds — what they are, why two

A **seed** initialises the random number generator: weight initialisation, dropout masks,
batch shuffling, and in our setup the **train/val/test split** as well.

One seed is one draw from a random process. A null could be luck; so could a difference.
Two seeds showing the same conclusion is much harder to get by accident.

| variant | seed 0 | seed 1 | verdict |
|---|---|---|---|
| no U bias | +0.0008 (p=0.823) | +0.0047 (p=0.132) | null twice, same sign |
| minimal | −0.0036 (p=0.282) | −0.0048 (p=0.153) | null twice, same sign |
| log-polar tree | +0.0040 (p=0.192) | −0.0008 (p=0.819) | **sign flips** |

**The log-polar row is the whole reason we did this.** At seed 0 it was the best-looking arm
in the sweep — tempting to chase. At seed 1 it goes negative. One seed would have sent us
down a dead end.

**What the p-values in that table are.** Each p is a **within-seed** McNemar test — that seed's
baseline against that seed's ablation, on that seed's own test jets. There is *no* statistical
test between seeds:

```
seed 0:  baseline_0 vs ablation_0  on seed-0 jets  ->  p = 0.823
seed 1:  baseline_1 vs ablation_1  on seed-1 jets  ->  p = 0.132
```

Two independent experiments. "Reproduces" means both answers agree — not that anything was
tested across them. The seed changes the split too, so the two seeds use different jets; the
*numbers* are not comparable across seeds, only the *conclusions*.

**The log-polar verdict does not come from its p-values.** Both are large (0.192, 0.819), so
both seeds independently say "no evidence". What condemns it is the **delta flipping sign** —
+0.0040 then −0.0008. A real effect does not reverse direction when you reshuffle the data.

> *"Each p is a within-seed test — baseline against ablation on that seed's own jets. Both
> seeds say the same thing independently. And for log-polar the effect flips sign between
> seeds, which is what noise looks like."*

Two is the minimum credible number; three to five would be better. Say so if pushed.

> *"Two independent draws. The conclusion has to survive both."*

## The ablations, one by one

| # | Ablation | Removes | Question | Result |
|---|---|---|---|---|
| 1 | ball size m ∈ {8,16,32,64} | nothing — a sweep | how local can attention be? | m=64 vs m=8: **p = 0.295** |
| 2 | no rotation | Erwin's rotated tree | does cross-ball mixing matter? | −0.0026, p = 0.381 |
| 3 | no U bias | ParT's pairwise bias | is the interaction bias earning its keep? | +0.0008, **p = 0.823** |
| 4 | no distance bias | Erwin Eq. 10 | does distance-decay help? | +0.0016, p = 0.580 |
| 5 | readout: fine | coarse levels | does the class token need subjets? | −0.0070, **p = 0.024 ✓** |
| 6 | readout: coarse | fine level | does it need constituents? | −0.0017, p = 0.612 |
| 7 | log-polar tree | — (changes geometry) | does expanding the collinear core help? | noise (sign flip) |
| 8 | minimal | 2 + 3 + 4 together | are they redundant substitutes? | −0.0036, p = 0.282 |

**1 · Ball size.** Not a removal but a sweep, and the cleverest arm of the study. At L = 64,
**m = 64 means one ball holding every constituent — that is dense attention**, inside our own
hierarchy. Same parameters, same data. A better control than ParT would be, because nothing
differs except locality.

**2 · Tree rotation.** Erwin alternates blocks between the tree and a rotated version so
balls straddle different boundaries and information leaks across them, like shifted windows
in a Swin Transformer. Removing it means balls exchange information only through pooling.

**3 · The U bias.** ParT's central contribution: a learned bias inside the softmax computed
from four physics quantities per pair — ln ΔR, ln k_T, ln z, ln m². The mechanism
`pair_features.py` exists to provide. Removing it changed nothing.

**4 · The distance bias.** Erwin's −softplus(σ)·d̂ penalty, making distant constituents
attend less. Tests whether an explicit distance penalty is needed when balls already impose
locality.

**5 & 6 · The readout.** The class token pools the jet into one vector. "Fine" = it sees only
constituents; "coarse" = only the 16 bottleneck subjets. **The only significant result in the
study.** Constituents-only is worse; coarse-only is fine — the **coarse levels carry the
discriminating signal**.

**7 · Log-polar tree.** Partition in radially-warped coordinates that spread the collinear
core where QCD dumps soft radiation. Physically motivated, empirically noise.

**8 · Minimal.** The decisive arm. Components 2, 3 and 4 are each individually null, but they
could be redundant substitutes covering for one another. Removing all three at once tests
that. They are not: still null, at **24% less compute**.

## Truncation — what it is, why it matters

The model uses a **fixed 64 slots per jet**. Jets are variable-length, so:

- **fewer than 64 constituents** -> the rest are **padding**, masked out. 39.5% of all slots
  are padding on average.
- **more than 64** -> the extras are **truncated**, discarded.

Constituents are **sorted by p_T descending first**, so what gets discarded is always the
softest.

| at L = 64 | |
|---|---|
| jets affected | 6.3% |
| particles dropped | 1.4% |
| **jet p_T lost** | **0.115%** |

The gap between 1.4% and 0.115% is the entire justification: p_T-ordering means we drop many
particles carrying almost no momentum.

**But it is class-biased**, and that is the real worry:

```
Hgg    24.0% of jets truncated    <- gluon jets, high multiplicity
Tbqq   14.2%
QCD     6.2%
Hqql    0.4%
```

A 23.6-point spread, concentrated exactly where **multiplicity is itself a discriminant**. Clip
everything at 64 and a 100-particle jet looks like a 64-particle jet — and that difference is
signal.

**What the control showed** (retrained at L = 128, which truncates nothing):

```
jets L=64 keeps whole (93.7%):  0.7766 -> 0.7766   p = 0.971   IDENTICAL
jets L=64 truncates  (6.3%):    0.6190 -> 0.6556   p = 0.013   +3.7 points
all test jets:                  0.7667 -> 0.7690   p = 0.449   null
Hgg AUC (24% truncated):        0.9538 -> 0.9540               +0.0002
```

Truncation has a real cost, strictly confined to the jets it touches. It is **not** why Hgg is
our hardest class — Hgg is hard because gluon jets are diffuse. We took the trade: 3.7 points
on 6% of jets, against 2.1x the FLOPs and 69% padding on *every* jet at L = 128.

> *"We keep the 64 hardest constituents. That loses 0.1% of the jet's momentum, and we ran
> the control to check — on the jets it doesn't truncate, the two models are identical."*

## The one-line versions

- **Ablation** — remove a part, retrain, see if anything breaks.
- **Paired McNemar** — same jets, both models, count only the disagreements.
- **Seeds** — independent random draws; the conclusion must survive more than one.
- **Detection limit** — ±0.006 accuracy; "not significant" means "smaller than that", not zero.
- **Truncation** — we keep the 64 hardest particles; the rest are the softest, and we measured
  what that costs.
