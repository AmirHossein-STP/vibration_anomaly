# 10 — Domain adaptation: the concepts, and what the experiment taught

Companion to `DOMAIN_ADAPTATION_PLAN.md`. That file is the chronological
record — what was run, what broke, what the numbers were. This one explains
the *ideas*, including the ones that turned out not to work, because the
failures here are more instructive than the successes.

## 1. What "zero-shot" actually means here

Your real machine will only ever hand you healthy data. Nobody is going to
damage a turbo pump bearing on purpose to build a training set. So the
question is: can a model learn what faults look like *somewhere else*, and
carry that knowledge to a machine it has never seen, using only that
machine's healthy baseline to orient itself?

That is the whole premise, and it is why the evaluation is built the way it
is. `TARGET_ADAPT` (the target condition's healthy runs) is the ONLY target
data training ever touches. Every real-damage bearing under that condition
is sealed off in `TARGET_TEST` and opened once, at the end. If that
discipline slips anywhere, the result means nothing — which is why the
split logic lives in one place (`data/dataset.py`) with the roles spelled
out at the top.

## 2. Domain shift, and why it is not just "different numbers"

CWRU and Paderborn recorded different bearings, on different rigs, with
different sensors, at different speeds. A model trained on one and shown
the other faces a *distribution shift*: the mapping from vibration to fault
is broadly the same physics, but the inputs live in visibly different
regions of feature space.

The specific shift that matters most for bearings is **speed**. Bearing
fault frequencies (BPFO, BPFI, BSF — see `notes/05_glossary.md`) are all
proportional to shaft rate. Drop from 1500rpm to 900rpm and every
characteristic frequency moves to 60% of where it was. On a linear-frequency
spectrogram that is a *multiplicative rescaling of the frequency axis* — not
a translation. A CNN has some translation tolerance built in; it has
essentially none for scaling. This is the gap the experiment never closed,
and the reason §6 below is the recommended next step.

## 3. DANN: the idea

A Domain-Adversarial Neural Network has three parts:

- **Feature extractor** `G_f` — the shared CNN, the thing you actually want.
- **Fault classifier** `G_y` — reads features, predicts healthy/inner/outer.
- **Domain discriminator** `G_d` — reads the same features and tries to
  guess which *dataset* they came from.

The trick is the **Gradient Reversal Layer** between `G_f` and `G_d`. Going
forward it does nothing. Going backward it multiplies the gradient by
`-lambda`. So while `G_d` is busy getting better at telling domains apart,
`G_f` receives the *negated* version of that signal and is pushed to make
domains indistinguishable. One optimizer, one `backward()`, no alternating
min/max loop. It is genuinely elegant (`models/grl.py`).

The hoped-for outcome: features that still support fault classification but
carry no domain information, so a classifier trained on source labels works
unchanged on the target.

## 4. Why it did not work here — the three failure modes, in order

**(a) The classifier wins the race.** Classification on source data is
*easy* — 81% accuracy within one epoch. Meanwhile `lambda` starts near zero
by design (ramping it instantly would have the feature extractor fighting a
randomly-initialised discriminator). By the time adversarial pressure
arrives, the feature extractor has already committed to source-specific
shortcuts, and the classification gradient is more than strong enough to
keep it there. Fixing this — a faster ramp, a heavier domain weight — is
what kernel v2 tried.

**(b) The discriminator loses the race.** Raise the pressure and the
opposite failure appears: the discriminator gets scrambled by the reversal
faster than it can learn anything, so it emits *meaningless* gradients. The
tell is subtle and worth internalising: **a domain loss sitting at chance is
ambiguous.** It means either "features are aligned, the discriminator
genuinely cannot tell" (success) or "the discriminator learned nothing and
is being fooled" (failure). The training log cannot distinguish them.

This is why `analysis.domain_separability_probe` exists. It fits a *fresh*
classifier on the frozen features afterwards, with no gradient reversal
fighting it. That probe cannot be fooled the same way. When it reported
0.969 while in-training domain accuracy sat at chance, the diagnosis was
unambiguous: not aligned, discriminator simply losing. Build the instrument
that discriminates between your hypotheses — otherwise you tune blind, which
is what v1 did.

**(c) Marginal alignment is not conditional alignment.** Even a *perfectly*
domain-invariant feature distribution `P(f(x))` does not guarantee the
per-class structure `P(f(x)|y)` lines up. The healthy clusters can sit on
top of each other while the fault clusters land nowhere near each other.
The healthy-anchor loss was a cheap attempt at pinning one class
conditionally; it was far too weak a constraint for a gap this size.

## 5. The result that actually matters, and why it was nearly missed

Reported as a 3-class problem, every version looks like a failure — below
the accuracy of blindly guessing the majority class. Collapse the same
predictions to **healthy vs. faulty** and a usable detector appears:
95.4% precision, 76.1% recall, and 91.8% detection once aggregated per
recording.

Both framings are honest; they answer different questions. Typing the fault
requires distinguishing inner from outer race on an unseen machine, which
never transferred. *Noticing* the fault requires only separating healthy
from not-healthy, which partly did. The deployment question is the second
one. The lesson worth keeping: **check whether your metric is asking the
question your user has.**

Two related traps this experiment walked into:
- **Accuracy on a skewed test set.** `TARGET_TEST` is ~90% faulty, so
  "always say faulty" scores 0.90. `binary_detection_metrics` therefore
  reports `false_alarm_rate` (measured on healthy windows only, immune to
  the skew) and carries the null model's score alongside the headline.
- **Averages hiding structure.** A 16.7% false-alarm rate sounds like a
  noisy detector. It is not: *all* of it came from one of six healthy
  bearing units, which was misread in 100% of its windows while the other
  five were clean. That is a per-unit offset, and it is correctable by
  calibration — a completely different engineering response than "the model
  is noisy". Always disaggregate before believing an average.

## 6. What to try next, and why

**Order-domain features.** Instead of sampling against time, resample the
signal against *shaft angle* — so many samples per revolution rather than
per second. Frequency then becomes "orders of shaft rotation", and a fault
at BPFO lands in the same bin at 900rpm as at 1500rpm. This directly
cancels the multiplicative rescaling from §2, which is the one gap never
addressed across five runs. The data supports it: PU ships a `speed`
channel in every `.mat`, and CWRU encodes rpm in the filename.

It is also the honest conclusion of §4 — three rounds of adversarial tuning
could not fix a mismatch that is fundamentally about the frequency axis
being *stretched*. Better features beat better adversarial training here,
which is consistent with the other big finding: source diversity (a data
property) outperformed every optimisation knob tried.
