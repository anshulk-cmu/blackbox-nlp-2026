# Reading Script: "Language Models Use Trigonometry to Do Addition"

**Authors:** Subhash Kantamneni and Max Tegmark, MIT
**Venue:** arXiv:2502.00873v1, posted 2 February 2025
**Reader:** Anshul, Cohere Labs paper reading, 8 May 2026
**Audience level:** explained as if to a sharp grade-12 student
**Length target:** ~60 minute talk; this script is your full prep document

---

## How to use this script

This is not a slide deck. This is a line-by-line teaching walkthrough that you read,
internalize, and then talk from. Every figure, equation, table, and appendix gets
covered. The grade-12 framing means: assume your listener knows trigonometry, basic
linear algebra (dot products, matrices), and what a function is. Assume they have
never seen a transformer. Build everything else from scratch.

Where you see "SAY:" the line is meant to be spoken roughly as written. Where you
see "BOARD:" you are writing on the whiteboard or pointing at the figure. Where
you see "PAUSE:" you stop and let the audience catch up.

---

# Part 0: Setting the stage (5 minutes)

## The one-sentence summary

SAY: "This paper claims that when a large language model adds two numbers, it does
not add them the way you and I add numbers. It rotates them on circles. Like clock
hands. And the authors say they can prove this by surgically intervening inside the
model and watching what happens."

## Why this matters

SAY: "We have known for years that large language models can do arithmetic. What
we have not known is how. There are two camps. Camp one says the model just
memorizes a giant lookup table. Camp two says the model learned an actual
algorithm. This paper is camp two's strongest piece of evidence. The authors
say: not only is there an algorithm, we can name it, locate it inside specific
layers of specific models, and disable it on demand."

## Who the authors are

SAY: "Subhash Kantamneni is a graduate student in Max Tegmark's group at MIT.
Tegmark is one of the central figures in the modern mechanistic interpretability
movement. His group produced the original 'grokking' paper that found circular
representations in toy transformers doing modular addition back in 2022. This
paper is the natural sequel: take the toy finding, look for it in real
production-scale models. They find it. That is the headline."

## What three models they study

SAY: "They look at three open-source language models in the 6 to 8 billion
parameter range: GPT-J from EleutherAI, Pythia-6.9B also from EleutherAI, and
Llama 3.1 8B from Meta. Most of the deep analysis is on GPT-J because its
internal architecture is the simplest. They confirm the main findings on the
other two."

## What I will cover today

BOARD: write the talk outline:

1. The setup: what addition looks like inside a transformer
2. The representation claim: numbers live on a helix
3. The algorithm claim: the Clock
4. The localization claim: which layers do what
5. Error analysis and limits
6. My critical take

PAUSE.

---

# Part 1: The abstract, line by line (3 minutes)

The abstract is a dense paragraph. Let me unpack it sentence by sentence so we
all have the same picture.

## Abstract sentence 1

> "Mathematical reasoning is an increasingly important indicator of large
> language model capabilities, yet we lack understanding of how LLMs process
> even simple mathematical tasks."

SAY: "This is the motivation. Models can do math. We do not understand how.
The paper is going to show us how, for one specific task: addition of two-digit
numbers."

## Abstract sentence 2

> "To address this, we reverse engineer how three mid-sized LLMs compute addition."

SAY: "Reverse engineer is the key phrase. They are not training the models.
They are taking pre-trained models and figuring out the algorithm those models
already learned on their own from internet text."

## Abstract sentence 3

> "We first discover that numbers are represented in these LLMs as a generalized
> helix, which is strongly causally implicated for the tasks of addition and
> subtraction, and is also causally relevant for integer division, multiplication,
> and modular arithmetic."

SAY: "Two things in this sentence. First, numbers are stored on a helix. We
will spend a lot of time on what that means. Second, the word 'causally' is
load-bearing. They do not just observe a helix. They surgically modify the
helix and watch the model break. That is what makes this paper strong."

## Abstract sentence 4

> "We then propose that LLMs compute addition by manipulating this generalized
> helix using the 'Clock' algorithm."

SAY: "The Clock algorithm is named for the obvious reason. If a number lives
on a circle, adding numbers is like turning a clock dial. This is not a new
idea. It was proposed by Neel Nanda's group in 2023 for tiny one-layer toy
models. The contribution here is finding it in real models."

## Abstract sentence 5

> "To solve a + b, the helices for a and b are manipulated to produce the a+b
> answer helix which is then read out to model logits."

SAY: "Three steps. Encode a as a helix. Encode b as a helix. Combine them to
get the helix for a+b. Read out the answer. That is the whole story."

## Abstract sentence 6

> "We model influential MLP outputs, attention head outputs, and even individual
> neuron preactivations with these helices and verify our understanding with
> causal interventions."

SAY: "They go all the way down to individual neurons. Roughly 4500 neurons in
GPT-J. They write a mathematical formula for what each neuron does and check
the formula by replacing the real neuron with their formula and seeing if the
model still works."

## Abstract sentence 7

> "By demonstrating that LLMs represent numbers on a helix and manipulate this
> helix to perform addition, we present the first representation-level
> explanation of an LLM's mathematical capability."

SAY: "First is a strong word. They are not the first to find circles in
neural networks. They are claiming to be the first to give a full
representation-level account of arithmetic in a production model. We will
come back to whether that claim is fully earned."

PAUSE.

---

# Part 2: Vocabulary you will hear me use (2 minutes)

I will not lecture on transformers. But six terms will come up over and over.
Here are one-line definitions so nobody is lost. If anything is unfamiliar,
catch me afterwards.

SAY: "Six terms. Token: a chunk of text — a word, part of a word, or a digit
— that the model converts to an ID and then to a vector.

Residual stream: the running vector for each token as it moves up through the
model's layers. Think of it as a 4096-number summary of what the model
currently 'thinks' about that token.

MLP: a small two-step neural network that processes each token's residual
stream. Has 16384 internal units called neurons. Section 5.4 zooms into
individual neurons.

Attention head: a component that lets one token look at earlier tokens and
copy information from them. GPT-J has 16 heads per layer, 28 layers — 448
heads total.

Logit: the model's raw score for a possible next token. Higher means more
preferred. The model's prediction is the token with the highest logit.

Activation patching: take a working model on a 'clean' prompt, and a broken
model on a 'corrupted' prompt, and surgically swap one component's value
from clean into corrupted. If the model now gives the clean answer, that
component carries the information. The number they report is the logit
difference: how much that swap boosts the clean answer versus the corrupted.
This is the workhorse of the paper. I will explain mechanics in detail when
we get to Figure 4."

SAY: "One caveat to keep in mind. They only run patching on prompts the model
gets right. We come back to this in the critique."

PAUSE briefly. Move on.

---

# Part 3: Section 1, the introduction (4 minutes)

## The opening

> "Large language models display surprising and significant aptitude for
> mathematical reasoning."

SAY: "Standard motivation. Models do math. We do not understand how. Skip."

## The problem statement

> "We reverse engineer how GPT-J, Pythia-6.9B, and Llama3.1-8B compute the
> addition problem a + b for a, b ∈ [0, 99]."

SAY: "Two things to flag. First, they restrict to two-digit addition. Numbers
0 through 99. They do this because all three of these models tokenize 0 through
99 as single tokens. So the model sees the number 'thirty-seven' as one symbol,
not as 'three' followed by 'seven'. This dramatically simplifies the analysis.
It also means the paper's findings do not directly apply to multi-digit
arithmetic, which is something we will come back to."

SAY: "Second, the answer a+b ranges from 0 to 198. They check that all answers
are also single tokens in all three models. So they are studying addition in a
clean single-token-input, single-token-output regime."

## The headline finding

> "Remarkably, we find that LLMs use a form of the 'Clock' algorithm to
> compute addition, which was previously proposed by Nanda et al. as a
> mechanistic explanation of how one layer transformers compute modular
> addition."

SAY: "The Clock algorithm. Named because the picture looks like clock hands.
Originally found in 2023 by Neel Nanda's team in tiny one-layer transformers
trained from scratch on modular addition. The big claim of this paper is that
real, large, generally-trained models also use it, even though nobody told them
to."

## Figure 1 — the headline picture

SAY: "Now let me walk you through Figure 1 because it summarizes the entire
paper in a single image."

BOARD: project Figure 1 or describe it.

SAY: "The figure shows the addition problem 3 + 63 in three steps."

SAY: "Step 1, top left. The numbers 3 and 63 are encoded as positions on
multiple circles. There are four circles, with periods T equals 2, 5, 10, and
100. Period means: after how many integers does the circle wrap around. So on
the T=10 circle, the number 3 sits at angle 3 out of 10, and 63 also sits at
angle 3 out of 10, because 63 mod 10 equals 3. They land at the same position
on this circle."

SAY: "Step 2, the middle. The model rotates the circles for 3 and 63 to produce
a circle for the answer 3 + 63 = 66. Imagine taking two clock hands and combining
them into a third clock hand."

SAY: "Step 3, the right. The model reads off the answer from the combined
helix and converts it back into a token."

SAY: "The reason it is called a helix and not just a circle is that the
authors find a linear axis in addition to the circles. As the number grows,
you also move along the linear axis. So the trajectory of integers is a
helix — a circle plus an upward motion. We will see this in detail in Figure 3."

## Why use the Clock at all?

> "Our work is in the spirit of mechanistic interpretability."

SAY: "Mechanistic interpretability — MI for short — is the subfield that tries
to reverse-engineer neural networks the way you would reverse-engineer a
computer program. Take it apart, figure out what each piece does, write down
the algorithm. The honest framing here is: we do this for the addition
algorithm in a real model."

> "Most LLM MI research focuses either on identifying circuits, which are
> the minimal set of model components required for computations, or
> understanding features, which are the representations of concepts in
> LLMs. A true mechanistic explanation requires understanding both."

SAY: "This is the key intellectual claim of the paper. Most MI work splits
into two camps: circuits people who say 'these are the layers and heads that
matter' and features people who say 'this is how the model represents the
concept'. KT are saying: a real explanation needs both. They claim to do both."

PAUSE.

---

# Part 4: Section 2, related work (3 minutes)

I will move through this fast because most of it is field-mapping for experts.

## Circuits research

SAY: "Olah, Olsson, Wang, Hanna — these are the foundational papers in circuits.
The most famous result is that in-context learning, the ability of a language
model to follow patterns in a prompt, is driven by attention heads called
induction heads. The point of citing these is just to say: people have been
finding circuits for various tasks for years."

## Features research

SAY: "Park, Engels, sparse autoencoders. The Linear Representation Hypothesis,
or LRH, says that meaningful concepts in a language model are stored as straight
lines in the high-dimensional residual stream. Move along that line and you
move along the concept. The paper is going to push against the strongest form
of LRH because helices are not lines."

SAY: "Engels et al. 2024 found that some features are not linear. Days of the
week, for example, are stored on a circle. This paper builds directly on that
methodology and extends it to numbers."

## Reverse engineering addition

SAY: "Liu 2022, Nanda 2023, Zhong 2023 — these papers found Clock and Pizza
algorithms in tiny toy models doing modular addition. Stolfo 2023 mapped the
circuit but not the representation. Nikankin 2024 argued the opposite of this
paper — that LLMs are bags of heuristics, not algorithms. Zhou 2024 found
Fourier features in fine-tuned GPT-2 doing addition."

SAY: "So the literature has clues from toy models and clues from fine-tuned
models. KT are connecting the dots and asserting it all holds in production
pre-trained LLMs."

PAUSE.

---

# Part 5: Section 3, the experimental setup (3 minutes)

## The three models

SAY: "GPT-J: 6 billion parameters, simple MLPs. Pythia-6.9B: 6.9 billion,
also simple MLPs. Llama 3.1 8B: 8 billion, gated MLPs. They picked these to
match an earlier paper by Nikankin, so the comparison is apples to apples."

## Single-token tokenization ranges

SAY: "GPT-J tokenizes 0 to 361 as single tokens. Pythia goes up to 557. Llama
goes up to 999. The intersection is 0 to 361, but they restrict their inputs
further to 0 to 99 to keep the answer 0 to 198 also in the single-token range."

## Performance baseline

SAY: "On 10,000 addition problems for a, b in [0, 99], GPT-J gets 80.5 percent
right. Pythia gets 77.2. Llama gets 98 percent. Llama is way better, but they
focus on GPT-J because GPT-J's simple MLPs are easier to interpret one neuron
at a time."

## The prompt

SAY: "The prompts are listed in Table 2 of the appendix. For GPT-J and Pythia
they use 'Output ONLY a number. {a}+{b}=' and for Llama they use 'The
following is a correct addition problem.\n{a}+{b}='."

SAY: "Notice the prompts are different for Llama. The authors do not justify
this. It is a small methodological inconsistency you might want to flag if you
were a reviewer. We will come back to it."

## Figure 11 — heatmaps of accuracy

SAY: "Figure 11 in the appendix is a heatmap. For each pair (a, b) it colors
the cell blue if the model got it right, white if not. You can see all three
models struggle more when both a and b are large. We will come back to this
when we discuss the model's bias toward smaller answers."

PAUSE.

---

# Part 6: Section 4 — LLMs represent numbers as a helix (15 minutes)

This is the longest part of the talk. The representation claim is the foundation
for everything else.

## The basic question

SAY: "If you opened up the model and looked at the vector for the number 17,
what would you see? It is a 4096-dimensional vector. Just a list of 4096 real
numbers. To a human eye it looks like noise. The question of this section is:
is there hidden structure in those 4096 numbers, and if so, what?"

## Section 4.1: Investigating numerical structure

### What they do

SAY: "They run GPT-J on the integers 0 through 360. Just one number per prompt.
They record the residual stream after layer 0. That gives them a matrix of
shape 360 by 4096. 360 numbers, each represented as a 4096-dimensional vector."

### Why layer 0?

SAY: "Why layer 0 and not the embeddings directly? Because earlier work by
Nikankin showed that layer 0 is special — most of the basic numerical
processing happens in the very first layer. The output of layer 0 is when
the model has finished forming its 'representation of this number' and before
it does anything task-specific."

### Why 0 to 360?

SAY: "Why 360 specifically? Because 360 has lots of integer divisors — 2, 3,
4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 40, 45, 60, 72, 90, 120, 180.
That makes any periodic structure easier to detect by Fourier analysis."

### Linear structure: the PCA finding

SAY: "First analysis: principal component analysis, or PCA. Quick refresher
on PCA. You have 360 vectors in 4096-dimensional space. PCA finds the single
direction in 4096-dimensional space along which the data varies the most.
Project all your vectors onto that direction and you get one number per data
point. That is PC1, the first principal component."

SAY: "When they plot PC1 as a function of the integer a, for a from 0 to 99,
they get the bottom panel of Figure 2. It is a roughly straight line. Look at
the dashed red line — R squared of 0.902. So the first principal component is
mostly linear in the value of the number."

SAY: "This makes intuitive sense. Numbers are ordered. 5 is between 4 and 6.
The first big direction of variation in number representations should encode
that ordering. And it does."

### Periodic structure: the Fourier finding

SAY: "Second analysis: Fourier transform. Quick refresher. A Fourier transform
takes a function and decomposes it into a sum of sines and cosines of different
frequencies. If your function is purely periodic with period 10, the Fourier
transform will have a big spike at frequency one-over-ten."

SAY: "They take the residual stream matrix, treat each column — each dimension
of the residual stream — as a function of a, and Fourier transform it.
Then they average the magnitude of the Fourier coefficients across all 4096
dimensions."

SAY: "Top panel of Figure 2. The result is a sparse spectrum with peaks at
frequencies corresponding to periods T = 2, 5, 10, and 100."

SAY: "Pause and think about why those periods make sense. Period 10 is base 10.
The units digit cycles with period 10. Period 100 is the full range of two-digit
numbers — once you hit 100, you wrap. Period 5 has to do with how digits
combine. Period 2 is parity — even versus odd."

SAY: "So the model has discovered, on its own, just from reading text, that
numbers have base-10 structure. Nobody told it. It learned the digit system."

### Side note on the T=2 feature

SAY: "Actually, the T=2 finding is a bit fragile. In Figure 12 of the appendix,
they show that if you Fourier-transform 0 through 361 instead of 0 through 360,
the T=2 peak goes away. The presence or absence of this feature depends on
whether your sample size is even or odd. They keep T=2 in the analysis because
later they find neurons that read with period 2. So it is real but fragile.
This is one of the small weaknesses of the paper — the T=2 evidence is not
overwhelming."

## Section 4.2: Parameterizing the structure as a helix

### The idea

SAY: "So we have linear structure in the data — PC1 is a straight line. And
we have periodic structure — the Fourier transform is sparse with peaks at
specific periods. How do we combine both?"

SAY: "Answer: a helix. A helix is what you get when you walk in a circle while
also moving forward along an axis. Think of a corkscrew. Or a spring. Or the
DNA double helix without the second strand."

### The math

BOARD: write Equation 2:
```
h^l_a = helix(a) = C · B(a)^T

B(a) = [a, cos(2π a / T_1), sin(2π a / T_1),
           ...,
           cos(2π a / T_k), sin(2π a / T_k)]
```

SAY: "Read this carefully. B of a is a row vector. The first entry is just a
itself, the linear component. Then for each period T_i in their list, they
have a cosine entry and a sine entry. With 4 periods, that gives them 1 plus
2 times 4 equals 9 entries in B(a)."

SAY: "C is a matrix that takes those 9 numbers and produces the residual
stream vector — a vector of size 4096. So C has shape 4096 by 9."

SAY: "Why one cosine AND one sine? Because together cos(θ) and sin(θ) trace
out a full circle as θ varies. cos alone gives you a back-and-forth on a
diameter; you need both to get the full circle. So a cos and a sin pair encode
'the position of a on a circle of period T'."

### What 'generalized helix' means

SAY: "If you have just one period — k equals 1 — you get a regular helix:
one circle plus a line. If you have multiple periods sharing the same linear
axis, the authors call it a generalized helix. Geometrically you cannot
visualize this in 3D, but the math is straightforward."

### Counting dimensions, slowly

SAY: "Let me count carefully because dimensional accounting is going to come
back over and over in this paper."

BOARD: write
```
B(a) for k = 4 periods has length 1 + 2k = 9.
   1 entry: the linear component a.
   2 entries per period × 4 periods = 8.
Total: 9 numbers per integer a.

C has shape (model_dim, 2k+1) = (4096, 9).
helix(a) = C · B(a)^T has shape (4096, 1) — a residual vector.
```

SAY: "So the helix family is a 9-dimensional subspace of the 4096-dimensional
residual stream. It is much smaller than the full space. The strong claim of
this paper is: those 9 dimensions are exactly the ones the model uses for
addition. The other 4087 dimensions are doing other things."

### Why cos and sin together — geometric intuition

SAY: "Quick aside on why we need both cos and sin. Picture a unit circle.
A point on the circle is (cos θ, sin θ). As θ goes from 0 to 2π, the point
traces the full circle. If you had only cos θ, you would just have a number
between -1 and 1 — that is the projection of the circular motion onto a
single axis, like watching a Ferris wheel from the side. Two points on
opposite sides of the circle — say θ = π/3 and θ = 5π/3 — have the same
cos value, so cos alone cannot tell them apart. You need sin to break that
symmetry. cos and sin together let you uniquely identify any angle."

SAY: "For the model, putting an integer a on a circle of period T means the
angle is θ = 2π a / T. So integers a and a + T sit at the same position on
this circle. Integers a and a + T/2 sit at opposite poles. The model can
then use this circular position to reason about modular arithmetic naturally
— addition becomes rotation, just like clock arithmetic."

### Why these specific periods?

SAY: "They choose T equals 2, 5, 10, 100 because:
- T equals 5, 10 had clear high-frequency peaks in Fourier (Fig 2)
- T equals 100 has a peak too plus base-10 inductive bias
- T equals 2 is the parity feature, somewhat fragile

They are explicit about being suspicious of low-frequency Fourier components
because those can be artifacts of the windowing in the Fourier transform.
T=100 they keep on the strength of base-10 reasoning."

## Section 4.3: Fitting the helix

### The procedure

SAY: "Now they want to actually fit this helix to the data. They have the
residual stream vectors for a from 0 to 99, on the a token in the prompt
'a + b ='. They want to find the matrix C in equation 2 that best explains
those vectors."

### Step 1: PCA down to 100 dimensions

SAY: "They first project the residual stream from 4096 dimensions down to
100 using PCA. This is for two reasons. One: keeps the regression tractable.
Two: removes a lot of the noise dimensions that have nothing to do with
numerical structure."

### Step 2: linear regression

SAY: "In the 100-dim PCA space, they fit a coefficient matrix C_PCA of shape
100 by (2k+1) such that PCA(h_a) ≈ C_PCA · B(a)^T. This is just least-squares
linear regression."

### Step 3: project back

SAY: "They use the inverse PCA — multiplying by the transpose of the PCA
matrix — to lift C_PCA back to a full 4096 by (2k+1) matrix C in the original
residual stream space."

### Avoiding overfitting

SAY: "They try k = 1, 2, 3, 4 — that is, 1 to 4 Fourier periods. For each k
they pick the best combination of periods from the candidate list. They also
do a train-test split: train on 80 percent of the a values, test on the other
20 percent. Results are essentially identical, so they are not overfitting.
This is in Appendix C.2 Figure 20 left panel."

## Figure 3 — visualizing the helix subspace

SAY: "Figure 3 is the visualization of the fit. Top row, four panels, one
per period. Each panel shows sin(2π a / T_i) on the vertical axis versus
cos(2π a / T_i) on the horizontal axis. So each number a from 0 to 99 lands
somewhere on the unit circle for each period."

SAY: "Look at the T=2 panel. The points are colored by 'a mod 2' — yellow
for even, dark for odd. They cluster in two regions, one for even, one for
odd. Two clusters because period 2 means the circle has two distinguishable
positions."

SAY: "T=5 panel. Five clusters. Numbers 0, 5, 10, 15... cluster together.
1, 6, 11, 16 cluster together. And so on. Five congruence classes mod 5."

SAY: "T=10 panel. Ten clusters arranged around a circle. Beautiful. This
is the digit clock. Numbers ending in 0 cluster, numbers ending in 1 cluster,
... numbers ending in 9 cluster."

SAY: "T=100 panel. Numbers spread around a single circle in order. Because
period 100 covers the full range, no two numbers in [0,99] alias on this
circle."

SAY: "Bottom row, the linear component. Just a straight line in a. Numbers
go from 0 on the left to 95 on the right, ordered."

SAY: "Look at all five panels together. That is the generalized helix. Five
coordinates per number: position on the linear axis, position on the T=2
circle, position on the T=5 circle, position on the T=10 circle, position
on the T=100 circle."

PAUSE.

## Section 4.4: Causal validation of the helix

### Why we need this

SAY: "So far we have a fit. The fit is good. But fits are cheap. The question
is: does the model actually USE this helix structure, or are we just finding a
post-hoc pattern that happens to correlate?"

SAY: "This is the critical question for any interpretability claim. Correlation
versus causation. The way you test causation in a neural network is activation
patching."

### Activation patching, walked through with a real example

SAY: "I told you in Part 2 that activation patching is a swap experiment. Now
that we are about to use it in anger, let me walk through one specific patching
trial in slow motion. This is the recipe."

BOARD: write
```
Step 1: Pick a clean prompt and a corrupted prompt.
        Clean:     "27 + 35 = "    correct answer 62
        Corrupted: "84 + 35 = "    correct answer 119

Step 2: Run the model on the clean prompt. At every layer,
        save h^l_a — the residual stream at the a token (here 27).

Step 3: Run the model on the corrupted prompt. Save residuals too.
        Model produces some output. Note its logit for "62".

Step 4: Pick a layer l (say layer 5). Run the corrupted prompt
        AGAIN, but at layer 5, when the model is processing
        token 0 (the a position), overwrite the residual stream
        with the saved value from the clean run. Let the model
        finish from there.

Step 5: Look at the logit for "62" in this patched run.
        Compute LD = logit_patched("62") - logit_corrupted("62").
```

SAY: "What does LD measure? It measures: by injecting the clean h at this one
location, did I push the model toward the clean answer 62? If LD is large and
positive, the answer is yes — the residual at layer 5 token 0 carries
information that determines whether the model says 62 or 119. If LD is near
zero, the residual at that location is irrelevant for the answer."

SAY: "Now the trick KT use. Instead of patching in the actual clean residual
h^l_a, they patch in the helix FIT to that residual. That is, they replace
the h with C·B(a)^T — a 9-parameter approximation. If LD is still large with
the helix fit, then those 9 parameters are sufficient to carry the meaningful
information. The other 4087 dimensions are noise as far as this task goes."

SAY: "They do this for 100 different clean-corrupted pairs and average. They
also restrict to clean prompts the model gets right, so that the 'flip from
corrupted to clean answer' signal is meaningful."

### Figure 4 — the money plot

SAY: "Figure 4. X axis: layer number, 0 through 27. Y axis: logit difference.
Higher means the patched activation recovered more of the clean answer."

SAY: "Six lines:
- Black solid: patch the actual full residual stream. This is the upper bound.
- Dashed blue: 9-dim PCA. Patch a 9-dim PCA projection of the clean residual.
- Solid blue: helix fit with all 4 periods, 9 parameters.
- Solid green: circle fit with all 4 periods (no linear term), 8 parameters.
- Dotted: 9th degree polynomial.
- Dashed red, orange, light blue: helices with progressively fewer Fourier
  periods.

Look at what happens. The full helix fit and the circle fit are nearly tied.
Both beat the 9-dim PCA baseline at most layers. Both approach the full layer
patch at early layers — meaning the helix captures essentially all the
information the model uses about the number a in those layers."

SAY: "The polynomial baseline does badly. So it is specifically Fourier
structure, not just any nonlinear fit, that matters."

SAY: "As we go deeper into the network, all the simple fits decay because by
later layers the model has already done its computation and the a-token
residual is no longer where the action is. That is fine. The point is at early
layers, when the model is still parsing 'what is this number a', the helix
captures essentially all the relevant structure."

### Quote: 'identified the correct variables of computation'

SAY: "The authors write: 'this suggests that we have identified the correct
variables of computation for addition'. Strong claim. The evidence is solid
for early layers. Take the claim with that scope in mind."

### Sharp jump between layer 0 input and layer 1 input

SAY: "In Figure 4 you see a sharp rise from layer 0 to layer 1. This is
consistent with Nikankin's finding that layer 0 does most of the heavy
lifting in numerical processing. By the time you finish layer 0, the helix
is fully formed."

## Section 4.5: Is the helix the full picture?

### The five additional tasks

SAY: "They check on five other numerical tasks:
1. a - 23 for a in [23, 99]
2. a // 5 — integer division — for a in [0, 99]
3. a * 1.5 for even a in [0, 98]
4. a mod 2 for a in [0, 99]
5. If x - a = 0, what is x = ? for a in [0, 99]"

### Table 1 — performance across tasks

SAY: "Table 1 in the paper. Six tasks across rows, five fit types across
columns: full layer, PCA, helix, circle, polynomial. The number is the
maximum logit difference across layers."

SAY: "For addition: helix 7.21 beats PCA 6.13. For subtraction: helix 7.05
beats PCA 6.16. So helix is causally relevant for the additive operations.
For integer division, multiplication by 1.5, and the equation-solving
problem, helix UNDERPERFORMS PCA. So those tasks need additional structure
beyond the helix."

SAY: "For 'a mod 2', the circle baseline does best. Makes sense — modular
arithmetic should live on a circle without needing the linear axis."

### What this means

SAY: "Their conclusion: the helix is sufficient for addition but not the
full story for arithmetic in general. This is appropriately scoped. They
do not overclaim."

PAUSE for questions.

---

# Part 7: Section 5 — the Clock algorithm (18 minutes)

This is the algorithm claim. Section 4 told us how numbers are stored. Section 5
tells us how addition is computed.

## Section 5.1: Introducing the Clock algorithm

### The four steps

SAY: "The Clock algorithm has four steps. Let me put them on the board."

BOARD: write
```
To compute a + b = , GPT-J:
  Step 1. Embeds a and b as helices on their own tokens.
  Step 2. A sparse set of attention heads, mostly in layers
          9-14, move the a and b helices to the last token.
  Step 3. MLPs 14-18 manipulate these helices to create
          the helix(a + b). A few attention heads help.
  Step 4. MLPs 19-27 and a few attention heads "read"
          from the a + b helix and output to model logits.
```

### What 'last token' means

SAY: "Important detail. The model is processing the prompt 'a + b ='. Each
token has its own residual stream. The final answer comes out at the equals
sign — the last token. So all the information the model needs to give an
answer has to be present in the residual stream at the equals-sign token by
the time the model gets to the top layer. The Clock algorithm describes how
the information gets there."

### Section 4 gave us Step 1

SAY: "We already verified Step 1 in Section 4 — numbers are stored as helices
on their own tokens. The rest of Section 5 is going to verify Steps 2, 3, 4
in detail for GPT-J."

## Figure 5 — last-token hidden states are well-modeled by helix(a+b)

SAY: "Figure 5 is the central evidence for the Clock algorithm."

SAY: "X axis: layer 0 through 27. Y axis: logit difference from activation
patching at the equals-sign token. Six lines:
- Black solid: patch the full layer.
- Dashed dotted: 27-dim PCA — a strong baseline with 27 parameters.
- Dotted: 9-dim PCA.
- Dashed orange: helix(a+b), with 9 parameters.
- Solid orange: helix(a, b, a+b), all three together.
- Solid green: helix(a, b), only the inputs without a+b.

The notation helix(x, y, z) is shorthand for helix(x) + helix(y) + helix(z).
So patching in helix(a+b) means: replace the residual stream with the
helix model would produce for the answer a+b."

SAY: "Watch what happens. At early layers — say layer 5 — helix(a, b) does
better than helix(a+b). Makes sense. Early layers have a and b but have not
yet computed a+b."

SAY: "Around layer 14, the curves cross. helix(a+b) starts winning. Around
layer 17, helix(a+b) — just 9 parameters — beats the 27-dim PCA, which is
an unconstrained baseline with three times the capacity. That is the killer
result."

SAY: "Why is this killer? Because if the model's last-token state at layer
17 were just generic numerical features, a 27-dim PCA would capture them
better than any task-specific 9-parameter fit. The fact that a 9-parameter
helix(a+b) beats 27-dim PCA means the model has SPECIFICALLY constructed the
helix for the answer at this layer."

SAY: "By layer 22 or so, helix(a+b) is matching the full layer patch and
starting to decline only because the model has already moved on to writing
the answer to logits."

PAUSE.

## Why this is strong evidence

SAY: "The argument is: if the model is computing on helix(a+b), then patching
in helix(a+b) should let the model continue and output the right answer.
And it does. With only 9 numbers."

## Section 5.1 caveat: helix(a+b) is weaker on Llama

SAY: "Appendix Figure 23 shows the same experiment on Pythia and Llama.
Pythia replicates GPT-J. Llama is much weaker. helix(a+b) on Llama barely
exceeds 27-dim PCA. The authors hypothesize that Llama's gated MLPs allow
for non-Clock algorithms. This is honest scope-flagging but it also means
the Clock claim for Llama is tentative."

## Figure 6 — MLPs drive the computation

SAY: "Figure 6 separates the contributions of attention layers from MLPs at
the last token. Four lines:
- Solid blue: MLP total effect (TE) — what activation patching tells us about
  the importance of each MLP.
- Dashed blue: MLP direct effect (DE) — what path patching tells us about
  how much each MLP contributes DIRECTLY to the answer logits.
- Solid orange: attention TE.
- Dashed orange: attention DE.

Total numbers: MLP DE total = 8.20, Attention DE total = 1.20. So MLPs are
about seven times more important for the direct contribution to the answer."

### Total effect versus direct effect

SAY: "Quick refresher. Total effect: how much does this component matter
overall — including downstream things it influences? Direct effect: how much
of the contribution to the final answer goes from this component STRAIGHT
to the logits, without going through any other components in between?"

SAY: "If a component has high TE but low DE, it means the component matters
because it FEEDS later components — it is a builder, not a writer."

SAY: "If a component has high DE — close to its TE — it writes directly to
the answer."

### What the figure shows

SAY: "MLP TE peaks around layers 14-18 and again around 19-22. MLP DE is
concentrated in layers 19-27. Attention has small effects throughout, with
peaks earlier in the model."

SAY: "This sets up the next subsection. MLPs do most of the work. Attention
moves stuff around. Let us look at attention first."

## Section 5.2: Attention heads

### How attention works in GPT-J

SAY: "GPT-J has 28 layers. Each layer has 16 attention heads. So there are
28 times 16 equals 448 attention heads in the model. Each head is doing its
own small thing. The total attention output at a layer is the sum of all 16
head outputs."

### How they identify which heads matter

SAY: "They activation-patch every head individually and rank by total effect.
Then they ask: how few heads do I need to patch in to recover most of the
attention contribution? Figure 25 in the appendix shows that 17 heads recover
80 percent. They round up to 20 heads, which gives 83.9 percent."

### The hypothesis: three head types

SAY: "They hypothesize three roles for these 20 heads:
- 'a, b heads' — heads that move the a and b helices from their own tokens
  to the last token, so downstream components can use them.
- 'a + b heads' — heads that take the constructed a+b helix and write it
  directly to logits.
- 'mixed heads' — heads that do both."

### The classification metrics

SAY: "They use two confidence scores. For each head:"

BOARD: write
```
c_{a,b} = (1 - DE/TE) · helix(a,b) / helix(a,b,a+b)
c_{a+b} = (DE/TE) · helix(a+b) / helix(a,b,a+b)
```

SAY: "Read the first one: 1 minus DE-over-TE is the fraction of effect that
goes through downstream components — a 'builder-ness' score. helix(a,b) over
helix(a,b,a+b) is the fraction of the head's output that is captured by
modeling it with just a and b helices, not a+b. So c_{a,b} is high when the
head is both a builder AND its output is well-explained by a and b alone."

SAY: "c_{a+b} is the symmetric story: high DE — writes to logits — and
output well-explained by a+b alone. So c_{a+b} is high for writers of a+b."

### Why this metric design — quick aside

SAY: "I want to spend ten seconds on why these metrics are clever, because
the construction is doing real work. They are simultaneously checking two
independent things and multiplying them together."

SAY: "First factor — direct vs indirect effect — answers 'does this head
write to logits, or does it pipe its output to other components?'. Second
factor — fraction of output explained by a particular helix — answers 'is
this head's output ABOUT a, b, or about a+b?'. A good a-and-b-mover head
should both pipe to other components AND have output that looks like
helix(a,b). A good a+b writer should both write to logits AND have output
that looks like helix(a+b). Multiplying the two factors gives a single
score that requires both conditions. Either factor alone would be misleading."

SAY: "Heads that score low on BOTH metrics are the genuinely mixed ones —
they neither cleanly read from inputs nor cleanly write the output, so they
must be doing the composition step. Mixed heads end up being the most
interesting because they are where some of the helix(a+b) construction
happens."

SAY: "They sort heads by max of these two scores. The lowest-scoring heads
are 'mixed'. They find that classifying just 4 heads as mixed gives 80
percent of the patching effect, meaning their classification is mostly right."

### What they find

SAY: "Final breakdown of 20 heads:
- 11 are a, b heads, mostly in layers 9-14
- 5 are a+b heads, mostly in layers 24-26
- 4 are mixed heads, in layers 15-18

The temporal order matches the algorithm. First a,b heads grab the inputs.
Then mixed heads in the middle do something complicated. Then a+b heads at
the end write to logits."

### Where do the helices get composed?

SAY: "Critically, the a, b heads alone cannot create helix(a+b). They just
copy. So the helix(a+b) must be created either by mixed heads or by MLPs.
The authors check Figure 29 in the appendix and find that mixed heads
receive input from earlier a, b heads — meaning mixed heads might be doing
some of the helix composition. But the bulk of the work happens in MLPs."

## Section 5.3: Looking at MLPs

### How they identify which MLPs matter

SAY: "Same procedure as for heads. Activation patch each MLP, sort by total
effect. They find 11 MLPs are needed for 95 percent of the effect — MLPs 14
through 27, excluding 15, 24, and 25. They use a stricter 95 percent
threshold here because MLPs dominate."

### The two-roles hypothesis

SAY: "They hypothesize MLPs split into:
- builders: read from helix(a, b), output helix(a+b) for downstream use
- readers: read from helix(a+b), output to model logits"

### The two metrics

SAY: "Two metrics:
- helix(a+b) / TE — how well does helix(a+b) explain this MLP's output?
  High value means the MLP is producing helix(a+b)-like vectors.
- DE/TE ratio — how much of this MLP's effect goes directly to logits?
  High value means writer."

## Figure 7 — the builder/reader split

SAY: "Figure 7. X axis: layer 14 through 27. Y axis: fraction of total
effect, 0 to 1. Two lines:
- Blue: helix(a+b) / TE
- Red: DE / TE

Blue starts high and decreases. Red starts low and increases. They cross at
layer 19."

SAY: "Interpretation. MLPs 14 through 18 — blue dominates — produce helix(a+b)
output but most of their effect is INDIRECT, meaning their output is consumed
by later layers. They are BUILDERS."

SAY: "MLPs 19 through 27 — red dominates — have output that is less
helix(a+b)-like and most of their effect is DIRECT, meaning they write to
logits. They are READERS."

SAY: "This is one of the cleanest mechanistic findings in the paper. There
is a literal phase transition at layer 19 between two roles."

## Section 5.4: zooming in on neurons

### Why neurons?

SAY: "We have looked at MLP outputs. We have looked at attention outputs.
But the MLPs are made of neurons, and we want to know what individual neurons
are doing."

### How many neurons in GPT-J?

SAY: "Each MLP layer in GPT-J has 16384 neurons. 27 MLP layers. Total of 27
times 16384 equals about 442,000 neurons. Activation-patching each one is
prohibitively expensive — would take weeks of GPU time."

### Attribution patching

SAY: "Instead they use a technique called attribution patching. Quick
description: instead of running a full forward pass for each patch, you
estimate the patching effect from the gradient of the output with respect
to the activation. One backward pass gives you approximate effects for all
neurons at once. Much cheaper."

### How few neurons matter?

SAY: "Figure 31 in the appendix. They keep the top k neurons by attribution
patching score, mean-ablate the rest. They find that keeping just 1 percent
of neurons — about 4587 of them — recovers 80 percent accuracy on addition.
That is sparse. Most of the model's neurons are doing other tasks. A small
sparse subset does addition."

## Section 5.4.1: Modeling neuron preactivations

### What is a neuron preactivation?

SAY: "Refresher. The MLP is sigma(x W_up) W_down. The vector x times W_up
has 16384 entries. Those 16384 numbers, before sigma is applied, are the
neuron preactivations. Each one is a single scalar number that depends on
the residual stream input x."

SAY: "Specifically the n-th preactivation is the dot product of x with the
n-th column of W_up. This is a linear function of the residual stream."

### What do top-neuron preactivations look like?

SAY: "Figure 8. Four heatmaps. For each one, x axis is b from 0 to 99,
y axis is a from 0 to 99. Color is the neuron's preactivation value."

SAY: "First heatmap, neuron L20N7741. The pattern is striped diagonally with
period 2 in (a+b). The fit is -5.0 cos(2π/2 · (a+b+83)). NRMSE is 0.06 —
very tight fit."

SAY: "Second heatmap. Diagonal stripes at period 5 and period 10. Fit is
sum of two cosines."

SAY: "Third heatmap. Single broad diagonal pattern at period 100."

SAY: "Fourth heatmap. Diagonal but no obvious period — looks like a linear
function of b. Fit is 1.31 b + 1.18 a. NRMSE 0.10."

SAY: "What you should take away. These neurons are NOT random. They have
clean periodic structure as functions of a, b, or a+b. The model has a small
sparse set of neurons each tuned to a specific Fourier feature."

### The functional form

SAY: "They fit each top neuron's preactivation with this formula. Equation 3."

BOARD: write
```
N^l_n(a, b) = Σ_{t ∈ {a, b, a+b}} c_t · t
            + Σ_{T ∈ {2, 5, 10, 100}} Σ_{t ∈ {a, b, a+b}}
                   c_{T,t} · cos(2π/T · (t - d_{T,t}))
```

SAY: "Read this. The neuron preactivation is a sum of two kinds of terms.
Linear terms: c_t times t, where t is one of a, b, or a+b. Plus periodic
terms: cosines with period T applied to a, b, or a+b, with phase shifts d."

SAY: "There are 3 linear coefficients plus 4 periods times 3 variables times
2 parameters per cosine — total 27 free parameters per neuron. They fit
these with gradient descent over 2500 epochs with cosine-annealed learning
rate."

### Why this functional form?

SAY: "Because the neuron preactivation is a linear function of the residual
stream, and the residual stream is approximately helix(a, b, a+b). So a
linear function of helix(a, b, a+b) gives you exactly this kind of expression:
a linear combination of cosines and the linear axis. Equation 3 is the
natural prediction."

### Walking the derivation slowly

SAY: "I want to make this concrete. Recall a neuron preactivation is the dot
product of the residual stream x with one column of the up-projection matrix.
That is a linear scalar function of x."

BOARD: write
```
N(x) = w · x       for some weight vector w (column of W_up)

If x = helix(a, b, a+b)
     = C_a B(a)^T + C_b B(b)^T + C_{a+b} B(a+b)^T

Then N(x) = w · C_a B(a)^T + w · C_b B(b)^T + w · C_{a+b} B(a+b)^T
         = α_a · B(a)^T + α_b · B(b)^T + α_{a+b} · B(a+b)^T

where α_t is a length-9 row vector.
```

SAY: "Each α_t · B(t)^T is a sum of: a linear term in t plus eight cosine and
sine terms. Combining a cos plus a sin into a single phase-shifted cosine —
which is just the standard identity A cos(θ) + B sin(θ) = R cos(θ - φ) where
R is the magnitude and φ is the phase — gives you exactly Equation 3. The c
parameters are the magnitudes, the d parameters are the phase shifts."

SAY: "So Equation 3 is not a guess. It is what you get if you assume the
residual stream is a helix and the neuron is a linear readout from that
residual stream. The fit quality is a test of both assumptions
simultaneously."

SAY: "The 75 percent accuracy of the fit then tells us: about three quarters
of what these neurons do is exactly 'read from the helix'. The remaining 25
percent is either nonlinear effects from the sigmoid in the MLP (which we
ignored above), or structure outside the helix subspace, or both."

### Validation

SAY: "Figure 9. They patch in their fitted preactivations for the top k
neurons and mean-ablate the rest, then measure model accuracy."

SAY: "X axis: number of neurons kept, log scale, 1 to ~5000. Y axis:
accuracy."

SAY: "Two curves:
- Blue: actual neuron preactivations (upper bound)
- Orange: fitted preactivations from Equation 3.

The orange curve is consistently below the blue curve. At the top end the
fit recovers about 75 percent of the accuracy of using actual preactivations.
75 percent is meaningful but not perfect."

SAY: "Important to understand the gap. 25 percent of the model's accuracy
on these top neurons is something the helix-based fit does NOT capture.
Either the helix is not the only structure these neurons read, or the
functional form is missing something. The authors do not investigate this
gap further. It is one of the loose ends."

## Section 5.4.2: Understanding MLP inputs

### From neurons to MLPs

SAY: "Now we use the neuron analysis to draw conclusions about MLP behavior.
For each top neuron, they compute two things:
- DE/TE ratio
- The fraction of the neuron fit's magnitude that comes from a+b terms,
  versus a or b terms.

Then they average across the top neurons in each MLP."

## Figure 10 — neuron trends

SAY: "Figure 10. Scatter plot. X axis: DE/TE ratio of an MLP, averaged across
its top neurons. Y axis: fraction of the fit explained by a+b terms.
Each dot is one MLP, labeled by layer number, colored by layer."

SAY: "MLPs 14, 16, 17, 18 cluster at the bottom-left: low DE/TE — these are
builders — and around 30 percent a+b in their fit, meaning their inputs are
mostly a, b features."

SAY: "MLPs 19 through 27 cluster at the upper-right: high DE/TE — these are
writers — and high a+b fraction, meaning their inputs are mostly a+b."

SAY: "R squared of 0.788 on the linear trend. Strong correlation."

SAY: "This confirms the builder/reader split, but now from a different angle.
We saw it earlier from MLP outputs in Figure 7. Now we see it from MLP
inputs via neurons."

### The conclusion

SAY: "Quote from the paper:
- MLPs 14-18 read from a, b helices to construct a+b helix.
- MLPs 19-27 read from a+b helix to write to logits.

That is the Clock algorithm, fully specified at the MLP level."

## Section 5.5: Limitations the authors flag

### The composition step is not isolated

> "While we provide compelling evidence that key components create helix(a+b)
> from helix(a, b), we do not know the exact mechanism they use to do so."

SAY: "This is the most important honest scope statement in the paper. They
hypothesize that MLPs use trigonometric identities like cos(a+b) = cos(a)cos(b)
- sin(a)sin(b) to construct the answer helix from input helices. But they
cannot demonstrate this mechanism. The 'how' of the composition is left open."

### Models might use multiple algorithms

SAY: "They cite Yip et al. 2024 showing that even in toy modular addition
models, MLPs can implement weird sub-strategies like numerical integration.
Real LLMs probably use ensembles of algorithms, with the Clock as one
component."

### Llama is weaker

SAY: "They note Llama's helix(a+b) fit is weaker. They guess gated MLPs are
the cause. They do not investigate further."

### Tokenization matters

SAY: "Models like Gemma-2 tokenize each digit separately. The Clock algorithm
described here is for the single-token regime only. Multi-digit tokenization
needs additional collation algorithms."

PAUSE for questions on Section 5.

---

# Part 8: Section 6, conclusion (2 minutes)

The conclusion is short. The main new content is the conjecture about why
LLMs use the Clock at all.

> "While LLMs could do addition linearly, we conjecture that LLMs use the
> Clock algorithm to improve accuracy."

SAY: "Their idea: a purely linear representation of numbers would have to be
extremely precise, because nearby answers would be hard to tell apart. A
helical representation gives you redundancy. Multiple Fourier features
encode the answer at different periods. If one is noisy, others can correct."

SAY: "They call this an error-correcting code. They cite work on grid cells
in the brain — the cells in the entorhinal cortex that encode position with
multiple periodicities — as a biological precedent. Grid cells are also a
helical-style code."

SAY: "We will see their evidence for this conjecture when we get to Appendix E.
The evidence is suggestive but weak. I will explain why."

PAUSE.

---

# Part 9: Appendix tour (12 minutes)

Most of the appendix is supplementary detail. I will hit the parts that
matter for the talk and that come up in Q&A.

## Appendix A: model performance

SAY: "Figure 11 shows accuracy heatmaps. Mostly blue everywhere. White cells
are wrong answers. Notice for GPT-J and Pythia the white cells cluster at
high a, high b. For Llama the heatmap is almost completely blue.

Table 2 lists the prompts. Note again the inconsistency — Llama's prompt is
different from the others."

## Appendix B: structure of numbers

### Figure 12 — T=2 sensitivity

SAY: "Figure 12 shows the Fourier transform on h^0_360 versus h^0_361. The
T=2 peak is prominent in the first plot but disappears in the second. The
authors keep T=2 because downstream neurons use it. This is honest but it
does flag that the T=2 finding is fragile."

### Figure 13 — Euclidean distance and cosine similarity

SAY: "Figure 13 shows two heatmaps for h^0_99. For each pair of integers
(a1, a2), color the cell by either Euclidean distance or cosine similarity.
You can see clear off-diagonal stripes with period 10 — meaning numbers
ending in the same digit are more similar to each other than to nearby
numbers."

### Figure 14 — sublinear distance

SAY: "Figure 14: Euclidean distance from a to a+δn. For a in [0,9], it's
linear in δn. For a in [0,99], it is sublinear, meaning the more numbers
you stride across, the slower the distance grows. This sublinearity is
because the helical wrapping starts to bring you back close to your start."

### Figure 15 — PC1 over [0, 360]

SAY: "Figure 15. The first principal component of the embedding for a in
[0, 360] has a sharp discontinuity at a=100 and another at a=300 or so.
This is why the authors restrict to two-digit numbers — three-digit numbers
have a different representation. The single-token tokenization for the
hundreds is processed differently from the tens."

## Appendix C.1: helix properties

### Figure 16 — Fourier feature magnitudes

SAY: "The columns of the matrix C in Equation 2, when their magnitudes
are plotted, show that magnitude roughly increases with period. T=100
features are the largest. This makes sense — the linear axis dominates,
and high-period features are also closer to linear."

### Figure 17 — orthogonality

SAY: "The cosine similarity matrix between columns of C. Mostly diagonal
with a notable off-diagonal entry between the T=100 sin and the linear
component. The T=2 sin component is small enough they ignore it. Otherwise
the helix axes are nearly orthogonal, as a clean helix should be."

### Figure 18 — continuity of the manifold

SAY: "This is the cleanest piece of feature-manifold evidence in the paper.
They fit a T=100 helix to all numbers ending in something OTHER than 3.
Then they project the residual streams for 3, 13, 23, ..., 93 onto this
fitted manifold. Result: each number lands at the position you would expect
if the manifold were a smooth continuous structure. 93 lands between 89
and 95 on the curve."

SAY: "Why this matters. It satisfies the formal definition of a 'nonlinear
feature manifold' from Olah and Jermyn 2024. Numbers really are stored on
a curved 1D manifold, not just on a high-dimensional cloud."

## Appendix C.2: more causal experiments

### Figure 19 — replication on Pythia and Llama

SAY: "Helix and circle fits also outperform PCA on the a token for Pythia
and Llama. So the representation claim holds for all three models, even
though the algorithm claim is weaker for Llama."

### Figure 20 — three checks

SAY: "Three subplots:
- Left: train-test split. Helix and circle still outperform PCA when trained
  on 80% of a values and tested on the other 20%. So they are not overfitting.
- Middle: randomized helix. If you fit a helix to randomized labels, the
  fit is not causally relevant. So the helix functional form is not
  trivially expressive.
- Right: same fits work for the b token. So the representation finding is
  symmetric in a and b."

### Figure 21 — ablation

SAY: "Ablating the helix dimensions from the residual stream — that is,
projecting out C_dagger — destroys performance about as much as ablating
the entire layer. So the helix really is necessary, not just sufficient."

### Figure 22 — task transfer

SAY: "Helix fits on individual tasks 2-5 from Section 4.5. Sometimes
underperforms PCA. Authors honest about this — additional structure is
present in numerical representations beyond what the helix captures."

## Appendix D: more Clock evidence

### Figure 23 — last token fits on other models

SAY: "helix(a+b) fits last-token states for Pythia (decent) and Llama (much
weaker). This is the figure that flags the Llama caveat."

### Figures 24-29: attention heads in detail

SAY: "Figure 24: heatmap of which attention heads have causal effect.
Sparse — most are zero, a few are very high.

Figure 25: 20 heads recover 80 percent of effect.

Figure 26: each head's output is well-modeled by helix(a, b, a+b).

Figure 27: 4 mixed heads suffice to recover 80 percent of the patching
effect.

Figure 28: properties of each head type — a+b heads attend to last token,
a,b heads attend to a and b tokens, mixed heads attend to all three.

Figure 29: the most informative one. They view a, b heads and mixed heads
as 'senders' and trace the path patching. Mixed heads receive input from
both a, b heads and earlier MLPs, suggesting they participate in helix(a+b)
construction."

### Figures 30-35: MLPs and neurons in detail

SAY: "Figure 30: 11 MLPs achieve 95 percent of patching effect.

Figure 31: 1 percent of neurons recovers 80 percent of accuracy. Note that
ablating SOME neurons increases logit difference while decreasing accuracy.
Authors do not investigate this asymmetry. It is a small loose end.

Figure 32: most top neurons live in circuit MLPs.

Figure 33: 700 neurons achieve 80 percent of direct effect.

Figure 34: top neurons' Fourier decomposition with respect to a+b shows
peaks exactly at T = 2, 5, 10, 100 — same as the helix periods. Beautiful
consistency.

Figure 35: NRMSE of helix-inspired neuron fit versus neuron logit difference.
More important neurons are fit better. Less important neurons make up the
25 percent gap from Figure 9."

## Appendix E: why use the Clock at all?

### The conjecture

SAY: "If the model can do addition linearly, why bother with helices? The
authors hypothesize: linear addition is fragile. Errors compound. Helices
provide error correction."

### The experiment

SAY: "They take the first 50 PCA dimensions of layer 0 representations for
a in [0, 99]. Fit a line ell to those representations. R squared of 0.997 —
very tight linear fit."

SAY: "They then perform 'linear addition' by computing ell(a1) + ell(a2),
checking which integer answer ell(c) is closest to that sum, and seeing if
c equals a1 + a2."

### Figure 36

SAY: "X axis: answer threshold alpha — only count problems where a + b is
less than alpha. Y axis: accuracy.

Two lines:
- Blue: GPT-J. Stays around 80% for all alpha.
- Orange: Linear addition with R^2 = 0.997. Drops to under 20% at alpha = 100.

Authors interpret as: GPT-J cannot do addition linearly even with very
precise linear representations. So the Clock is acting as error correction."

### Why this argument is weak

SAY: "I want to flag this for you. The argument is suggestive but not
demonstrated. Three problems:
1. The linear test is unfair. R squared of 0.997 sounds high, but residuals
   compound when you take 100 nearest-neighbor decisions.
2. There is no rank-matched control. They compare a 1-parameter line to
   the full GPT-J. A fairer baseline would be: linear AND helix
   representations of equal capacity, both with matched noise levels.
3. They never demonstrate the 'error correction' mechanism. They infer it
   from the failure of the linear baseline.

So I would treat the error-correction story as an interesting hypothesis,
not a demonstrated mechanism."

## Appendix F: error analysis

### What kind of mistakes does GPT-J make?

SAY: "GPT-J is wrong 19.5 percent of the time on these problems. Of those
errors, 59.8 percent are negative — model predicts a number smaller than
the truth. Looking at numerical errors specifically, 45.7 percent are off
by exactly minus 10. 27.9 percent are off by exactly plus 10. So 73 percent
of errors are off by ten."

### Hypothesis 1: failed carry

SAY: "If the model is failing to carry — for example, 27 + 35 should be
62, but the units 7 + 5 = 12 require a carry, and if the carry fails you
get 52, an error of minus 10 — then we expect that errors of minus 10
correlate with cases where the units of a and b sum to more than 10."

### Figure 38 — chi-squared test

SAY: "Histogram of the units digit sum for cases where the error is minus
10 versus cases where it is not minus 10. Chi-squared test. p = 0.14. Not
significant. The carry hypothesis is rejected."

### Hypothesis 2: logit readout is fragile

SAY: "If the model has correctly built helix(a+b) but the readout layer
confuses neighbors on the modular circle, then the most common errors
should be at the period of the dominant Fourier component in the logit
landscape."

### What is LogitLens? Quick aside

SAY: "Before I walk these figures I should explain LogitLens because we
have not used it yet. LogitLens is a trick for asking 'what would the model
predict if it stopped HERE instead of running through all the layers?'."

BOARD: write
```
Final layer:  logits = h_final · W_unembed

LogitLens at intermediate point x: pretend logits = x · W_unembed.
```

SAY: "You take any intermediate vector — a residual stream, a single MLP
output, even a single neuron's contribution — and you push it through the
unembedding matrix as if it were the final residual. The result is a
'preview' of what that intermediate vector would say if it had to make a
prediction right now. It tells you which tokens that vector points toward
and which it points away from."

SAY: "Why this is useful here. Each neuron's contribution to the final
prediction is its preactivation times its column of W_down. We can take
that contribution vector and LogitLens it across all 199 possible answer
tokens. The result is a curve over [0, 198] showing which answers this
neuron promotes and which it suppresses. If the curve is periodic in the
answer index, that neuron is reading from a Fourier feature in the helix
and writing to logits with the same Fourier feature."

### Figures 39-42 — logit landscape Fourier analysis

SAY: "Figure 39: LogitLens of top direct-effect neurons. The contribution
of each neuron to the logit at each token from 0 to 198 is periodic. The
neuron is boosting and suppressing tokens at fixed periods.

Figure 40: For neurons with top fit period T_i, the LogitLens has a top
Fourier period of T_i too. So neurons read the helix at period T_i and
write to logits with period T_i.

Figure 41: actual model logits over tokens 0 to 198 for several example
problems. Strong period-10 oscillation. Plus a downward trend — model
prefers smaller answers.

Figure 42 top: slope of best-fit linear trend in logits. Almost always
negative. Small-answer bias.

Figure 42 bottom: histogram of dominant Fourier period in logits across
all problems. Period 10 is the modal answer."

### What this means

SAY: "The error is in the readout. helix(a+b) is built correctly. But the
function from helix(a+b) to logits has oscillations with period 10 and a
downward slope. So nearby answers — a+b minus 10 and a+b plus 10 — are also
strongly promoted. The minus 10 bias comes from the downward slope."

SAY: "This is a satisfying mechanistic explanation of an error pattern.
Compositionally consistent: the helix is correct, the readout is fragile."

### Important for your project

SAY: "I want to highlight this finding because it sets up a clean cross-task
contrast with multiplication. In addition, the breakdown is in logit readout.
Carries work, helix is built, model just confuses neighbors. In multi-digit
multiplication — my own area — the breakdown is earlier, in the composition
of correctly encoded atomic concepts into answer digits. Different operations
fail at different stages."

## Appendix G: tooling

SAY: "All experiments were done with the Python library nnsight, on a single
NVIDIA RTX A6000 GPU with 48 GB VRAM, in two days of compute. Reproducible
on a single workstation. This is admirable."

PAUSE.

---

# Part 10: My critical take (5 minutes)

I have set up the paper as fairly as I can. Now let me give you what I
actually think, having read this paper a dozen times.

## What the paper got right

### Multiple methods converging

SAY: "The strongest aspect of the paper is methodological convergence. Four
independent methods — Fourier decomposition, helix fitting, activation
patching, neuron-level fits — all agree on the same story. No single method
is airtight, but the convergence is. This is what good interpretability
work looks like."

### Sufficiency tests via patching-the-fit

SAY: "When they hypothesize a representational form, they patch in the fit
and check whether the model still works. This is the right kind of causal
test. They got this right."

### Honest scope statements

SAY: "They flag that Llama is weaker. They flag that the composition step
is not isolated. They flag that the helix is not sufficient for tasks
beyond addition. These are appropriate hedges."

### Density of appendix material

SAY: "Forty-seven appendix figures. Train-test splits. Random-label controls.
Continuity checks. Replications on other models. The controls are there.
Compare this to weaker MI papers that ship one figure and call it a day."

## What the paper got wrong

### Correct-only filtering

SAY: "All patching experiments use only prompts the model gets correct. This
is a methodological habit in MI but it throws away the failure cases. We
just saw in Appendix F that 19.5 percent of the model's behavior is errors,
and those errors have meaningful structure. Filtering them out limits what
you can claim."

### Pre-specified parametric form

SAY: "They commit to the helix form before checking the data. This biases
the analysis. A more conservative procedure would extract subspaces from
the data first — say with LDA or sparse autoencoders — and then test
whether those data-driven subspaces have helical structure."

### Rank-mismatched baselines

SAY: "The 9-parameter helix beats the 27-dim PCA. But why is 27 the right
number? Why not 9? Or 100? They never explain. The PCA baseline spends
capacity on noise dimensions because PCA is unsupervised. A task-informed
9-parameter fit beating an unsupervised 27-dim fit is not a complete
victory."

### The Clock-as-error-correction argument

SAY: "Appendix E. As I said earlier, the linear baseline at R squared 0.997
is unfair. The argument is suggestive but not demonstrated."

### The T=2 fragility

SAY: "T=2 disappears when you analyze 0 to 361 instead of 0 to 360. They
keep it because downstream neurons use it. But this is mildly circular —
the downstream neurons are identified assuming T=2 matters."

### Composition step is unidentified

SAY: "By the authors' own admission, they cannot say HOW MLPs build helix(a+b)
from helix(a) and helix(b). They guess trigonometric identities. The actual
mechanism is open. This is a real gap in the story."

## Bottom line

SAY: "I rate the representation claim — numbers are stored as helices —
at A. Strong evidence, multiple methods, honest scope.

I rate the algorithm claim — the Clock algorithm is what the model uses —
at B for GPT-J and Pythia, C for Llama. Strong for the simple-MLP models
but the composition step is not nailed down.

The localization claim — which layers, which heads — is at A for GPT-J
and softer for the other models.

Overall: this is the strongest paper on representation-level mechanistic
interpretability of arithmetic in production LLMs as of early 2025. It is
the necessary starting point for anyone working in this area. Read it,
build on it, but do not treat it as the final word."

PAUSE.

---

# Part 11: Reviewer Q&A prep (anticipated questions)

These are questions I expect from a Cohere Labs audience. Memorize the
answers or get close.

## Q: How exactly do they fit the helix?

ANSWER: "PCA the residual stream down to 100 dimensions. Linear regression
in that 100-dim space against the basis B(a) with 2k+1 entries. Inverse PCA
to lift back to 4096 dimensions. Train-test split on a values to avoid
overfitting. The fit minimizes mean-squared error in the 100-dim PCA
projection, not in the full residual."

## Q: Why these specific Fourier periods?

ANSWER: "T = 5 and 10 from the peaks in Figure 2 Fourier decomposition.
T = 100 from base-10 inductive bias plus moderate Fourier magnitude. T = 2
from parity reasoning, but it is the most fragile finding — it depends on
whether you analyze 0 to 360 or 0 to 361. They keep it because downstream
neurons read with period 2."

## Q: What is the difference between activation patching and path patching?

ANSWER: "Activation patching gives total effect. Replace a component's
activation; measure the change in output. The component matters if the
change is large. Path patching isolates direct effect by patching only the
specific edge from one component to the output, blocking all indirect
routes through other components. The DE/TE ratio in Figure 7 separates
'this MLP affects the answer through downstream components' from 'this
MLP writes directly to logits'."

## Q: Does this transfer to multiplication?

ANSWER: "Table 1 shows the helix underperforms PCA on three of five non-
addition tasks, including multiplication by 1.5. The authors flag that
additional structure is needed for those tasks. So the helix is sufficient
for addition but not for arithmetic in general."

## Q: What does 9-parameter helix(a+b) beating 27-dim PCA mean?

ANSWER: "It means at the relevant layers of the model, the model is computing
on 9 specific task-relevant dimensions corresponding to the helix axes. A
27-dim unconstrained PCA captures more variance overall, but PCA spends its
capacity on whatever varies most in the data, which includes noise dimensions
that have nothing to do with the answer. The 9 helix parameters are precisely
the ones the model uses, so they outperform an unconstrained 27-dim baseline.
This is the strongest single piece of evidence for the algorithm claim, but
it is also subtle."

## Q: What about Llama? You said it is weaker.

ANSWER: "On the a token in isolation, Llama's helix fit is causally relevant
just like GPT-J. But on the last token — where the model has constructed
helix(a+b) — Llama's fit is much weaker. The authors hypothesize that gated
MLPs in Llama allow alternative algorithms beyond the Clock. They do not
investigate further. So the Clock claim for Llama is tentative."

## Q: Why didn't they study failure cases?

ANSWER: "They explicitly filter to correct-only prompts when activation
patching, to reduce noise. This is a methodological choice common in MI but
it limits what you can say about model errors. They do study errors in
Appendix F via Fourier decomposition of logits, but not via patching the
internal helix on failure cases. Studying failure cases is one of the
extensions I am pursuing in my own work."

## Q: How do MLPs actually compute helix(a+b) from helix(a) and helix(b)?

ANSWER: "The authors do not know. They hypothesize trigonometric identities
like cos(a+b) = cos(a)cos(b) - sin(a)sin(b). But they cannot isolate the
mechanism in the model. This is the biggest open question of the paper."

## Q: If this is right, how do we use it?

ANSWER: "Three uses. One: error analysis. Knowing that addition errors come
from logit-readout fragility tells us where to intervene to reduce errors.
Two: scaling. If addition uses helices, what does multiplication use? What
does longer arithmetic use? Three: feature manifolds in general. This paper
adds to the case that LLMs use nonlinear feature manifolds, not just linear
directions, for many concepts. That changes how we should think about
interpretability tools like sparse autoencoders, which assume linearity."

## Q: How does this relate to grokking?

ANSWER: "Grokking is when a model trained on modular addition suddenly
generalizes after a long plateau. The 2022 grokking work from Tegmark's
group found that grokked models develop circular representations. The
present paper is asking whether pre-trained LLMs trained on text — which
have not 'grokked' anything in the formal sense — also develop these
representations. They do. So the helix story may be a fundamental property
of how neural networks learn numerical structure, not a quirk of grokking."

## Q: What is your one-sentence assessment?

ANSWER: "The representation finding is solid and important; the algorithm
finding is suggestive and incomplete; this is required reading for anyone
in mechanistic interpretability of arithmetic but should not be cited as
the final word on how LLMs do math."

---

# Part 12: Closing (1 minute)

## What to remember

SAY: "Three things to take home today:
1. Numbers in pre-trained LLMs are stored on a generalized helix with Fourier
   periods 2, 5, 10, 100 plus a linear axis.
2. Addition is computed by manipulating these helices through a Clock
   algorithm, with builder MLPs at layers 14-18 and reader MLPs at 19-27 in
   GPT-J.
3. The methods used here — Fourier screening, parametric helix fitting,
   activation patching, neuron-level fits — are now standard tools you can
   apply to other arithmetic operations and other models.

Thank you. Questions?"

---

# End of script

## Total length and pacing notes

Approximately 60 minutes for the talk. Section budget:

- Part 0 (stage setting): 5 min
- Part 1 (abstract): 3 min
- Part 2 (background): 8 min
- Part 3 (intro): 4 min
- Part 4 (related work): 3 min
- Part 5 (setup): 3 min
- Part 6 (Section 4, helix): 15 min
- Part 7 (Section 5, Clock): 18 min
- Part 8 (conclusion): 2 min
- Part 9 (appendices): 12 min — TRIM if running long
- Part 10 (your take): 5 min
- Part 11 (Q&A prep): not part of talk, for your prep
- Part 12 (closing): 1 min

Total: 79 min if you do everything. You will run over. **Trim Part 9 to 5
minutes** by skipping Figures 13, 14, 25-27, 32 in the appendices. Keep
Figures 18 (continuity), 36 (Appendix E), 38 (chi-squared), 41-42 (logit
periodicity).

Realistic 60-min talk: Parts 0, 1, 2, 3, 5, 6, 7, 9-trimmed, 10, 12.
Skip Part 4 (related work) entirely — audiences hate it and it does not
help your story. Mention 1-2 references inline when needed.

## Presentation tips

1. **Practice the math twice.** Equation 2 and Equation 3 must come out of
   your mouth without hesitation.
2. **Talk from figures.** Do not memorize prose. Know what each figure says
   and rebuild the surrounding sentences live.
3. **Pace yourself on Section 4.** The instinct will be to rush past the
   helix because it feels obvious. Audience needs time.
4. **For Q&A, "I don't know" is fine.** "My best guess is X" is also fine.
   Improvising is not fine.
5. **End on time.** A talk that finishes 5 minutes early is better than one
   that runs 5 minutes over.

## What I'd do differently if I had infinite time

If you had 90 minutes you could:
- Walk Equation 2 derivation more carefully
- Show one full path-patching example in detail
- Cover the Pythia and Llama appendix in depth
- Discuss the Pizza-versus-Clock distinction in toy models

But you have 60 minutes. Cut deep.

Good luck on May 8.