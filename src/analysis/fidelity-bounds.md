---
title: "Swapping fidelity: a tight lower bound at equal source fidelity"
date: "25 October 2026"
header-includes:
  - \usepackage{needspace}
  - \usepackage{etoolbox}
  - \BeforeBeginEnvironment{longtable}{\Needspace{8\baselineskip}}
---

**This sharpens the supplied paper's lower bound for a specific case, as an
application of an established coding-theory result.** After reading the main
text and Appendices A–E, the relevant
comparison is with Lemma 4 and Corollary 4 in Section III, equations (32)–(33).
Their lower bound is $\prod_k F_k$. The remark immediately afterwards explicitly
leaves tightness for $N>2$ to future work. The derivation below resolves the case
$F_1=\cdots=F_N=F\geq 1/2$: it proves the exact minimum and constructs states
attaining it. Numerical tests support the derivation; the proof establishes
tightness beyond the tested sizes.

The assumption is **equal fidelities, not necessarily identical states**.
Different links may have different error types while all having fidelity $F$.
Here $N$ counts elementary links: the chain has $N+1$ nodes. Inputs are independent,
swaps and Pauli corrections are ideal, and there is no extra noise between swaps.
For general two-qubit inputs, all corrected outcomes are accepted. For
Bell-diagonal inputs, conditioning on Bell outcomes gives the same output state,
so the result also covers the paper's postselected swapping in that case.

This is a partial resolution of the paper's open question. The unequal-fidelity
case remains outside the closed-form result below. The later SDP lower bounds
in Section IV and Appendices B–D concern a different problem: two-link
postselection on general states with fixed parameters $p$ and $F$. Our result
does not replace those bounds. The pure partially entangled example later in
these notes is already discussed in Section IV.A of the paper.

**Updated novelty assessment: do not claim this as a new standalone mathematical
bound.** A broader search on 25 October 2026 located equivalent classical
optimal-code results. Kløve's *Codes for Error Detection* (2007), Section 3.2,
Lemma 3.3 and Theorem 3.3/Table 3.1, pp. 133–136, gives the balanced column
multiplicities for optimal binary dimension-two codes. Section 3.7, p. 152,
attributes that material to his 1992 paper. The precise dictionary below
recovers our formula and attaining family. See the
[book's Section 3.2, readable text mirror](https://pubhtml5.com/dtiq/mvwp/basic/101-150),
the [publisher record](https://doi.org/10.1142/6400), and
[Kløve, "Optimal codes for error detection," IEEE Transactions on Information
Theory 38(2), 479–489 (1992)](https://doi.org/10.1109/18.119708).

The bound still improves the interval in [Davies, Avis and Wehner,
arXiv:2509.16689v1](https://arxiv.org/html/2509.16689v1). What remains unestablished
is whether this specific repeater-chain application was stated previously.
A suitable claim is: "Applying classical optimal error-detection results, we
obtain a tight equal-fidelity refinement of Davies et al.'s lower bound."

The convolution machinery is established as well. For example,
[Goodenough, Coopmans and Towsley, Section 2.2 and Appendix C,
arXiv:2404.07146](https://arxiv.org/html/2404.07146) already express Pauli-channel
composition and entanglement swapping as convolution, with Fourier coordinates
that multiply. The optimization step below can be understood as an application
of the classical code result, with an elementary reduction from general
Bell-error mixtures to single-type errors. The earlier limited search in these
notes missed this equivalence; the updated assessment replaces it.

The bounds in `main.tex`, lines 823–843, quantify what can be inferred from
the initial fidelities alone. They do not assume that the initial states are
Werner states. Our target Bell state is $\lvert\Phi^+\rangle$.

Reproduce from the repository root:

```bash
source venv/bin/activate
python -m src.analysis.fidelity_bounds
```

The script uses the visualization app's `EvaluationService` with full real-X
states. Generation and swap hardware succeed deterministically; memory noise,
cutoffs and distillation are disabled. This isolates the instantaneous swapping
claim from waiting-time noise. The expanded study has 244 evaluations, with 512-slot horizons.
Every run has completion coverage above 0.9999999999998. Representative run
bundles include full configurations, backend options, source fingerprints and
arrays; `runs.csv` includes all evaluations. The script checks the Bell-diagonal
results against independent Pauli-label convolution and the closed-form bounds.

**Two links: error cancellation explains both bounds.**

A Bell-diagonal link can be regarded as a perfect Bell pair with an unknown
Pauli error on one endpoint. Swapping composes these errors, ignoring an
irrelevant global phase. Two identical errors cancel; two different errors
produce a third error.

Give both links fidelity $F=0.9$. In Bell order $(\Phi^+,\Psi^+,\Psi^-,\Phi^-)$:

| First link | Second link | Final fidelity |
|---|---|---:|
| (0.9, 0.1, 0, 0) | (0.9, 0, 0.1, 0) | 0.810000 |
| (0.9, 1/30, 1/30, 1/30) | same | 0.813333 |
| (0.9, 0.1, 0, 0) | same | 0.820000 |

The probability that neither link has an error is $0.9^2=0.81$: this is the lower
bound. The probability that both have an error is $0.1^2=0.01$. Distinct error
types never cancel in that event; identical types always cancel; independent
Werner errors cancel with probability 1/3. Consequently the three output
fidelities are $0.81$, $0.81+0.01/3$ and $0.81+0.01$.

**Longer chains: the same initial fidelity can conceal different behavior.**

| Links | Paper lower bound | Exact minimum | Werner | Upper bound |
|---:|---:|---:|---:|---:|
| 2 | 0.810000 | 0.810000 | 0.813333 | 0.820000 |
| 3 | 0.729000 | 0.730000 | 0.738222 | 0.756000 |
| 4 | 0.656100 | 0.666000 | 0.673126 | 0.704800 |
| 8 | 0.430467 | 0.479376 | 0.488714 | 0.583886 |

When every link's error has the same Pauli type, success means an even total
number of errors. The even-parity probability is $[1+(2F-1)^N]/2$, exactly the
upper bound. Thus errors do not simply destroy fidelity independently: later
errors can undo earlier ones. Left-comb, balanced and right-comb four-link trees
all gave 0.7048 in this ideal setting.

The scan covers every allocation of the three error types to rank-two sources
at fixed $F$ (161 count configurations across $N=2,\ldots,8$). The minimum distributes
the types as evenly as possible. For $N=3$, one $X$, one $Y$ and one $Z$ error type
gives $F'=F^3+(1-F)^3=0.730$: all three errors together also cancel, so the
paper's lower bound 0.729 is unattainable here.

There is also a short analytic refinement for **equal source fidelities
$F\geq 1/2$**, within the paper's ideal independent-input setting. At fixed $F$,
the output fidelity is affine in each link's three error probabilities.
Therefore a minimum occurs at a vertex of each error simplex: each link has
just one error type. If their counts are $n_X,n_Y,n_Z$ and $v=2F-1$, Pauli
convolution gives

$$
F'=\frac{1+v^{N-n_X}+v^{N-n_Y}+v^{N-n_Z}}4.
$$

For $0<v<1$ this sum is minimized by counts differing by at most one: balancing
the counts minimizes the sum of the convex functions $v^{-n}$.

**Definitions beside the bound:**

$$
v:=2F-1,\qquad
m:=\left\lfloor\frac{N}{3}\right\rfloor,\qquad
r:=N-3m=N\bmod3\in\{0,1,2\}.
$$

With these definitions, the tight equal-fidelity lower bound is

$$
\begin{gathered}
\boxed{F'\geq L_N(F):=\frac{1+(3-r)v^{N-m}+r v^{N-m-1}}4,}\\[3pt]
m=\lfloor N/3\rfloor,\qquad r=N\bmod3,\qquad v=2F-1.
\end{gathered}
$$

Here $N$ is the number of elementary links and $F$ their common initial
fidelity. The code equivalents are `m = N // 3` and `r = N % 3`.
The integer $m$ is the base
number of links assigned to each of the three error types. The remainder $r$
counts how many types receive one extra link. Thus $3-r$ error types occur on
$m$ links each, and $r$ types occur on $m+1$ links each. Both numbers are fixed
by the chain length.

| Links $N$ | $m=\lfloor N/3\rfloor$ | $r=N\bmod3$ | Optimal counts, up to permutation |
|---:|---:|---:|---|
| 2 | 0 | 2 | $(1,1,0)$ |
| 3 | 1 | 0 | $(1,1,1)$ |
| 4 | 1 | 1 | $(2,1,1)$ |
| 8 | 2 | 2 | $(3,3,2)$ |
| 16 | 5 | 1 | $(6,5,5)$ |

For example, $N=8$ fixes $m=2$ and $r=2$, so
$L_8(F)=[1+v^6+2v^5]/4$. For $N=4$, $m=1$ and $r=1$, so
$L_4(F)=[1+2v^3+v^2]/4$. The coefficients count error-type groups;
each exponent counts links outside that group.

The endpoints $F=1/2$ and $F=1$ follow by continuity, for $N\geq2$. All scanned minima match
this expression. It also extends to general independent input states under
non-postselected swapping using the paper's Bell-twirling equivalence. This
argument does not solve the case of unequal source fidelities.

**The same argument in computer-science language: two bits and XOR.**

Replace the four possible Bell components by four two-bit labels. One valid
encoding is:

| Error | Label | Meaning for the link |
|---|---|---|
| $I$ | `00` | Target Bell component; probability $F$ |
| $X$ | `01` | One possible wrong Bell component |
| $Y$ | `10` | A second wrong Bell component |
| $Z$ | `11` | A third wrong Bell component |

These are labels for Pauli errors modulo global phase. Swapping combines labels
by bitwise XOR. Once the standard measurement-dependent corrections are applied,
the remaining end-to-end error is

$$
E_{\mathrm{out}}=E_1\oplus E_2\oplus\cdots\oplus E_N,
\qquad F'=\Pr(E_{\mathrm{out}}=\texttt{00}).
$$

Thus our quantity is the probability that a two-bit XOR checksum is zero.
It is **not** the probability that every input label was zero. Several nonzero
labels can XOR to zero. For example:

```text
X XOR X         = 01 XOR 01         = 00
X XOR Y         = 01 XOR 10         = 11 = Z
X XOR Y XOR Z   = 01 XOR 10 XOR 11   = 00
```

The paper's lower bound counts just the all-zero input. The improved bound
also accounts for cancellations that cannot be avoided. Associativity and
commutativity explain why changing the swap tree or the order of the links
does not change this calculation in the ideal setting.

An error-type assignment such as `XXYZ` means that the first two links can
each suffer an $X$ error, the third a $Y$ error, and the fourth a $Z$ error.
Each error occurs independently with probability $1-F$. It does **not** mean
that these errors occur on every attempt.

**Three links: avoiding pair cancellations creates a triple cancellation.**

At $F=0.9$, compare three possible assignments:

| Assignment | Error events that return the target | Final fidelity |
|---|---|---:|
| `XXX` | No errors, or any two errors | $0.9^3+3(0.9)(0.1)^2=0.756$ |
| `XXY` | No errors, or the two $X$ errors | $0.9^3+(0.9)(0.1)^2=0.738$ |
| `XYZ` | No errors, or all three errors | $0.9^3+(0.1)^3=0.730$ |

Using three distinct types removes all pair cancellations, but creates the
identity $X\oplus Y\oplus Z=I$. That extra event has probability $0.001$.
So the original lower bound $0.729$ misses a contribution that is unavoidable
even in the worst case. The exact minimum is $0.730$.

This also gives a simple reason for strict improvement when $N\geq3$ and
$1/2\leq F<1$. Among three nonzero error labels, either two repeat, or all
three are distinct. In the first case a pair cancels; in the second a triple
cancels. With all remaining links error-free, either event has positive
probability. The reduction to single-type errors justified below makes this
reasoning cover arbitrary error mixtures as well.

**Four links: count the successful error patterns explicitly.**

Again take $F=0.9$ and write $e=1-F=0.1$.

| Assignment | Contribution to final fidelity | Value |
|---|---|---:|
| `XXXX` | $F^4+6F^2e^2+e^4$ | 0.7048 |
| `XXYY` | $F^4+2F^2e^2+e^4$ | 0.6724 |
| `XXYZ` | $F^4+F^2e^2+2Fe^3$ | **0.6660** |

For `XXYZ`, the successful cases are: no errors ($0.6561$); both $X$ errors
and no others ($0.0081$); or one of the two $X$ errors together with the $Y$
and $Z$ errors ($2\times0.0009$). Their sum is $0.6660$.
This is the worst-case assignment, although each link has exactly the same
initial fidelity as in `XXXX`. Here "balanced" refers to the counts of error
types, $(2,1,1)$, and has nothing to do with balancing the protocol tree.

**Why testing single error types is enough: a weighted average cannot beat its minimum.**

A general Bell-diagonal source with fidelity $F$ has the probability vector

$$
(F,(1-F)u_X,(1-F)u_Y,(1-F)u_Z),
\qquad u_X+u_Y+u_Z=1,\quad u_a\geq0.
$$

It is a mixture of three simpler sources: an $X$-only source with weight $u_X$,
a $Y$-only source with weight $u_Y$, and a $Z$-only source with weight $u_Z$.
Each simpler source still has fidelity $F$.

Expanding the independent product of $N$ such mixtures expresses the output
fidelity as a weighted average over $3^N$ fixed error-type assignments. A
weighted average cannot be smaller than its smallest entry. Therefore at least
one minimizing configuration uses one fixed error type on each link. This
establishes a global reduction of the continuous optimization; it is not an
assumption that the numerical scan happens to make.

For eight links this reduces the problem to $3^8=6561$ assignments. Since order
does not matter and all fidelities are equal, only the three counts matter:
there are $\binom{8+2}{2}=45$ count triples. The balancing argument below removes
even that search.

**Where the formula comes from: three parity bits.**

Let $A,B,C$ be the parity bits of the numbers of actual $X,Y,Z$ errors. The
overall XOR is zero exactly when $A=B=C$. This includes both all-even and
all-odd parity. Because the groups involve disjoint independent links, their
parity bits are independent.

For a group of $n$ links, the even and odd probabilities are

$$
P_{\mathrm{even}}(n)=\frac{1+v^n}{2},\qquad
P_{\mathrm{odd}}(n)=\frac{1-v^n}{2},\qquad v=2F-1.
$$

One way to see the power $v^n$ is that each independent link multiplies the
even-minus-odd bias by $F-(1-F)=v$. Consequently,

$$
\begin{aligned}
F'&=P_{\mathrm{even}}(n_X)P_{\mathrm{even}}(n_Y)P_{\mathrm{even}}(n_Z)\\
&\quad+P_{\mathrm{odd}}(n_X)P_{\mathrm{odd}}(n_Y)P_{\mathrm{odd}}(n_Z)\\
&=\frac{1+v^{n_X+n_Y}+v^{n_X+n_Z}+v^{n_Y+n_Z}}4.
\end{aligned}
$$

Since $n_X+n_Y+n_Z=N$, this is the expression used in the original derivation.

**Why balance is optimal: a discrete load-balancing argument.**

For fixed $N$ and $0<v<1$, minimizing the preceding expression is equivalent
to minimizing

$$
v^{-n_X}+v^{-n_Y}+v^{-n_Z}.
$$

Think of distributing $N$ identical jobs among three bins, where putting $n$
jobs in a bin incurs an exponential cost $b^n$, with $b=1/v>1$. If one bin has
$a$ jobs and another has $c$, with $a\geq c+2$, moving one job from the larger
bin to the smaller reduces the cost:

$$
(b^a+b^c)-(b^{a-1}+b^{c+1})
=(b-1)(b^{a-1}-b^c)>0.
$$

Repeat until no two counts differ by more than one. This is a proof of the
optimal counts, rather than a heuristic. For $N=8$, they are $(3,3,2)$, giving

$$
L_8(0.9)=\frac{1+2(0.8)^5+(0.8)^6}{4}=0.479376.
$$

To attain the bound physically, choose three kinds of Bell-diagonal source,

$$
\rho_a=F\lvert\Phi^+\rangle\langle\Phi^+\rvert
 +(1-F)\lvert\Phi_a\rangle\langle\Phi_a\rvert,
\qquad \lvert\Phi_a\rangle=(I\otimes a)\lvert\Phi^+\rangle,
$$

with $a\in\{X,Y,Z\}$, and assign them to links in those balanced counts.
These are valid states, independent across links, that achieve $L_N(F)$
exactly. Together with the global reduction and minimization, this proves
both parts of tightness: every allowed input is above the bound, and an
allowed input reaches it. The paper's Theorem 1 then extends the result to
general independent inputs under non-postselected swapping.

**More numerical examples, including cleaner and noisier sources.**

All entries below were also checked with the shared evaluator. "Minimum"
allows different error types on different links; "Upper" is attained with
one common error type.

| $F$ | Links | Paper lower bound | Exact minimum | Werner | Upper |
|---:|---:|---:|---:|---:|---:|
| 0.98 | 4 | 0.922368 | 0.922768 | 0.923143 | 0.924673 |
| 0.98 | 8 | 0.850763 | 0.853376 | 0.854163 | 0.860695 |
| 0.90 | 4 | 0.656100 | 0.666000 | 0.673126 | 0.704800 |
| 0.90 | 8 | 0.430467 | 0.479376 | 0.488714 | 0.583886 |
| 0.75 | 4 | 0.316406 | 0.375000 | 0.398148 | 0.531250 |
| 0.75 | 8 | 0.100113 | 0.269531 | 0.279264 | 0.501953 |
| 0.60 | 8 | 0.016796 | 0.250176 | 0.251687 | 0.500001 |
| 0.50 | 4 | 0.062500 | 0.250000 | 0.259259 | 0.500000 |

For very clean links and short chains, extra cancellations are rare, so the
improvement can be small. For noisier or longer chains, counting only the
all-clean event becomes excessively pessimistic. At fixed $1/2\leq F<1$,
the exact minimum approaches $1/4$, whereas $F^N$ approaches zero. The two-bit
error label becomes close to uniformly random in the minimizing construction.
A fidelity near $1/4$ is not a useful-entanglement or positive-key guarantee;
this is a statement about the tight fidelity envelope.

**Equal fidelity versus identical sources changes the answer.**

If the same density matrix must be used on every link, the `XXYZ` construction
is unavailable. For identical sources with $F\geq1/2$, the exact minimum is
instead attained by a Werner state:

$$
W_N(F)=\frac{1+3\bigl((4F-1)/3\bigr)^N}{4}.
$$

For example, with four links and $F=0.9$, the minimum is $0.666000$ when
states may differ and $0.673126$ when all states must be identical.
For eight links the corresponding minima are $0.479376$ and $0.488714$.

To verify this distinction analytically, let the common source's three error
probabilities be $p_X,p_Y,p_Z$, summing to $1-F$. Its three nontrivial parity
biases are $t_a=2F-1+2p_a\geq0$, with mean $(4F-1)/3$. Repeating that same
source gives

$$
F'=\frac{1+t_X^N+t_Y^N+t_Z^N}{4}.
$$

Convexity of $t^N$ for $t\geq0$ says the sum is smallest when the three biases
are equal, which means $p_X=p_Y=p_Z=(1-F)/3$: the Werner source. Independent
choices of error type per link and identical per-link distributions are
different constraints. Calling our first result merely a "homogeneous bound"
would hide this distinction; "equal-fidelity bound" is more precise.

**What this application of the classical result would contribute to the paper.**

It supplies the exact lower endpoint of the fidelity interval in Lemma 4 and
Corollary 4 when $F_k=F\geq1/2$, together with an attaining family. For this
case the paper's interval can be sharpened to

$$
L_N(F)\leq F'\leq U_N(F),\qquad
U_N(F)=\frac{1+(2F-1)^N}{2}.
$$

Since $W_N(F)$ is fixed at those source fidelities, this also gives the exact
maximum absolute fidelity error of the Werner approximation over the same
allowed inputs:

$$
\max_{\rho_1,\ldots,\rho_N}\lvert F'-W_N(F)\rvert
=\max\{W_N(F)-L_N(F),\ U_N(F)-W_N(F)\}.
$$

For $F=0.9$ and $N=8$, the Werner prediction can overestimate fidelity by
$0.009338$, or underestimate it by $0.095172$. Both extremes are attainable.
This is a finite-size statement for the equal-fidelity case, complementing
the paper's high-fidelity asymptotic accuracy result. It does not directly
bound SKR, which also depends on individual error rates and waiting times.
Applying the result to our noisy-memory protocols would require checking the
states entering each swap, rather than substituting the fresh-source fidelity
throughout the chain.

**Novelty audit: the exact coding-theory dictionary.**

The connection is stronger than a shared proof technique. Here is a direct
translation of the optimization problem. This translation is our mathematical
comparison with the cited prior art.

After the single-error-type reduction, assign each link a nonzero two-bit
label $h_k$. Put these labels into the columns of a binary $2\times N$ matrix
$H$. Let $B_k$ indicate whether link $k$ actually has its allowed error, with
independent probability $p=1-F$. The final error label is the syndrome

$$
S=HB\pmod2,\qquad F'=\Pr(S=0).
$$

Consider the classical binary linear code $C=\ker H$. When $H$ has rank two,
this code has length $N$, dimension $N-2$, and two independent parity checks.
An error detector misses exactly the nonzero error patterns belonging to $C$.
Thus, with $P_{\mathrm{ue}}$ denoting its probability of an undetected error,

$$
\boxed{F'=F^N+P_{\mathrm{ue}}(C,1-F).}
$$

The first term is the probability of no error. It is fixed when $N$ and $F$
are fixed. Consequently, minimizing our final fidelity is exactly minimizing
this classical undetected-error probability. The same cancellation patterns
are successful Bell transmission in our calculation and undetected errors in
the classical detector; the terminology changes, but the optimization does not.

For a second, convenient description, let $D=C^\perp$ be the row space of $H$.
It is a dimension-two code with four codewords. Define its weight enumerator

$$
A_D(z)=\sum_{d\in D}z^{\operatorname{wt}(d)}.
$$

The three nonzero codewords have weights $N-n_X,N-n_Y,N-n_Z$. Averaging the
four binary characters of the syndrome yields

$$
\boxed{F'=\frac{A_D(2F-1)}4.}
$$

This is also the relevant specialization of the MacWilliams transform. It
explains why a classification of optimal dimension-two codes answers our
question. More explicitly, introduce an auxiliary classical bit-error probability

$$
\theta=\frac{2F-1}{2F}=\frac{v}{1+v}.
$$

For $1/2<F<1$, it lies in $(0,1/2)$ and satisfies

$$
\begin{aligned}
P_{\mathrm{ue}}(D,\theta)
  &=(1-\theta)^N\left[A_D\!\left(\frac{\theta}{1-\theta}\right)-1\right],\\
F'&=\frac{1+(2F)^N P_{\mathrm{ue}}(D,\theta)}4.
\end{aligned}
$$

For fixed $N,F$, the prefactor is positive and constant. The known optimal
dimension-two construction therefore minimizes precisely the quantity we need.
Writing its weight enumerator in our notation gives

$$
A_D(z)=1+(3-r)z^{N-m}+r z^{N-m-1},
\qquad m=\lfloor N/3\rfloor,\quad r=N\bmod3.
$$

Substituting $z=2F-1$ and dividing by four recovers $L_N(F)$ exactly,
including all three residue classes of $N$ and the attaining construction.
The rank-one case, where all links have the same error type, gives the upper
bound instead; it cannot lower the minimum. The endpoint fidelities follow
by continuity.

For a concrete example, use the assignment `XXYZ`:

$$
H=\begin{pmatrix}0&0&1&1\\1&1&0&1\end{pmatrix}.
$$

The four vectors in its kernel are `0000`, `1100`, `1011`, and `0111`.
The first is error-free; the others give one weight-two and two weight-three
undetected patterns. At $p=0.1$,

$$
P_{\mathrm{ue}}(C,0.1)=0.0081+2(0.0009)=0.0099,
\qquad F'=0.9^4+0.0099=0.6660.
$$

The dual enumerator is $A_D(z)=1+z^2+2z^3$, so equivalently
$F'=A_D(0.8)/4=0.6660$. This is the same numerical example as before, now
expressed as a classical code calculation.

The source comparison was checked against the author's 2007 book: Section 3.2
for the construction and Section 3.7 for its attribution. The original 1992
article's full text was not accessible in this session. Related quantum work
on composition bounds was also checked: [Carignan-Dugas, Wallman and Emerson,
arXiv:1610.05296](https://arxiv.org/abs/1610.05296), especially Theorem 5,
addresses accumulation of Pauli-channel infidelity. The decisive exact match
for our optimization is the classical code construction.

Independent enumeration of all $2^N$ binary error patterns and the dual
codewords was compared with the 161 saved backend cases for $N=2,\ldots,8$.
All agree; the maximum absolute difference is $2.23\times10^{-16}$.
There are 140 rank-two and 21 rank-one cases. The comparison is saved under
`results/analysis/`, in directory `fidelity-code-equivalence-20261025T153941Z`,
as `code-equivalence.csv` and `summary.json`.

The supported contribution is therefore an explicit application and
interpretation of classical optimal-code results that sharpens the interval
in Davies et al. The broader search rules out presenting the formula and
balanced construction as new standalone mathematics. It does not establish
priority for this particular quantum-network application.

**Postselection: why the corollary specifies non-postselected swapping.**

Prepare two identical pure sources
$\sqrt{0.75}\lvert00\rangle+\sqrt{0.25}\lvert11\rangle$.
Each has $\Phi^+$ fidelity $0.933013$. Its Bell-twirled version has probabilities
$(0.933013,0,0,0.066987)$, with exactly the same source fidelity.

| Accepted outcomes | PE fidelity | Bell-twirled fidelity |
|---|---:|---:|
| All corrected | 0.875000 | 0.875000 |
| $\Phi$ pair | 0.800000 | 0.875000 |
| $\Psi$ pair | 1.000000 | 0.875000 |

The upper bound computed from these input fidelities is 0.875. The original
PE sources attain it when accepting all corrected outcomes. Postselection on
$\Psi$ instead gives a perfect Bell pair and exceeds that number. This does not
contradict the corollary: its extension to general, non-Bell-diagonal inputs
requires non-postselected swapping. Bell twirling removes information that
the conditional branches can exploit.

For the PE sources, $\Phi$ acceptance has probability $0.625$ and $\Psi$ acceptance
$0.375$. The average fidelity is $0.625\times0.8+0.375\times1=0.875$. The fidelity gain
comes with retries: $\Psi$ has mean waiting time 2.666667 slots, versus 1 slot
for all corrected outcomes. Their respective SKRs are 0.375000 and 0.456436
bits/slot: higher conditional fidelity alone does not establish a rate gain.

**A domain caveat in the displayed statement.**

The quoted lemma does not explicitly state $F_k\geq1/2$. That is a sufficient
restriction for its upper bound, and all of the main examples above obey it.
Without a domain restriction, the upper bound is false: three Bell-diagonal
sources (0.1,0.9,0,0), (0.1,0,0.9,0), (0.1,0,0,0.9) give final fidelity
0.730000 in the evaluator, whereas the displayed upper expression is
$[1+(-0.8)^3]/2=0.244000$. The no-error branch contributes 0.001; all three
errors together cancel and contribute 0.729. This concerns fidelities to the
fixed target Bell state, rather than fidelities after individually choosing
a different target Bell state for each source. The paper's inductive proof
uses monotonicity of the two-link upper expression in the intermediate
fidelity, which requires the next source's $2F_k-1$ to be nonnegative.

**Reproducibility and checks.**

The expanded run is saved in
`results/analysis/fidelity-bounds-20261025T145051Z/`.
It contains `runs.csv`, `scaling.csv`, `additional-examples.csv`, the plot, and
14 representative importable run bundles. The original $F=0.9$ scan is retained.
Additional evaluations cover $F\in\{0.5,0.6,0.75,0.98,1\}$ and
$N\in\{2,3,4,8\}$, plus the six explicit three- and four-link error patterns.
Independent XOR convolution agrees with the evaluator. A further 1,000 seeded
random Bell-error mixtures satisfy the derived bounds under direct convolution;
these checks are supplementary evidence, not the proof of optimality.

To regenerate the PDF with typeset mathematics from the repository root:

```bash
pandoc src/analysis/fidelity-bounds.md \
  --from=markdown+tex_math_dollars \
  --pdf-engine=xelatex \
  -V geometry:margin=22mm -V fontsize=10pt \
  -V colorlinks=true -V urlcolor=blue \
  -o src/analysis/fidelity-bounds.pdf
```
