
- when the noise model changes the optimal protocol? what physical mechanism explains that change?
- extracting design rules

In the following, I may abbreviate Amplitude Damping (AD)

I may refer to work on twirling accuracy and amplitude-damping advantages. 
The sharper contribution would be *optimization loss across swapping, distillation, and cutoffs*. Also include a control regime where Bell-diagonal reduction is known to preserve the relevant observables for non-postselected swapping. [Davies et al.](https://arxiv.org/abs/2509.16689), [Mondal et al.](https://arxiv.org/abs/2509.18064)


1. **How much performance do we lose by optimizing with the wrong noise model?**

    1. Set a scenario

       **Question:** What noise model should we use? It seems unsurprising that a Werner approximation is inaccurate under AD; how do Davies et al. and Mondal et al. pose their comparisons?

       **Answer:** Start from one physical memory model—amplitude damping plus independent dephasing—and derive each approximation from that same model. The research question is whether discarding particular state information changes the best protocol, and by how much. Neither the presence of AD nor an error in fidelity prediction guarantees a meaningful optimization loss. Davies et al. actually supply an important example in which discarding information is *exact for the observables of interest*.

       A necessary correction to the terminology below: **Bell-diagonal state twirling and Pauli channel twirling are different approximations.** They should be separate baselines. This distinction explains why the main conclusions of the two papers are compatible.

       **What Davies et al. fix and compare.** Their starting point is a product of specified two-qubit input states, \(\rho_{\mathrm{in}}=\bigotimes_{k=1}^{N}\rho_k\), where \(N\) counts elementary pairs. They do not select a particular time-dependent memory-noise channel for their main theorems. Instead, they compare swapping these states with swapping their state-twirled versions:

       \[
       \mathcal B(\rho)=\sum_{b\in\mathrm{Bell}}\langle b|\rho|b\rangle|b\rangle\langle b|,
       \qquad
       \mathcal W(\rho)=F\Phi^++\frac{1-F}{3}(I-\Phi^+),
       \]

       where \(\Phi^+=|\phi^+\rangle\langle\phi^+|\) and \(F=\mathrm{Tr}(\Phi^+\rho)\). Thus \(\mathcal B\) preserves all four Bell populations, whereas \(\mathcal W\) preserves only the target fidelity and equalizes the three error populations. These are prescribed transformations of the same input, not independently fitted noise strengths. See their [definitions and setup](</Users/lorenzolacorte/Documents/SoA/on-the-accuracy-of-twirled-approximations-in-repeater-chains/main.tex:416>).

       For **non-postselected swapping**, Theorem 1 establishes

       \[
       \mathcal B\!\left(\Lambda_{\mathcal P}(\rho_{\mathrm{in}})\right)
       =\Lambda_{\mathcal P}\!\left(\bigotimes_k\mathcal B(\rho_k)\right),
       \]

       for their class of swap-and-correct protocols \(\mathcal P\). All Bell outcomes are retained, corrected, and averaged with their actual probabilities. The equality concerns the Bell-diagonal part of the output; the full output states need not coincide. Since the BB84 error rates depend only on \(\langle XX\rangle\) and \(\langle ZZ\rangle\), the pooled BB84 secret fraction is also exactly preserved. This holds even if AD produced the input states, provided that subsequent processing is the swap-and-correct map covered by the theorem. See [Theorem 1](</Users/lorenzolacorte/Documents/SoA/on-the-accuracy-of-twirled-approximations-in-repeater-chains/main.tex:790>) and the [secret-fraction argument](</Users/lorenzolacorte/Documents/SoA/on-the-accuracy-of-twirled-approximations-in-repeater-chains/main.tex:2865>).

       **The scope matters:** the theorem does not include arbitrary extra memory channels between swaps, distillation, or state-dependent decisions. Its proof rearranges Pauli corrections and Bell projections; it does not commute those operations through AD. In particular, deferring a physical Pauli correction through a damping interval is generally not equivalent to applying it immediately. Our initial comparison should hold the correction convention fixed. See the [technical protocol definition and proof](</Users/lorenzolacorte/Documents/SoA/on-the-accuracy-of-twirled-approximations-in-repeater-chains/main.tex:1934>).

       Their Werner result is weaker: with \(1-F_k\leq\epsilon\), the non-postselected output-fidelity error is bounded asymptotically by

       \[
       |F'_{\mathrm{full}}-F'_{\mathrm W}|
       \leq {N\choose2}\epsilon^2+\mathcal O(N^3\epsilon^3).
       \]

       Thus \(N\epsilon\ll1\) identifies a high-fidelity regime where the Werner approximation is accurate **for output fidelity**. This does not establish the same bound for key rate or guarantee the same optimizer: the key fraction depends on the error composition, and protocol rankings can be close. [Theorem 2 and discussion](</Users/lorenzolacorte/Documents/SoA/on-the-accuracy-of-twirled-approximations-in-repeater-chains/main.tex:850>).

       For **postselected swapping**, Davies et al. ask a different question: how much can the fidelity of a particular Bell outcome vary among states with prescribed information? They use

       \[
       S_{p,F}=\{\rho=p\Phi^++(1-p)\sigma:\ \sigma\geq0,\ \mathrm{Tr}\sigma=1,\ \mathrm{Tr}(\Phi^+\rho)=F\}.
       \]

       Here \(p\) is a guaranteed ideal-state mixture weight, not a link-generation or swap probability, and generally \(p\neq F\). Fixing \(F\) alone gives a loose comparison: a partially entangled pure input can yield a perfect pair for some Bell outcomes. Fixing both \(p\) and \(F\) restricts the admissible noise structure. Their tight upper bound for the two-input problem is \(F'_{\max}(p,F)=1-2p(1-F)\); they also derive lower bounds and study fixed outcome probability to expose the probability–fidelity tradeoff. This is an extremization over input states, not an optimization over memory lifetimes or repeater trees. [State family and bounds](</Users/lorenzolacorte/Documents/SoA/on-the-accuracy-of-twirled-approximations-in-repeater-chains/main.tex:972>).

       Their QKD example further distinguishes

       \[
       r\!\left(\sum_s p_s\rho_s\right)
       \quad\text{and}\quad
       \sum_s p_s r(\rho_s).
       \]

       The latter processes data separately by swap syndrome; it does not require immediately rejecting a link and restarting the repeater whenever an unwanted syndrome occurs. They compare an extremal coherent-error family with the R-state family \(\rho_R=F\Phi^++(1-F)|01\rangle\langle01|\), at the same input fidelity, and find different benefits from syndrome information. Their plots concern secret fraction, with suitable measurement-basis choices, rather than our waiting-time-normalized rate. The R state is representable by our real-X backend; their coherent superposition of even- and odd-parity Bell states is not generally a real-X state in our fixed computational basis. We should not assume that every example in their paper is directly supported. [QKD example](</Users/lorenzolacorte/Documents/SoA/on-the-accuracy-of-twirled-approximations-in-repeater-chains/main.tex:1587>).

       **What Mondal et al. compare.** They instead fix a homogeneous repeater hardware model: fresh \(\Phi^+\) pairs, probabilistic generation, local AD during storage, and deterministic instantaneous swapping. Their approximation replaces each local AD channel by its Pauli channel twirl,

       \[
       \overline{\mathcal A}(\rho)=\frac14\sum_{P\in\{I,X,Y,Z\}}P\,\mathcal A(P\rho P)P.
       \]

       They compare fidelity and concurrence under specified swap-at-last, nesting, and swap-ASAP policies, without distillation or cutoffs. Their Pauli-twirled family is generally Bell-diagonal, **not Werner**; its single age parameter does not imply isotropic error populations. Their reported advantage compares the two models for a fixed policy, not the physical performance of protocols independently optimized in each model. This gives us a channel-approximation baseline, while our proposed contribution concerns the decisions selected by an optimizer. [Mondal et al., Sections II.6–III](https://arxiv.org/html/2509.18064v1#S2.SS6).

       **Why the two comparisons differ: an explicit calculation.** The following derivation uses our channel conventions. Write a normalized real-X state as

       \[
       x=(1,z_R,z_L,c_{ZZ},c_{XX},c_{YY}).
       \]

       Bell state twirling sets \(z_L=z_R=0\), leaving the three correlators unchanged. In contrast, a local affine channel acts as \((x,y,z)\mapsto(a_xx,a_yy,a_zz+b)\); Pauli channel twirling removes \(b\). For independent left and right channels,

       \[
       c'_{ZZ}=a_{z,L}a_{z,R}c_{ZZ}
       +a_{z,L}b_Rz_L+b_La_{z,R}z_R+b_Lb_R.
       \]

       There are two distinct losses of information:

       - Projecting the **input state** to Bell-diagonal form removes the middle two terms, but retains \(b_Lb_R\) when the physical channels are applied.
       - Twirling the **local channels** removes both translations, including the \(b_Lb_R\) term.

       For a perfect Bell pair stored under equal AD channels with survival parameter \(\eta\), the full state has

       \[
       z_L=z_R=1-\eta,\qquad
       c_{XX}=-c_{YY}=\eta,\qquad
       c_{ZZ}=\eta^2+(1-\eta)^2.
       \]

       Twirling this state afterwards preserves its fidelity. Applying the locally twirled channels instead gives \(c_{ZZ}=\eta^2\). Therefore

       \[
       F_{\mathrm{AD}}=\frac{1+\eta^2}{2},\qquad
       F_{\mathrm{channel\ twirl}}=\frac{(1+\eta)^2}{4},\qquad
       F_{\mathrm{AD}}-F_{\mathrm{channel\ twirl}}=\frac{(1-\eta)^2}{4}.
       \]

       For \(\eta=0.8\), the full state and its Bell-diagonal state twirl both have \(F=0.82\), whereas local channel twirling gives \(F=0.81\). After two such intervals, full AD gives \(F=0.7048\); inserting a Bell state twirl between intervals gives \(F=0.6888\); using the twirled channels throughout gives \(F=0.6724\). These values were checked with the current repository's channel maps. They are illustrative state-evolution calculations, not optimized repeater results.

       Consequently,

       \[
       \mathcal B\circ(\mathcal A_L\otimes\mathcal A_R)
       \neq\overline{\mathcal A}_L\otimes\overline{\mathcal A}_R
       \]

       even on Bell input. Davies' exactness theorem places no requirement that these different channels agree. Nor does its preservation of a final Bell-diagonal observable imply that deleting local biases *before further damping* is harmless. The latter biases can later feed into \(c_{ZZ}\), hence into the bit-error rate. [Our affine-channel implementation](src/core/pauli_adc/protocol_units.py).

       **A useful refinement concerning distillation.** For the particular real-X recurrence currently implemented—bilateral CNOT, accepting and summing equal target outcomes—the unnormalized success map \(D\) satisfies

       \[
       \mathcal B[D(\rho,\sigma)]=D(\mathcal B\rho,\mathcal B\sigma)
       \]

       on real-X inputs. The output trace and the \(XX,YY,ZZ\) components depend only on the input traces and correlators; only the local output biases require the input biases. The all-corrected swap has the same property. Thus this distillation step alone does not create a discrepancy in its immediate success probability or Bell populations when only those biases were removed. **Subsequent AD can make the discrepancy observable.** This follows directly from the implemented maps and is narrower than a statement about arbitrary distillation protocols. It supplies a useful zero-discrepancy control as well as a mechanism to test.

       **A precise set of approximation baselines.** Let \(G\) be a physical waiting or operation map, carrying unnormalized successful branches so their traces retain success probabilities. Use the same generation process, operations, reset rules, and available protocol choices in every model:

       | Model | State update | What the comparison isolates |
       |---|---|---|
       | Full real-X | Apply \(G\) without state projection | Reference dynamics within the supported real-X family |
       | Bell state projection | Apply \(G\), then \(\mathcal B\) | Loss of local biases before subsequent processing |
       | Werner state projection | Apply \(G\), then \(\mathcal W\) | Combined loss of biases and error anisotropy |
       | Local channel twirl | Replace each memory channel by its Pauli channel twirl | Loss of the non-unital channel translation, connecting to Mondal's comparison |

       For the state-projection models, specify the projection locations: after generation, after each actual waiting interval before the operation, and after each successful operation. Apply the maps linearly to weighted states, preserving their trace. This prescription defines a reproducible approximation rather than an unspecified “Werner noise model.” In particular, it does not allow choosing a convenient depolarizing coherence time independently of the reference channel. Changing the projection schedule defines a different approximation and should be reported as such.

       These projections belong to the **prediction model**. When evaluating the selected protocol in the full physical model, execute the selected swaps, distillations, and cutoffs without adding physical twirling gates. If deliberate physical twirling is to be a protocol choice, include it explicitly and consistently in the action space instead. Otherwise prediction error and the performance of a physically modified protocol get conflated.

       For the current local Pauli–AD channel, with Pauli-mode rates \((\Gamma_X,\Gamma_Y,\Gamma_Z)\) and damping rate \(\kappa\), the local channel-twirl baseline is particularly simple. Set AD to zero and use Pauli-mode rates

       \[
       (\Gamma_X+\kappa/2,\ \Gamma_Y+\kappa/2,\ \Gamma_Z+\kappa).
       \]

       This preserves the three contraction factors of the full channel and removes its affine translation exactly. It can be evaluated through the general Pauli-rate inputs of the real-X backend. The state-projection baselines require explicit projection hooks; the existing optional Werner twirl after real-X distillation does not implement the complete schedule above.

       **The physical scenario I would choose first.** Use four elementary links, initially with identical generation probabilities and the same elementary Werner state of fixed fidelity \(F_0\). Choosing Werner inputs deliberately removes initial-state anisotropy as an explanation for the outcome. Keep the swap hardware success probability and distillation hardware model identical across all comparisons. Start with all corrected Bell outcomes and the existing pooled X/Z BB84 objective; leave syndrome processing and outcome-selective rejection to separate experiments.

       Model each memory with AD plus pure Z dephasing:

       \[
       \mathcal E_t=\mathcal D_{\phi,t}\circ\mathcal A_t,\qquad
       \kappa=1/T_1,\qquad
       a_x=a_y=e^{-(\kappa/2+\gamma_\phi)t},\qquad
       a_z=e^{-\kappa t},\qquad b=1-e^{-\kappa t}.
       \]

       Here \(\gamma_\phi\) is defined as the transverse Bloch-decay rate from pure dephasing, so \(T_2^{-1}=(2T_1)^{-1}+\gamma_\phi\). The corresponding repository inputs are \(\texttt{pauli\_mode\_decay\_rates}=(\gamma_\phi,\gamma_\phi,0)\) and \(\texttt{amplitude\_damping\_rate}=\kappa\). This gives identifiable relaxation and dephasing mechanisms. In this case the two channels commute, so the current sequential implementation also represents the combined continuous-time semigroup. General Pauli noise with \(\Gamma_Z>0\) needs more care: composing the two finite-duration channels is not generally the same as simultaneous evolution under the sum of their generators.

       Sweep the ratio of relaxation to dephasing and the memory times relative to the elementary generation time. A useful controlled slice holds \(T_2\) fixed and varies \(\kappa\), with \(\gamma_\phi=T_2^{-1}-\kappa/2\geq0\): identical transverse decay then conceals different longitudinal relaxation. This tests whether a coherence specification alone predicts the best protocol. Within *each* physical parameter point, all approximation baselines must come from that same \(\mathcal E_t\); do not refit their parameters to the optimum or to end-to-end output data.

       **Experiments that make the conclusion substantive.** Run the following sequence, separating established controls from questions requiring numerical results:

       - **Davies control:** start from prescribed real-X inputs (including AD-prepared states), then perform only ideal non-postselected swapping, with no further memory noise. Full and Bell state-twirled inputs must give identical output Bell populations and pooled secret fraction. This is a correctness check, not evidence of an optimization advantage.
       - **Pauli control:** set \(\kappa=0\) and use Bell-diagonal inputs. Full real-X, Bell projection, and local channel twirling must agree for the supported maps. The Werner baseline can differ because it loses anisotropy, including anisotropy generated by untwirled recurrence distillation.
       - **Memory experiment:** turn on AD between operations and optimize swap trees and cutoffs, initially without distillation. Determine whether the discarded biases or channel translations change tree rankings or optimal cutoff values. A discrepancy in predicted rate is not sufficient evidence.
       - **Distillation experiment:** allow a bounded number of recurrence rounds and jointly optimize their locations with cutoffs. Look for parameter regions where the approximate model chooses unnecessary purification, misses beneficial purification, or chooses a different storage–retry tradeoff. These outcomes are hypotheses, not guaranteed effects of AD.
       - **Heterogeneous experiment:** introduce one slow link or one memory with a different \(T_1/T_2\), keeping the hardware fixed across models. Test whether approximation loss is concentrated where the swap tree changes which memories remain exposed to waiting noise.

       Record the full-model rate of every selected protocol, changes in the selected actions, and absolute rate loss as well as relative loss. If several protocols tie or are nearly tied in an approximate model, evaluate their full-model spread rather than allowing arbitrary tie-breaking to determine the reported result. Use enumeration on the small protocol space where feasible; otherwise report a best-found reference rather than claiming a certified optimum. A zero full-model optimum makes relative loss undefined, and a tiny positive rate can make a large percentage gain practically unimportant.

       There is a useful general bound behind this distinction. If an approximate rate \(\widehat R\) obeys \(|\widehat R(\pi)-R(\pi)|\leq\delta\) uniformly over the candidate protocols, and both optimizations are exact, then

       \[
       0\leq R(\pi^*)-R(\widehat\pi^*)\leq2\delta.
       \]

       This follows by inserting \(\widehat R\) between the two true rates and using the optimality of \(\widehat\pi^*\). It is a statement about **rate-prediction error**, not Davies' fidelity bound. A ranking reversal can still occur when the top protocols are separated by less than the approximation error, while a large prediction error can leave the ranking unchanged.

       The resulting question is therefore precise: **for a fixed physical relaxation/dephasing model and a fixed protocol space, when are Bell populations—or fidelity alone—sufficient to choose a near-optimal protocol, and when does the future use of discarded information change that decision?** Mapping the safe regimes and explaining the failures would be a stronger result than demonstrating that AD states are not Werner states.

    2. Extract (to later compare) optimal protocols under:
        - (a) Werner approximation;
        - (b) Bell-diagonal state-projection approximation, with local Pauli channel twirling evaluated as a separate baseline;
        - (c) Pauli-AD model (full model)

   3. Evaluate (a,b,c) protocol under the same full model.
    The central quantity should be the loss from choosing the wrong protocol:
   \[
   \mathcal L=
   1-\frac{R_{\mathrm{full}}(\pi_{\mathrm{approx}}^*)}
           {R_{\mathrm{full}}(\pi_{\mathrm{full}}^*)},
   \]
   wherever the denominator is positive.

   This distinguishes inaccurate rate predictions from consequentially wrong decisions. 
   An approximation could misestimate rates yet select an excellent protocol — or predict rates fairly well while selecting the wrong distillation placement near a transition.

   **Useful insight:** identify where a simple model is sufficient and where accurate noise characterization actually pays off.


2. **Are cutoffs and distillation substitutes, or do they work best together?**

   Compare four optimizations under identical hardware:
   - neither distillation nor finite cutoffs;
   - cutoffs alone;
   - distillation alone;
   - jointly optimized cutoffs and distillation.

   Repeat across Pauli bias, damping strength, generation probability, and swap efficiency. The interesting boundary is where removing old pairs makes distillation worthwhile—or where distillation makes longer storage worthwhile.

   **Useful insight:** a regime map prescribing “discard aggressively,” “purify,” or “combine both,” with operation-specific cutoff rules.

   A stronger extension asks whether equally old pairs should have different cutoffs depending on their error composition and which operation comes next. Memory-time cutoffs are implemented; state-dependent real-X cutoffs would need additional logic and careful treatment of the available classical history.

3. **Under amplitude damping, when is accepting fewer Bell outcomes worth the retries?**

   Your implementation already supports accepting all corrected outcomes, the \(\Phi\) pair, or the \(\Psi\) pair. For normalized input links, its intrinsic pair-acceptance probabilities are
   \[
   p_{\Phi}=\frac{1+z_R^{(1)}z_L^{(2)}}2,\qquad
   p_{\Psi}=\frac{1-z_R^{(1)}z_L^{(2)}}2.
   \]
   Here the \(z\)'s are the local polarizations of the measured qubits. These vanish for Bell-diagonal inputs but generally develop under damping. The accepted output states also differ. [Implemented maps](/Users/lorenzolacorte/Research/lilla/src/core/pauli_adc/protocol_units.py:277)

   Optimize outcome selection together with the protocol and cutoffs, measuring the final key rate including failed attempts.

   **Useful insight:** determine whether postselection can replace a distillation round, extend the positive-key regime, or becomes too expensive when generation is slow.

   Fixed outcome policies are supported now. Choosing them separately at each swap, or according to observed waiting times, requires extending the controls. Comparisons should use the outcomes actually distinguishable by the assumed hardware.


Together, (1) approximation-induced optimization loss, (2) joint distillation–cutoff regime maps, (3) and Bell-outcome selection under damping
could support a coherent paper showing *when realistic noise changes protocol decisions, why those decisions change, and how much performance is recovered*.
Timing binning is an attractive additional experiment because much of the required output is already available.

For a manageable first campaign, use four elementary links, enumerate all swap trees, cap distillation at one round per tree location, and optimize memory cutoffs.
Begin with homogeneous chains, then introduce one slow link or one poor memory to isolate mechanisms. Report protocol changes alongside absolute key rates, positive-key boundaries, and relative gains; verify winning protocols with a longer time horizon.

Two modeling details should remain explicit throughout: the current combined channel applies amplitude damping followed by Pauli noise over each waiting interval, and the protocol evaluator uses fixed recursive protocols with full-reset retries. These define the scope of the conclusions; arbitrary online scheduling and simultaneous noise generators are out of scope.
