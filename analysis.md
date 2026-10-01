
- when the noise model changes the optimal protocol? what physical mechanism explains that change?
- extracting design rules

In the following, I may abbreviate Amplitude Damping (AD)

I may refer to work on twirling accuracy and amplitude-damping advantages. 
The sharper contribution would be *optimization loss across swapping, distillation, and cutoffs*. Also include a control regime where Bell-diagonal reduction is known to preserve the relevant observables for non-postselected swapping. [Davies et al.](https://arxiv.org/abs/2509.16689), [Mondal et al.](https://arxiv.org/abs/2509.18064)


1. **How much performance do we lose by optimizing with the wrong noise model?**

    1. Set a scenario
    The key question is: what noise model should I use there? (It is kinda trivial that if I put AD then the Werner approximation is unprecise.. how are Davies et al. Mondal et al. posing this comparison?)

    2. Extract (to later compare) optimal protocols under:
        - (a) Werner approximation;
        - (b) Bell-diagonal or Pauli-twirled approximation;
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

2. **How should distillation change with the dominant Pauli error?**

   Sweep the error composition while holding initial fidelity fixed, and optimize the number and placement of distillation rounds. Ask when the best choice changes between elementary-link distillation, intermediate distillation, final distillation, and no distillation.

   There is a particularly useful mechanism in your [current recurrence map](/Users/lorenzolacorte/Research/lilla/src/core/pauli_adc/protocol_units.py:326). For two identical inputs
   \[
   \rho=F\Phi^+ +(1-F)\Phi^-,
   \]
   its fixed-basis recurrence succeeds with probability one but produces
   \[
   F'=F^2+(1-F)^2<F \qquad (1/2<F<1).
   \]
   Thus, this recurrence can amplify phase errors despite always “succeeding.”

   **Useful insight:** establish which error compositions justify purification, and which require changing its basis. Extending the protocol space with basis choices could reveal much larger gains than optimizing the number of rounds alone.

   Fixed-basis comparisons are supported now. Basis-adaptive purification requires new operations; under amplitude damping, some rotations also require extending the state representation.

3. **Are cutoffs and distillation substitutes, or do they work best together?**

   Compare four optimizations under identical hardware:
   - neither distillation nor finite cutoffs;
   - cutoffs alone;
   - distillation alone;
   - jointly optimized cutoffs and distillation.

   Repeat across Pauli bias, damping strength, generation probability, and swap efficiency. The interesting boundary is where removing old pairs makes distillation worthwhile—or where distillation makes longer storage worthwhile.

   **Useful insight:** a regime map prescribing “discard aggressively,” “purify,” or “combine both,” with operation-specific cutoff rules.

   A stronger extension asks whether equally old pairs should have different cutoffs depending on their error composition and which operation comes next. Memory-time cutoffs are implemented; state-dependent real-X cutoffs would need additional logic and careful treatment of the available classical history.

4. **Under amplitude damping, when is accepting fewer Bell outcomes worth the retries?**

   Your implementation already supports accepting all corrected outcomes, the \(\Phi\) pair, or the \(\Psi\) pair. For normalized input links, its intrinsic pair-acceptance probabilities are
   \[
   p_{\Phi}=\frac{1+z_R^{(1)}z_L^{(2)}}2,\qquad
   p_{\Psi}=\frac{1-z_R^{(1)}z_L^{(2)}}2.
   \]
   Here the \(z\)'s are the local polarizations of the measured qubits. These vanish for Bell-diagonal inputs but generally develop under damping. The accepted output states also differ. [Implemented maps](/Users/lorenzolacorte/Research/lilla/src/core/pauli_adc/protocol_units.py:277)

   Optimize outcome selection together with the protocol and cutoffs, measuring the final key rate including failed attempts.

   **Useful insight:** determine whether postselection can replace a distillation round, extend the positive-key regime, or becomes too expensive when generation is slow.

   Fixed outcome policies are supported now. Choosing them separately at each swap, or according to observed waiting times, requires extending the controls. Comparisons should use the outcomes actually distinguishable by the assumed hardware.

5. **Which memories should survive longest—and where should better hardware go?**

   In a heterogeneous chain, swapping determines which memories are measured out and which remain as endpoints of larger links. This creates a competition between balancing preparation times and avoiding prolonged storage in poor memories.

   Hold the hardware inventory fixed, permute its placement, and optimize the swap tree and distillation placement. Separately, improve one node’s damping or Pauli-noise rate at a time and reoptimize.

   **Useful insight:** determine whether good memories belong at endpoints, beside slow links, or at particular internal positions—and whether the answer changes between dephasing and relaxation.

   The resulting marginal gains would answer a practical question: **which hardware improvement gives the largest increase in optimized key rate?** Evaluating this after reoptimization matters because an upgrade may change the preferred protocol entirely.

6. **How much key rate can be recovered simply by retaining timing information?**

   The [current objective](/Users/lorenzolacorte/Research/lilla/src/utils/utility_functions.py:144) averages the error rates before computing the secret fraction. But the simulator already returns the conditional mean state \(\rho(t)\) for each completion time.

   Compare
   \[
   R_{\mathrm{pooled}}
   =\frac{r\!\left(\sum_t p_t\rho(t)\right)}{\mathbb E[T]}
   \quad\text{with}\quad
   R_{\mathrm{binned}}
   =\frac{\sum_t p_t\,r(\rho(t))}{\mathbb E[T]},
   \]
   using the same asymptotic BB84 expression and a converged PMF. Separate processing of sufficiently populated timing bins can exploit differences hidden by pooling; under this expression, convexity gives \(R_{\mathrm{binned}}\ge R_{\mathrm{pooled}}\).

   **Useful insight:** measure whether timing information changes the optimal cutoff or removes the need for a distillation round. Start with a few coarse bins to assess how much information is actually needed.

   Binning has already been studied for swap-ASAP chains; your opportunity is its interaction with heterogeneous noise, distillation, and protocol optimization. Completion time also does not reveal the entire storage history, so this measures the value of one accessible label. [Goodenough et al.](https://arxiv.org/abs/2404.07146)

7. **Does it matter which distillation input we keep?**

   Compare keeping the fresher pair, keeping the older pair, and the current fixed input ordering.

   There is a subtle reason this is interesting: in your recurrence map, exchanging the inputs preserves the immediate success probability and two-qubit correlators, but can change the local polarizations. Consequently, the immediate BB84 secret fraction can be identical, while subsequent amplitude damping produces different performance. The repository already [tests this role asymmetry](/Users/lorenzolacorte/Research/lilla/src/tests/pauli_adc/test_protocol_metamorphic.py:126).

   **Useful insight:** show whether two outputs with identical immediate fidelity and key fraction have different value for the remaining repeater protocol.

   This would expose a concrete limitation of optimizing individual operations by their immediate output quality. The maps exist; selecting the kept input according to arrival time requires an added policy.

**I would start with three connected studies:** approximation-induced optimization loss, joint distillation–cutoff regime maps, and Bell-outcome selection under damping. Together they could support a coherent paper showing *when realistic noise changes protocol decisions, why those decisions change, and how much performance is recovered*. Timing binning is an attractive additional experiment because much of the required output is already available.

For a manageable first campaign, use four elementary links, enumerate all swap trees, cap distillation at one round per tree location, and optimize memory cutoffs. Begin with homogeneous chains, then introduce one slow link or one poor memory to isolate mechanisms. Report protocol changes alongside absolute key rates, positive-key boundaries, and relative gains; verify winning protocols with a longer time horizon.

Two modeling details should remain explicit throughout: the current combined channel applies amplitude damping followed by Pauli noise over each waiting interval, and the protocol evaluator uses fixed recursive protocols with full-reset retries. These define the scope of the conclusions; arbitrary online scheduling and simultaneous noise generators would be separate extensions.