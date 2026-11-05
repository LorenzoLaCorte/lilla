I want to present my research to my collaborators.

Slides concise and minimal, almost no text.
Include as many figures and infographics as possible.
If there some detail needed, add it concisely as \note{}

**Title**: Efficient Tracking of Quantum Repeater Protocols under Non-unital Memory Noise.

**Abstract skeleton**: *We develop a deterministic evaluation of repeater-chain protocols including entanglement swapping, distillation and memory cutoffs. We track the evolution of Werner, Bell-diagonal, or X- states under local Pauli noise and amplitude damping. We find state representations whose coefficients evolve according to simple multiplicative relations. A factorization of the operational and noise maps enables quasilinear evaluation in the time horizon of the protocol. We establish when the Bell-diagonal approximation of states preserves the relevant [...], and identify how non-unital noise and outcome selection instead invalidate [...]. We show how optimizing repeater protocols with twirled approximations can select policies with substantially lower physical secret-key rate. Furthemore, we explore [...]. Our contributions [...].* 

**Method:** 
- Develop an algorithm based on analytical expressions for computing distributions for the E2E link:
- - the probability distribution of waiting time
- - the distribution of average fidelity of the produced entanglement

- Track repeater chain protocols with
- - entanglement generation (direct transmission)
- - entanglement swapping: doubling scheme [Li21], asymmetric [LaC25], postselected [new]
- - entanglement distillation (BBPSW): on all the links of the level [Li2021], on any link of any level [LaC25]
- - cutoffs [Li2021], e.g., maximum storage time

- Optimize:
- - cutoffs to maximize the secret-key rate [Li21]: finding that the use of the optimal cutoff increases the secret-key rate, that [...]
- - swapping and distillation to maximize the secret-key rate: finding the optimal swapping scheme/order and distillation count and order, via protocol space enumeration and a Bayesian optimization algorithm; finding that distillation, when beneficial, should be performed early on in the protocol, and that the optimal swapping scheme depends on the heteroeneity of the hardware.

- Considering (state representation / memory noise channel)
- - Werner states and 2-memory joint depolarizing noise channel [Li2021], local depolarizing noise channels [LaC25]
- - Bell-diagonal states and local Pauli noise channels [new]
- - X-states and local Pauli and amplitude damping channels [new]

About our work:
Towards more general than Werner or Bell-diagonal simulation, while cheaper than propagating a generic 4×4 density matrix.
Under the assumptions, we develop a state representation of six unnormalized Pauli coordinates close under the operations.
Their (success-weighted) waiting time and fidelity distributions have a finite exponential factorization, enabling efficient deterministic evaluation. 
We do not cover arbitrary two-qubit states/channels: [make an example]


**Assumptions:**
- independent local Pauli noise plus amplitude damping, 
- real-X input states, 
- corrected/selected Bell swaps, 
- one recurrence primitive, 
- heterogeneous node rates and memory-time cutoffs


**Theory:** identify the precise reduction boundary. Argue that (and why) the Bell approximation (only tracking Bell state populations) is sufficient under the unital/all-corrected operation set; argue that local affine translations and postselection expose the discarded biases. 

**Analysis:** 
Use identical hardware (a systematic and realistic hardware grid) to measure the true SKR lost by a policy optimized under an approximated/reduced state/noise model. 
The preliminary evidence includes 59.3% two-link and 54.2% four-link rate loss from a fidelity-matched Werner policy, a separate 13.8% three-link cutoff loss, and a 4.94% overestimate of the memory lifetime needed for key when the local amplitude-damping channel is Pauli-twirled.
Design experiment to:
- Show when there is or isn't loss, e.g., validating the ~/Documents/SoA/on-the-accuracy-of-twirled-approximations-in-repeater-chains/ non-postselection result “For non-postselected swapping, the Bell-diagonal approximation is exact for the Bell-diagonal parts of the final state, including the fidelity” and checking the same exact argument if distillation is involved.
- Show if loss if affected by:
- - noise (I expect Bell-diagonal states to be quite accurate for Pauli noise, but not accurate for non-unital)
- - size of the chain

Two states, a Werner state and an X-state, with identical fidelity can respond oppositely to BBPPSW distillation: their error composition determines whether fidelity increases or decreases.
Thus, fidelity alone (Werner parameter) cannot even predict the *sign* of this recurrence's fidelity change for Werner state phase-only inputs. These statements are exact algebraic results, with Davies et al. and prior X-state work explicitly credited. An experimenter can deliberately twirl the state before distillation. We could optimize this choice - wether before distillation we should preserve the state's error structure, rotate the purification basis, or erase that structure by twirling.

All these evaluations might then bring out **which are the optimal protocols and most importantly which are the good principles to follow to design good protocols**.

**Toward submission:** 
We want to publish in https://www.nature.com/collections/aahafehfad

Slides for possible TODOs:
(i) add another distillation recurrence? or explicitly motivate why the hardware has only the implemented recurrence (standard BBPPSW)? 
(ii) is SKR enough?
(iii) scaling and accuracy analysis?
(iv) other things to analyze?