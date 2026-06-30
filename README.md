# MLSE & Viterbi Algorithm Tutorial

Maximum Likelihood Sequence Estimation (MLSE) and Viterbi Algorithm — a hands-on tutorial with Python simulations, trellis visualizations, and BER performance comparisons.

---

## Content Structure

```
MLSE-Viterbi-Tutorial/
├── docs/
│   ├── MLSE-Tutorial.md               # Complete theory tutorial (Chinese)
│   ├── Trellis-State-Transitions.md  # Detailed Trellis state transition walkthrough
│   ├── viterbi-forward-trace.txt     # Forward recursion step-by-step log
│   └── sliding-window-viterbi.txt    # Real-time sliding-window Viterbi log
├── src/
│   ├── mlse_viterbi_demo.py          # Main simulation (BER comparison + plots)
│   ├── viterbi_forward_trace.py      # Forward recursion detailed demo code
│   └── sliding_window_viterbi.py     # Real-time sliding-window Viterbi code
└── images/
    ├── ber-performance.png             # BER performance comparison
    ├── viterbi-trellis-path.png      # Survivor path Trellis diagram
    └── trellis-convergence.png       # Path convergence illustration
```

---

## Quick Start

1. Clone the repo
   ```bash
   git clone https://github.com/rongjun72/MLSE-Viterbi-Tutorial.git
   cd MLSE-Viterbi-Tutorial
   ```

2. Run the main simulation
   ```bash
   cd src
   python mlse_viterbi_demo.py
   ```

3. Run the forward trace demo
   ```bash
   python viterbi_forward_trace.py
   ```

4. Run the sliding-window Viterbi demo
   ```bash
   python sliding_window_viterbi.py
   ```

---

## Core Concepts

### Problem Definition

In digital communication, transmitted symbol sequences pass through a multipath channel, causing each received sample to contain interference from neighboring symbols (ISI). MLSE aims to find the **complete sequence** that makes the received signal "look most like" it.

### Viterbi Algorithm

The Viterbi algorithm uses **Trellis (grid graph)** and **dynamic programming** to reduce complexity from $O(M^N)$ to $O(N \cdot M^L)$:

- **State**: combination of the most recent $L-1$ symbols
- **Branch Metric**: Euclidean distance between received signal and expected output
- **Survivor**: only the path with the smallest cumulative metric is kept for each state
- **Traceback**: trace back from the final state to obtain the optimal sequence

### Engineering Practice: Sliding-Window Viterbi

Real-time systems cannot wait until the end of the frame to traceback. When the decision depth $D = 5L$, paths have already converged, so the symbol at time $t-D$ can be safely output with a fixed latency of only $D$ symbol periods.

---

## Simulation Highlights

| Metric | Result |
|--------|--------|
| Channel | 3-tap ISI: `[0.8, 0.5, 0.3]` |
| Modulation | BPSK |
| States | 4 ($2^{L-1}$) |
| Decision Depth | 15 ($5L$) |
| High-SNR BER | **0%** (MLSE completely eliminates ISI) |
| No Equalization BER | ~12-20% (ISI floor) |

---

## References

1. G. D. Forney Jr. — "The Viterbi Algorithm" (1973)
2. Proakis & Salehi — *Digital Communications* — Chapter 10
3. J. R. Barry, E. A. Lee, D. G. Messerschmitt — *Digital Communication* — Section 6.4
4. S. M. Kay — *Fundamentals of Statistical Signal Processing*

---

*Created with [Kimi Work](https://www.moonshot.cn)*