# Neural Foundations 

## Core Question

**Can machines learn like biological neurons?**

This era is the "birth of the idea" that a simple artificial neuron + learning rule could mimic the brain.

## What Happened Here

- **1943** — McCulloch & Pitts: First mathematical model of a neuron (binary threshold logic gate).
- **1949** — Donald Hebb: "Cells that fire together, wire together" → Hebbian learning rule.
- **1957–1958** — Frank Rosenblatt invents the **Perceptron** (Mark I hardware version actually built and demonstrated at Cornell).
  - First machine that could **learn from examples** automatically.
  - Rosenblatt called it "the first machine capable of having an original idea."
- **1959** — Widrow & Hoff create ADALINE/MADALINE — first real-world application (noise cancellation on phone lines).
- **1969** — Minsky & Papert publish the book _Perceptrons_. (The world belivied in his words)
  - Mathematically proved what single-layer perceptrons **cannot** do.
  - Famous example: **XOR problem** (non-linearly separable) → impossible for a single perceptron.
- Result: Massive disappointment → funding dried up → first **AI Winter** (1970s–mid 1980s). Neural nets were declared "dead" by most of the community.

## Major Breakthroughs

1. **Perceptron** — First trainable artificial neural network.
2. **Hebbian Learning** — Unsupervised "fire together → wire together" rule.
3. **Perceptron Convergence Theorem** — Proven that it will find a solution **if** the data is linearly separable.
4. **Weight-update learning** — Core idea still used today: "error → adjust weights".
5. Early gradient descent ideas and simple optimization (though not yet backprop).

## Biggest Problems We Solved

- Machines could now **automatically learn** simple patterns from data (instead of hard-coded rules).
- Proved that connectionist (neural) approach was possible at all.

## Biggest Problem That Remained

**Cannot solve non-linearly separable problems** (XOR, most real-world data).

→ Single-layer networks are too weak.

## What Shifted Us to Era 1 (1986–2011)

The XOR critique made everyone realize:  
**We need multi-layer networks (hidden layers) + non-linear activations.**

But… there was **no known way** to train the hidden layers efficiently.

The breakthrough that ended the first AI Winter:

- **Backpropagation** (popularized in 1986 by Rumelhart, Hinton & Williams) + sigmoid activations + chain rule.
- Suddenly we could train **multilayer perceptrons (MLPs)**.
