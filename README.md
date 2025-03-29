# Robustness Evaluation of RobustBench Models using AutoAttack and FMN

## Overview

This project compares the robustness of 5 models from RobustBench when attacked using:
- **AutoAttack**: The standard benchmark attack suite.
- **FMN**: A fast and effective minimum-norm adversarial attack.

---

## Models 

| Model | Description | Paper Link |
|-------|-------------|------------|
| Carmon2019Unlabeled | Unlabeled data improves adversarial robustness by leveraging semi-supervised learning (TRADES + pseudo-labels) | [Unlabeled Data Improves Adversarial Robusntess](https://arxiv.org/abs/1905.13736) |
| Wang2023Better | Proposes improved loss landscape shaping for better robust generalization | [Better Diffusion Models Further Improve Adversarial Training](https://arxiv.org/abs/2302.04638) |
| Cui2023Decoupled | Introduces Decoupled Sharpness-Aware Minimization (DSAM) to reduce sharpness and improve adversarial robustness | [Decoupled Kullback-Leibler Divergence Loss](https://arxiv.org/abs/2305.13948) |
| Xu2023Exploring | Explores flatness and sharpness in the loss landscape to enhance model robustness | [Exploring and Exploiting Decision Boundary Dynamics for Adversarial Robustness](https://arxiv.org/abs/2302.03015) |
| Rade2021Helper | Leverages helper models to guide robust training and achieve better certified and empirical robustness | [Helper-based Adversarial Training: Reducing Excessive Margin to Achieve a Better Accuracy vs. Robustness Trade-off](https://openreview.net/forum?id=BuD2LmNaU3a) |

---

## Dataset

**CIFAR-10**, tested on the **first 100 validation samples**.

---


## Attacks Comparison

In this project, we compare two strong adversarial attacks widely used in adversarial robustness evaluation:

### ✅ AutoAttack

AutoAttack is a **reliable robustness benchmark** based on an ensemble of four attacks:

- **APGD-CE**: Maximizes the cross-entropy loss.
- **APGD-DLR**: Maximizes the DLR loss, often more effective against robust models.
- **FAB**: Finds boundary-crossing adversarial examples with **minimum ℓp-norm**.
- **Square Attack**: A **black-box** attack based on random square perturbations.

AutoAttack works under a **fixed ε constraint** and systematically explores different adversarial directions. It is considered the gold standard for evaluating robustness in RobustBench.

---

### ✅ Fast Minimum-Norm (FMN) Attack

FMN is an efficient gradient-based attack that **directly searches for the minimal perturbation** able to fool the model:

- FMN focuses on finding the **closest boundary-crossing adversarial sample**, minimizing the perturbation norm.
- Can operate:
    - In a **bounded** way (e.g., ε = 8/255).
    - Or **unbounded**, searching for the closest adversarial even outside the usual ε constraint.
- Faster than AutoAttack and effective when computing minimal-norm adversarial examples.

---

### Comparison

| Aspect | AutoAttack | FMN |
|--------|------------|-----|
| Composition | Ensemble of 4 attacks | Single attack |
| Goal | Explore multiple adversarial objectives | Find the closest misclassified sample |
| Perturbation | Always inside ε-ball | Can be bounded or unbounded |
| Behavior | Maximizes loss within ε | Minimizes perturbation norm |
| Speed | Slower | Faster |
| Robustness Scores | RobustBench Standard | Comparable when ε-bounded |

---

The figure below illustrates the difference between AutoAttack and FMN on a toy problem:

- The **red curve** represents the model's decision boundary.
- The **black dashed circle** is the ε-ball constraint.
- The **black cross** is the original input.
- AutoAttack (left) explores multiple directions within the ε-ball to find adversarial examples.
- FMN (right) finds the **closest boundary-crossing perturbation**, either inside the ε-ball (**bounded**) or outside (**unbounded**).

![AutoAttack vs FMN](https://github.com/user-attachments/assets/8ebd82a2-a5b2-4ac5-95f3-97b72988d5e3)

This explains why FMN is often faster and sometimes finds different adversarial samples compared to AutoAttack, especially when used without an ε-bound.

---
