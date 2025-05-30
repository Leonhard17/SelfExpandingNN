# SelfExpandingNN

**SelfExpandingNN** is an experimental research project exploring a novel architecture for **self-organizing neural networks**. Inspired by spiking neural networks (SNNs) and parallel computation principles, the network uses **iterative activation updates** and **entropy-based signals** to determine when and where to grow its structure dynamically.

This project is developed independently by me as part of an ongoing exploration into scalable, adaptive, and interpretable neural network designs.

---

## 📚 Project Overview

The core idea is to create a neural network that:
- Processes activations **iteratively** in a parallel, feedback-driven manner (inspired by SNNs, but without explicit spikes),
- Uses **entropy minimization** as a training signal to promote order and structured representations,
- Dynamically **adds new nodes** based on **information density** and **entropy**—allowing the network to **expand itself** when the task demands more capacity,
- Lays groundwork for **adaptive, self-organizing systems** that evolve their structure over time.

---

## ⚙️ Current Development Status

✅ Core architecture in place:  
- Node and Network classes implemented  
- Iterative activation computation loop designed

🛠️ **In progress**:
- Data-to-spike data generator: preps input for iterative processing  
- Entropy minimization objective: under design for training loop  
- Node expansion algorithm: conceptual framework in place; implementation pending  
- Initial priming logic: needed to seed the network for learning

🚧 Not yet scaled or tested on real-world datasets—early-stage prototype limited by compute and ongoing development.

---

## 🔍 Research Focus

- Parallel, iterative computation inspired by spiking and energy-based models
- Self-expanding networks driven by internal uncertainty and information density
- Entropy minimization as a guiding principle for learning and self-organization
- Aiming for adaptive, low-compute, interpretable architectures for future AI systems

---

## 📫 Contact

For questions, collaboration ideas, or feedback, feel free to reach out:

Email: [leonhardwaibl@gmail.com](mailto:leonhardwaibl@gmail.com)  

---

This project is an exploratory, self-funded research effort. Feedback, suggestions, and collaborations are warmly welcome!
