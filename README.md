# From GANs to RAG: A Journey Through Modern Deep Learning

<div align="center">

*A curated collection of foundational papers tracing the evolution from early generative models to modern retrieval-augmented language systems*

[![Papers](https://img.shields.io/badge/papers-14-blue.svg)](#-the-papers)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

## 📖 Overview

This repository presents a carefully structured reading path through 14 landmark papers that tell the story of modern deep learning. Starting with Generative Adversarial Networks and culminating in efficient Retrieval-Augmented Generation systems, this collection shows how each breakthrough built upon the previous one to shape today's AI landscape.

## 🎯 Why This Sequence?

Rather than presenting papers chronologically or by topic in isolation, this collection follows the **conceptual evolution** of ideas:

- **Generative foundations** → How neural networks learned to create
- **Attention mechanisms** → The architectural breakthrough that changed everything
- **Scale and pretraining** → Making models smarter through more data and compute
- **Cross-domain unification** → Extending transformers beyond text
- **Efficiency innovations** → Making powerful models practical
- **Knowledge augmentation** → Combining reasoning with retrieval

## 📚 The Papers

### Part I: Generative Foundations (2013-2014)

#### 1. [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
**Goodfellow et al., 2014**

The paper that launched a revolution in generative modeling. Introduced the adversarial training paradigm where two networks compete—one generating fake data, the other trying to detect it.

> *Key Innovation*: Game-theoretic approach to learning data distributions

**[📄 Read Paper](https://arxiv.org/abs/1406.2661)**

---

#### 2. [Auto-Encoding Variational Bayes (VAEs)](https://arxiv.org/pdf/1312.6114)
**Kingma & Welling, 2013**

A probabilistic approach to generative modeling with stable training and interpretable latent spaces. Provides an alternative path to GANs with better theoretical foundations.

> *Key Innovation*: Variational inference for learning continuous latent representations

**[📄 Read Paper](https://arxiv.org/pdf/1312.6114)**

---

### Part II: The Transformer Revolution (2017-2020)

#### 3. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
**Vaswani et al., 2017**

The foundational paper that introduced the Transformer architecture. Eliminated recurrence entirely, replacing it with self-attention mechanisms that could be parallelized efficiently.

> *Key Innovation*: Self-attention as the core building block, enabling parallelization and long-range dependencies

**[📄 Read Paper](https://arxiv.org/abs/1706.03762)**

---

#### 4. [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
**Devlin et al., 2018**

Demonstrated the power of pretraining Transformers on massive text corpora. Introduced masked language modeling and set new benchmarks across NLP tasks.

> *Key Innovation*: Bidirectional pretraining with masked language modeling

**[📄 Read Paper](https://arxiv.org/abs/1810.04805)**

---

#### 5. [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
**Liu et al., 2019**

Showed that BERT was undertrained. By scaling data, removing next sentence prediction, and training longer, achieved significant improvements without architectural changes.

> *Key Innovation*: Empirical demonstration that scale and training methodology matter as much as architecture

**[📄 Read Paper](https://arxiv.org/abs/1907.11692)**

---

#### 6. [An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale (ViT)](https://arxiv.org/abs/2010.11929)
**Dosovitskiy et al., 2020**

Proved that Transformers could match or exceed CNNs in computer vision when given sufficient data. Unified the architecture across text and vision domains.

> *Key Innovation*: Direct application of Transformers to image patches, enabling cross-modal architecture unification

**[📄 Read Paper](https://arxiv.org/abs/2010.11929)**

---

### Part III: Advanced Generative Models (2020)

#### 7. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2502.09992)
**Ho et al., 2020**

Reintroduced diffusion models with practical improvements that made them competitive with GANs. Became the foundation for modern image generation systems like Stable Diffusion and DALL-E.

> *Key Innovation*: Iterative denoising process that generates high-quality samples with stable training

**[📄 Read Paper](https://arxiv.org/abs/2502.09992)**

---

### Part IV: Large Language Models Era (2023-2024)

#### 8. [Large Language Models: A Survey](https://arxiv.org/abs/2402.06196)
**Zhao et al., 2023**

A comprehensive overview of the LLM landscape covering architecture, training, capabilities, and limitations. Essential context for understanding the current state of the field.

> *Key Innovation*: Systematic organization of LLM knowledge including emergent abilities and scaling laws

**[📄 Read Paper](https://arxiv.org/abs/2402.06196)**

---

#### 9. [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
**Touvron et al., 2023**

Demonstrated that high-performing language models could be trained efficiently with public data. Sparked the open-source LLM movement and enabled widespread experimentation.

> *Key Innovation*: Efficient training recipes and open release strategy that democratized LLM research

**[📄 Read Paper](https://arxiv.org/abs/2302.13971)**

---

#### 10. [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
**Jiang et al., 2024**

Introduced a practical sparse mixture-of-experts architecture that increases model capacity while keeping inference costs manageable through conditional computation.

> *Key Innovation*: Efficient scaling through sparse activation of expert sub-networks

**[📄 Read Paper](https://arxiv.org/abs/2401.04088)**

---

### Part V: Efficient Adaptation & Knowledge Integration (2020-2024)

#### 11. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
**Hu et al., 2021**

Enabled practical fine-tuning of massive models by learning low-rank updates to weight matrices. Made domain adaptation accessible without full retraining.

> *Key Innovation*: Parameter-efficient fine-tuning through low-rank decomposition

**[📄 Read Paper](https://arxiv.org/abs/2106.09685)**

---

#### 12. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
**Lewis et al., 2020**

Combined neural retrieval with generation to ground language models in external knowledge. Reduced hallucinations and enabled access to up-to-date information.

> *Key Innovation*: Hybrid architecture merging parametric knowledge (LLM) with non-parametric knowledge (retrieval)

**[📄 Read Paper](https://arxiv.org/abs/2005.11401)**

---

#### 13. [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)
**Gao et al., 2023**

A comprehensive survey of RAG techniques, frameworks, and applications. Provides systematic taxonomy of retrieval methods, generation strategies, and evaluation approaches for RAG systems.

> *Key Innovation*: Unified framework for understanding RAG architectures, from naive RAG to advanced RAG and modular RAG

**[📄 Read Paper](https://arxiv.org/abs/2312.10997)**

---

#### 14. [LightRAG: Simple and Fast Retrieval-Augmented Generation](https://huggingface.co/papers/2410.05779)
**Recent Work, 2024**

Streamlined RAG architecture focused on speed and simplicity. Represents the current direction toward efficient, production-ready knowledge-augmented systems.

> *Key Innovation*: Optimized retrieval pipeline with reduced latency and simpler deployment

**[📄 Read Paper](https://huggingface.co/papers/2410.05779)**

---

## 🗺️ Visual Timeline

```
2013  VAEs ─────────┐
                     ├──> Generative Foundations
2014  GANs ─────────┘
                     
2017  Transformers ──┐
2018  BERT ──────────┤
2019  RoBERTa ───────┼──> Transformer Revolution
2020  ViT ───────────┘

2020  Diffusion Models ──> Advanced Generation

2021  LoRA ──────────┐
2023  LLaMA ─────────┤
2023  LLM Survey ────┼──> Modern LLMs & Efficiency
2024  Mixtral ───────┘

2020  RAG ───────────┐
2023  RAG Survey ────┤──> Knowledge Integration
2024  LightRAG ──────┘
```

## 📊 Quick Reference Table

| # | Paper | Year | Category | Key Contribution |
|---|-------|------|----------|------------------|
| 1 | [GANs](https://arxiv.org/abs/1406.2661) | 2014 | Generative | Adversarial training |
| 2 | [VAEs](https://arxiv.org/pdf/1312.6114) | 2013 | Generative | Variational inference |
| 3 | [Transformers](https://arxiv.org/abs/1706.03762) | 2017 | Architecture | Self-attention mechanism |
| 4 | [BERT](https://arxiv.org/abs/1810.04805) | 2018 | Pretraining | Masked language modeling |
| 5 | [RoBERTa](https://arxiv.org/abs/1907.11692) | 2019 | Pretraining | Optimized training |
| 6 | [ViT](https://arxiv.org/abs/2010.11929) | 2020 | Vision | Transformers for images |
| 7 | [DDPM](https://arxiv.org/abs/2502.09992) | 2020 | Generative | Diffusion models |
| 8 | [LLM Survey](https://arxiv.org/abs/2402.06196) | 2023 | Survey | Comprehensive LLM overview |
| 9 | [LLaMA](https://arxiv.org/abs/2302.13971) | 2023 | LLM | Efficient open models |
| 10 | [Mixtral](https://arxiv.org/abs/2401.04088) | 2024 | LLM | Mixture of experts |
| 11 | [LoRA](https://arxiv.org/abs/2106.09685) | 2021 | Fine-tuning | Parameter efficiency |
| 12 | [RAG](https://arxiv.org/abs/2005.11401) | 2020 | Retrieval | Knowledge integration |
| 13 | [RAG Survey](https://arxiv.org/abs/2312.10997) | 2023 | Survey | RAG taxonomy & methods |
| 14 | [LightRAG](https://huggingface.co/papers/2410.05779) | 2024 | Retrieval | Fast RAG systems |

## 🎓 How to Use This Repository

### For Students
Start from paper 1 and work sequentially. Each paper builds on concepts from previous ones. Take notes on how ideas evolve and connect.

**Suggested Reading Path:**
1. **Foundations** (Papers 1-2): Understand generative modeling basics
2. **Core Architecture** (Papers 3-6): Master the Transformer and its applications
3. **Modern Techniques** (Papers 7-11): Learn about scaling and efficiency
4. **Knowledge Integration** (Papers 12-14): Explore how to combine LLMs with external knowledge

### For Researchers
Use this as a reference for understanding how current techniques emerged. Jump to specific sections based on your interests, but review the foundational papers if unfamiliar.

**Research Focus Areas:**
- **Generative AI**: Papers 1, 2, 7
- **Transformer Architecture**: Papers 3, 4, 5, 6
- **LLM Development**: Papers 8, 9, 10, 11
- **RAG Systems**: Papers 12, 13, 14

### For Practitioners
Focus on papers 8-14 for practical modern techniques, but skim papers 3-6 to understand the architectural foundations you're building on.

**Implementation Priority:**
1. Paper 11 (LoRA) - For fine-tuning existing models
2. Papers 12-14 (RAG) - For building knowledge-grounded applications
3. Paper 10 (Mixtral) - For understanding efficient scaling

## 💡 Key Themes Across Papers

- **Scale matters**: Larger models and more data consistently improve performance
- **Architecture unification**: Transformers work across modalities (text, images, etc.)
- **Efficiency is crucial**: Methods like LoRA and Mixtral make large models practical
- **External knowledge helps**: RAG systems reduce hallucinations and improve factuality
- **Simple ideas compound**: Each paper adds relatively simple innovations that combine powerfully
- **From theory to practice**: Evolution from foundational concepts to production-ready systems

## 🔄 How Papers Connect

```
GANs + VAEs → Generative Modeling Foundation
                    ↓
            Transformers → Revolutionary Architecture
                    ↓
        BERT + RoBERTa → Pretraining Paradigm
                    ↓
              ViT → Cross-Modal Extension
                    ↓
        LLaMA + Mixtral → Efficient Scaling
                    ↓
             LoRA → Adaptation Layer
                    ↓
    RAG + RAG Survey → Knowledge Integration
                    ↓
           LightRAG → Production Systems
```

## 📖 Additional Resources

### Interactive Learning
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual guide to understanding Transformers
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) - Line-by-line implementation with explanations
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) - Interactive visualization

### Courses & Tutorials
- [Hugging Face Course](https://huggingface.co/course) - Practical NLP with Transformers
- [Andrej Karpathy's Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) - Build neural networks from scratch
- [Fast.ai Deep Learning Course](https://course.fast.ai/) - Practical deep learning

### Code & Implementation
- [Papers With Code](https://paperswithcode.com/) - Papers with implementation code
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - State-of-the-art NLP models
- [PyTorch Examples](https://github.com/pytorch/examples) - Official PyTorch examples

### Conceptual Understanding
- [Distill.pub](https://distill.pub/) - Clear explanations of ML concepts
- [Lil'Log](https://lilianweng.github.io/) - Deep dives into ML topics
- [Sebastian Ruder's Blog](https://www.ruder.io/) - NLP research insights

## 🗂️ Repository Structure

```
.
├── papers/              # PDF copies of papers (add your own)
│   ├── generative/      # GANs, VAEs, Diffusion
│   ├── transformers/    # Attention, BERT, RoBERTa, ViT
│   ├── llms/           # LLaMA, Mixtral, Surveys
│   └── rag/            # RAG papers
├── notes/              # Reading notes and summaries
│   ├── paper-summaries/
│   └── concept-maps/
├── implementations/    # Code implementations
│   ├── transformers/
│   ├── lora/
│   └── rag/
├── visuals/           # Diagrams and visualizations
└── README.md          # This file
```

## 🤝 Contributing

Found a broken link? Have suggestions for additional papers that fit this narrative? Open an issue or submit a pull request!

**Contribution ideas:**
- ✍️ Add reading notes or summaries
- 🎨 Create visualizations of key concepts
- 💻 Implement simplified versions of the models
- 📚 Suggest related papers that fit the progression
- 🔧 Add practical implementation guides
- 📝 Write tutorials connecting multiple papers

## 📄 License

This repository is MIT licensed. Note that individual papers are copyrighted by their respective authors and publishers.

## 🙏 Acknowledgments

Thanks to all the researchers whose groundbreaking work made this collection possible, and to the broader ML community for open research and knowledge sharing.

**Special thanks to:**
- The authors of all papers featured in this collection
- The open-source community for making implementations accessible
- ArXiv and Hugging Face for hosting research papers
- All contributors to this repository

---

<div align="center">

**[⭐ Star this repo](../../stargazers)** if you find it useful!

Made with ❤️ for the ML community

*Last updated: December 2024*

</div>
