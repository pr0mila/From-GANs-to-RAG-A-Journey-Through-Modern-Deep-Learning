## 📖 Overview

This repository presents a carefully structured reading path through 13 landmark papers that tell the story of modern deep learning. Starting with Generative Adversarial Networks and culminating in efficient Retrieval-Augmented Generation systems, this collection shows how each breakthrough built upon the previous one to shape today's AI landscape.

## 🎯 Why This Sequence?

Rather than presenting papers chronologically or by topic in isolation, this collection follows the **conceptual evolution** of ideas:

- **Generative foundations** → How neural networks learned to create
- **Attention mechanisms** → The architectural breakthrough that changed everything
- **Scale and pretraining** → Making models smarter through more data and compute
- **Cross-domain unification** → Extending transformers beyond text
- **Efficiency innovations** → Making powerful models practical
- **Knowledge augmentation** → Combining reasoning with retrieval

## 📚 The Papers

### Part I: Generative Foundations (2014-2013)

#### 1. [Generative Adversarial Networks](papers/gan.pdf)
**Goodfellow et al., 2014**

The paper that launched a revolution in generative modeling. Introduced the adversarial training paradigm where two networks compete—one generating fake data, the other trying to detect it.

> *Key Innovation*: Game-theoretic approach to learning data distributions

#### 2. [Auto-Encoding Variational Bayes (VAEs)](papers/vae.pdf)
**Kingma & Welling, 2013**

A probabilistic approach to generative modeling with stable training and interpretable latent spaces. Provides an alternative path to GANs with better theoretical foundations.

> *Key Innovation*: Variational inference for learning continuous latent representations

---

### Part II: The Transformer Revolution (2017-2019)

#### 3. [Attention Is All You Need](papers/attention.pdf)
**Vaswani et al., 2017**

The foundational paper that introduced the Transformer architecture. Eliminated recurrence entirely, replacing it with self-attention mechanisms that could be parallelized efficiently.

> *Key Innovation*: Self-attention as the core building block, enabling parallelization and long-range dependencies

#### 4. [BERT: Pre-training of Deep Bidirectional Transformers](papers/bert.pdf)
**Devlin et al., 2018**

Demonstrated the power of pretraining Transformers on massive text corpora. Introduced masked language modeling and set new benchmarks across NLP tasks.

> *Key Innovation*: Bidirectional pretraining with masked language modeling

#### 5. [RoBERTa: A Robustly Optimized BERT Pretraining Approach](papers/roberta.pdf)
**Liu et al., 2019**

Showed that BERT was undertrained. By scaling data, removing next sentence prediction, and training longer, achieved significant improvements without architectural changes.

> *Key Innovation*: Empirical demonstration that scale and training methodology matter as much as architecture

#### 6. [An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale (ViT)](papers/vit.pdf)
**Dosovitskiy et al., 2020**

Proved that Transformers could match or exceed CNNs in computer vision when given sufficient data. Unified the architecture across text and vision domains.

> *Key Innovation*: Direct application of Transformers to image patches, enabling cross-modal architecture unification

---

### Part III: Advanced Generative Models (2020)

#### 7. [Denoising Diffusion Probabilistic Models](papers/ddpm.pdf)
**Ho et al., 2020**

Reintroduced diffusion models with practical improvements that made them competitive with GANs. Became the foundation for modern image generation systems like Stable Diffusion and DALL-E.

> *Key Innovation*: Iterative denoising process that generates high-quality samples with stable training

---

### Part IV: Large Language Models Era (2023-2024)

#### 8. [Large Language Models: A Survey](papers/llm-survey.pdf)
**Zhao et al., 2023**

A comprehensive overview of the LLM landscape covering architecture, training, capabilities, and limitations. Essential context for understanding the current state of the field.

> *Key Innovation*: Systematic organization of LLM knowledge including emergent abilities and scaling laws

#### 9. [LLaMA: Open and Efficient Foundation Language Models](papers/llama.pdf)
**Touvron et al., 2023**

Demonstrated that high-performing language models could be trained efficiently with public data. Sparked the open-source LLM movement and enabled widespread experimentation.

> *Key Innovation*: Efficient training recipes and open release strategy that democratized LLM research

#### 10. [Mixtral of Experts](papers/mixtral.pdf)
**Jiang et al., 2024**

Introduced a practical sparse mixture-of-experts architecture that increases model capacity while keeping inference costs manageable through conditional computation.

> *Key Innovation*: Efficient scaling through sparse activation of expert sub-networks

---

### Part V: Efficient Adaptation & Knowledge Integration (2021-2024)

#### 11. [LoRA: Low-Rank Adaptation of Large Language Models](papers/lora.pdf)
**Hu et al., 2021**

Enabled practical fine-tuning of massive models by learning low-rank updates to weight matrices. Made domain adaptation accessible without full retraining.

> *Key Innovation*: Parameter-efficient fine-tuning through low-rank decomposition

#### 12. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](papers/rag.pdf)
**Lewis et al., 2020**

Combined neural retrieval with generation to ground language models in external knowledge. Reduced hallucinations and enabled access to up-to-date information.

> *Key Innovation*: Hybrid architecture merging parametric knowledge (LLM) with non-parametric knowledge (retrieval)

#### 13. [LightRAG: Simple and Fast Retrieval-Augmented Generation](papers/lightrag.pdf)
**Recent Work, 2024**

Streamlined RAG architecture focused on speed and simplicity. Represents the current direction toward efficient, production-ready knowledge-augmented systems.

> *Key Innovation*: Optimized retrieval pipeline with reduced latency and simpler deployment

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
2024  LightRAG ──────┘──> Knowledge Integration
```

## 🎓 How to Use This Repository

### For Students
Start from paper 1 and work sequentially. Each paper builds on concepts from previous ones. Take notes on how ideas evolve and connect.

### For Researchers
Use this as a reference for understanding how current techniques emerged. Jump to specific sections based on your interests, but review the foundational papers if unfamiliar.

### For Practitioners
Focus on papers 8-13 for practical modern techniques, but skim papers 3-6 to understand the architectural foundations you're building on.

## 💡 Key Themes Across Papers

- **Scale matters**: Larger models and more data consistently improve performance
- **Architecture unification**: Transformers work across modalities (text, images, etc.)
- **Efficiency is crucial**: Methods like LoRA and Mixtral make large models practical
- **External knowledge helps**: RAG systems reduce hallucinations and improve factuality
- **Simple ideas compound**: Each paper adds relatively simple innovations that combine powerfully

## 🛠️ Suggested Projects

After reading these papers, consider implementing:

1. **Miniature versions** of key architectures (small Transformer, simple VAE)
2. **Comparison studies** between techniques (GAN vs Diffusion for a specific task)
3. **Hybrid systems** combining ideas (LoRA-finetuned model with RAG)
4. **Efficiency experiments** measuring the tradeoffs between model size and performance

## 📖 Additional Resources

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Hugging Face Course](https://huggingface.co/course)
- [Andrej Karpathy's Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- [Papers With Code](https://paperswithcode.com/)

## 🤝 Contributing

Found a broken link? Have suggestions for additional papers that fit this narrative? Open an issue or submit a pull request!

## 📄 License

This repository is MIT licensed. Note that individual papers are copyrighted by their respective authors and publishers.

## 🙏 Acknowledgments

Thanks to all the researchers whose groundbreaking work made this collection possible, and to the broader ML community for open research and knowledge sharing.

---

<div align="center">

**[⭐ Star this repo](../../stargazers)** if you find it useful!

*Last updated: December 2024*

</div>
