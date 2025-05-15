<div align="center">

# CheckEval: A Reliable LLM-as-a-Judge Framework for Evaluating Text Generation Using Checklists

📃 [Paper Link](https://arxiv.org/abs/2403.18771) (2025,arXiv preprint)

**Yukyung Lee**¹*, **Joonghoon Kim**²*, **Jaehee Kim**³^, **Hyowon Cho**⁴^, **Jaewook Kang**⁵, **Pilsung Kang**³†, **Najoung Kim**¹†

¹Boston University, ²SK Telecom, ³Seoul National University, ⁴KAIST, ⁵NAVER

*Equal contribution, ᵏCo-second authors, †Corresponding authors

</div>

## Abstract

Existing LLM-as-a-Judge approaches for evaluating text generation suffer from rating inconsistencies, with low agreement and high rating variance across different evaluator models. We attribute this to subjective evaluation criteria combined with Likert scale scoring in existing protocols. To address this issue, we introduce CheckEval, a checklist-based evaluation framework that improves rating reliability via decomposed binary questions. Through experiments with 12 evaluator models across multiple datasets, we first demonstrate that CheckEval strongly correlates with human judgments, improving the average correlation with human judgments by 0.10. More importantly, CheckEval dramatically improves the average agreement across evaluator models by 0.45 and reduces the score variance. CheckEval scores furthermore have the benefit of being more interpretable because it decomposes evaluation criteria into traceable binary decisions, allowing analyses of specific attributes driving quality judgments.

## Installation

```bash
git clone https://github.com/yourusername/CheckEval.git
cd CheckEval
pip install -r requirements.txt
```

## Project Structure

```
CheckEval/
├── src/                   # Source code
│   ├── inference_checkeval.py  # Main CheckEval inference script
│   ├── inference_geval.py      # G-Eval implementation for comparison
│   ├── aggregation.py          # Score aggregation utilities
│   ├── correlation.py          # Correlation analysis between methods
│   └── vllm_inference.sh       # Shell script for vLLM inference
├── prompt/                # LLM prompts
│   └── topical_chat_questions/ # Decomposed question templates
│       ├── coherence_seed.yaml       # Coherence evaluation questions
│       ├── coherence_diversification.yaml
│       ├── groundedness_elaboration.yaml
│       └── engagingness_elaboration.yaml
├── data/                  # Evaluation data and results
└── README.md              # This file
```

## Citation

```bibtex
@article{lee2025checkeval,
  title={Checkeval: A reliable llm-as-a-judge framework for evaluating text generation using checklists},
  author={Lee, Yukyung and Kim, Joonghoon and Kim, Jaehee and Cho, Hyowon and Kang, Pilsung and Kim, Najoung},
  journal={arXiv preprint arXiv:2403.18771},
  year={2025}
}
```

## Contact
* Yukyung Lee (ylee5@bu.edu)
* Joonghoon Kim (wndgns7686@gmail.com)

