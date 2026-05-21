# Task A Human Evaluation Checklist

Use this as a small judge-facing rubric for manual review of generated reviews.
Score each item from 1 (poor) to 5 (excellent).

| Criterion | What To Look For |
| --- | --- |
| Persona voice match | The review sounds like the target user's prior tone, length, specificity, and rating strictness. |
| Rating plausibility | The star rating is consistent with the review text and the user's historical rating behavior. |
| Item grounding | The review mentions plausible product attributes without hallucinating unsupported details. |
| Non-generic tone | The review avoids bland template language and includes user-like concrete judgment. |
| Locale/register fit | Nigerian users should sound contextually Nigerian where appropriate, without forced slang. |
| Price/value awareness | The review reflects price sensitivity, value-for-money, or authenticity concerns when relevant. |
| Behavioral consistency | The generated complaint/praise pattern matches the user's previous category behavior. |
| Safety and realism | The review is coherent, realistic, and free of irrelevant or policy-breaking content. |

Suggested paper note: report these human checks separately from automated ROUGE-L,
BERTScore, and RMSE, because they capture behavioral fidelity that automatic text
metrics only approximate.
