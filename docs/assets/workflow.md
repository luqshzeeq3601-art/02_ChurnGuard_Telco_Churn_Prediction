# ChurnGuard workflow image

Asset: [workflow.png](workflow.png). Generated with the built-in OpenAI image-generation tool on 6 October 2026 and reviewed against the project source. Style: modern light theme, warm white background, navy text and teal accents. This is an architecture illustration, not a screenshot or measured result chart.

## Meaning and source

The README supplies the accessible text summary and links to the owning technical/data/protocol documents. Model metrics remain sourced from the saved analytical artifacts. The diagram does not certify deployment or operational impact.

## Generation prompt

```text
Use case: infographic-diagram. Asset type: a GitHub README workflow image. Wide landscape 2:1, at least 1600 pixels wide. Modern light theme: warm white background, navy text, restrained teal accents, flat rounded cards, thin clean arrowheads, original small line icons, generous whitespace. No dark panels, gradients, logos, watermarks or fake screenshots. Large readable typography at a 900-pixel display width. Use EXACT labels given below. Explain implemented source architecture; do not invent metrics, successful deployment, automatic retraining or measured commercial impact.
Exact title: "ChurnGuard". Exact subtitle: "From customer records to a ranked retention list".
Main left-to-right path, with clear numbered stages:
"Customer records" -> "Validate + engineer features" -> "Calibrated Logistic Regression" -> "Risk + SHAP reasons" -> "API + Streamlit" -> "Scores + ranked export".
Below the calibrated-model card a short note says "Selected champion: M2". A small supporting card labelled "Trusted model bundle" connects only to the calibrated-model card.
The footer contains exactly two short notes: "Threshold selected from OOF predictions" and "Protected fields excluded; group gaps still audited".
Constraints: this is an inference workflow, not a causal claim that changing a feature prevents churn. The shared scoring layer supports both interfaces; CSV export is a dashboard capability. Do not depict LightGBM as the served champion. All arrows must follow the specified order.
```
