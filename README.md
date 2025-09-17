# ChartEval: LLM-Driven Chart Generation Evaluation Using Scene Graph Parsing

A comprehensive chart evaluation system that compares generated chart images with ground truth using advanced scene graph parsing and LLM-driven analysis.

Demo Video: https://youtu.be/HcPuJaVO04s
Live Demo Link: https://d97c12eb37eaeba040.gradio.live

## Overview

ChartEval addresses a critical challenge in automated chart generation: **how do we reliably evaluate the quality of generated charts?** Current evaluation methods suffer from significant limitations:

- **Human evaluation** is costly and difficult to scale
- **Pixel-based metrics** (like SSIM) ignore data accuracy and unfairly penalize semantically equivalent charts
- **Data-centric measures** (like SCRM) overlook visual design quality
- **LLM-based evaluators** show concerning inconsistencies due to prompt sensitivity

**ChartEval's Solution:** Transform chart images into structured scene graphs and apply graph-based similarity measures for comprehensive quality assessment across visual similarity, semantic alignment, and data fidelity.

### Key Innovation

Instead of treating charts as mere images or data tables, ChartEval views charts as **visual scene graphs** where:
- Visual objects (data marks, legends, axes) become **nodes**
- Attributes (colors, sizes, positions) define **node properties**  
- Relationships (spatial arrangements, data mappings) become **edges**

## Key Features

### Comprehensive Evaluation Metrics
- **GraphBERT Score**: Semantic similarity between charts (F1, Precision, Recall)
- **Hallucination Rate**: Detection of spurious/incorrect information
- **Omission Rate**: Identification of missing critical elements
- **Graph Edit Distance**: Structural differences between charts

### Multi-LLM Support
- **Claude Sonnet 3.5**: Excellent detailed chart analysis and precise data extraction
- **GPT-4 Vision**: Strong vision capabilities with thorough analytical insights
- Easy switching between providers with unified interface

### Multiple Chart Types
- Line charts, Bar charts, Pie charts, Scatter plots
- 2D and 3D visualizations
- Support for complex multi-series data

### Detailed Human-Readable Analysis
- Executive summary with accuracy scores
- Specific examples of errors with chart element references
- Element-by-element comparison (titles, data, axes, visual design)
- Actionable recommendations for improvement
- Impact assessment for decision-making

### Web Interface
- User-friendly Gradio interface
- Pre-loaded example chart pairs
- Real-time evaluation with progress tracking
- Comprehensive results visualization

## Concept Diagram

```
┌─────────────────┐    ┌─────────────────┐
│   Ground Truth  │    │   Predicted     │
│      Chart      │    │     Chart       │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│ ChartSceneParse │    │ ChartSceneParse │
│   (LLM-based)   │    │   (LLM-based)   │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│  Scene Graph    │    │  Scene Graph    │
│   (Vega JSON)   │    │   (Vega JSON)   │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          └──────────┬───────────┘
                     ▼
          ┌─────────────────────┐
          │   Graph Comparison  │
          │                     │
          │ • GraphBERT Score   │
          │ • Hallucination     │
          │ • Omission Rate     │
          │ • Edit Distance     │
          └─────────────────────┘
```

## Installation

### Prerequisites
- Python 3.8+
- API key for Claude (Anthropic) or OpenAI GPT-4

### Quick Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-username/charteval.git
cd charteval
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up API keys** (choose one method):

   **Method A: Environment Variables**
   ```bash
   export CLAUDE_API_KEY="your-claude-api-key"
   export OPENAI_API_KEY="your-openai-api-key"
   ```

   **Method B: Direct Configuration**
   Edit the script and update:
   ```python
   CLAUDE_API_KEY = "your-claude-api-key"
   OPENAI_API_KEY = "your-openai-api-key"  
   ```

4. **Run the application**
```bash
python charteval_demo.py
```

The interface will be available at `http://localhost:7860`

## Requirements

```txt
gradio>=4.0.0
anthropic>=0.8.0
openai>=1.0.0
sentence-transformers>=2.2.0
networkx>=3.0
scikit-learn>=1.3.0
matplotlib>=3.6.0
pandas>=2.0.0
numpy>=1.24.0
Pillow>=9.0.0
```

### API Requirements
- **Claude API**: Get your key from console.anthropic.com
- **OpenAI API**: Get your key from platform.openai.com/api-keys

## Configuration

### LLM Provider Settings

The system supports different model configurations:

```python
# Claude Configuration
claude_config = {
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 4000,
    "temperature": 0.1
}

# GPT-4 Configuration  
gpt4_config = {
    "model": "gpt-4-vision-preview",
    "max_tokens": 4000,
    "temperature": 0.1
}
```

### Adding Custom Examples

Update the `EXAMPLE_CHART_PAIRS` dictionary:

```python
EXAMPLE_CHART_PAIRS = {
    "Your Example Name": {
        "ground_truth": "path/to/ground_truth.png",
        "predicted": "path/to/predicted.png", 
        "description": "Description of your chart example"
    }
}
```

## Usage

### Web Interface

1. **Select LLM Provider**: Choose between Claude or GPT-4
2. **Input Charts**: Either select a pre-loaded example OR upload your own charts
   - Chart 1: Ground truth (reference) chart
   - Chart 2: Predicted/generated chart to evaluate
3. **Run Evaluation**: Click "Evaluate Charts" 
4. **Review Results**: Get comprehensive metrics and detailed analysis

### Programmatic Usage

```python
from charteval import ChartEval

# Initialize evaluator
evaluator = ChartEval(
    llm_provider="Claude",
    api_key="your-api-key"
)

# Compare charts
bert_score, hall_score, omis_score, ged_score = evaluator.compare(
    chart1_path="ground_truth.png",
    chart2_path="predicted.png"
)

# Get detailed explanation
explanation = evaluator.generate_detailed_explanation(
    graph1, graph2, metrics, chart1_b64, chart2_b64
)

print(f"GraphBERT F1: {bert_score['f1']:.3f}")
print(f"Hallucination Rate: {hall_score['hallucination_rate']:.3f}")
print(f"Omission Rate: {omis_score['omission_rate']:.3f}")
```

## Metrics Explained

### GraphBERT Score
- **Purpose**: Measures semantic similarity between charts
- **Components**: 
  - Precision: How much of predicted chart matches ground truth
  - Recall: How much of ground truth is captured in predicted chart  
  - F1: Harmonic mean of precision and recall
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: >0.8 indicates strong semantic alignment

### Hallucination Rate  
- **Purpose**: Detects spurious/incorrect information in predicted chart
- **What it catches**: Extra data points, wrong labels, incorrect values
- **Range**: 0.0 to 1.0 (lower is better)
- **Interpretation**: <0.2 indicates minimal false information

### Omission Rate
- **Purpose**: Identifies missing critical elements
- **What it catches**: Missing data points, absent labels, incomplete information
- **Range**: 0.0 to 1.0 (lower is better)  
- **Interpretation**: <0.2 indicates comprehensive information coverage

### Graph Edit Distance (Normalized)
- **Purpose**: Measures structural differences between charts
- **What it measures**: Layout changes, component differences, design variations
- **Range**: 0.0 to 1.0 (lower is better)
- **Interpretation**: <0.3 indicates similar structure

## Key Results

ChartEval demonstrates **significantly stronger correlation with human judgments** compared to existing metrics:

| Metric | ChartCraft | ChartMimic | ChartX | Text2Chart31 |
|--------|------------|------------|---------|--------------|
| **ChartEval (Ours)** | **0.76** | **0.79** | **0.85** | **0.78** |
| GPT-Score | 0.25 | 0.27 | 0.33 | 0.25 |
| SSIM | 0.09 | 0.11 | 0.24 | 0.18 |
| SCRM | 0.13 | 0.15 | 0.29 | 0.19 |

*Pearson correlation coefficients with human quality ratings across 4K chart evaluations*

### Performance Highlights
- **38-203% improvement** over existing metrics
- **Consistent performance** across chart types (line, bar, pie, scatter)
- **Robust evaluation** across different LLM generators (GPT-4o, Claude, Qwen2.5-VL)
- **High inter-annotator agreement** (α = 0.74-0.85) in human evaluation

## Examples

### Example 1: High-Quality Chart Match

**Input:**
- Ground Truth: Line chart showing maternal mortality rate (2010-2013)
- Predicted: Nearly identical chart with minor styling differences

**ChartEval Results:**
```
GraphBERT F1: 0.94
Hallucination Rate: 0.02  
Omission Rate: 0.01
Normalized GED: 0.15

Assessment: Excellent semantic preservation with minimal visual differences
```

### Example 2: Data Hallucination Detection

**Input:**
- Ground Truth: Bar chart with 4 data points
- Predicted: Similar chart but with extra data point added

**ChartEval Results:**
```
GraphBERT F1: 0.73
Hallucination Rate: 0.25 ← Correctly identifies spurious data
Omission Rate: 0.03
Normalized GED: 0.31

Assessment: Structural similarity but significant data hallucination detected
```

### Example 3: Missing Information

**Input:** 
- Ground Truth: Pie chart with 6 segments and labels
- Predicted: Pie chart missing 2 segments and some labels

**ChartEval Results:**
```
GraphBERT F1: 0.61
Hallucination Rate: 0.08
Omission Rate: 0.35 ← Correctly identifies missing elements  
Normalized GED: 0.42

Assessment: Major information omissions significantly impact chart completeness
```

## Input Format Examples

### Supported Chart Types

**Line Charts**
```
- Time series data
- Multiple series
- Trend analysis
- Temporal patterns
```

**Bar Charts**
```  
- Categorical comparisons
- Horizontal/vertical orientation
- Grouped/stacked bars
- Statistical distributions
```

**Pie Charts**
```
- Percentage breakdowns
- Market share analysis  
- Categorical proportions
- Donut variations
```

**Scatter Plots**
```
- Correlation analysis
- Data point relationships
- Trend identification
- Multi-dimensional data
```

### Image Requirements
- **Format**: PNG, JPG, JPEG
- **Resolution**: High-resolution preferred (>800x600)
- **Quality**: Clear, readable text and labels
- **Content**: Single chart per image

## Limitations

### Current Limitations
1. **Image Quality Dependency**: Performance degrades with low-resolution images (<400x300)
2. **Complex Chart Support**: Limited support for highly complex multi-panel visualizations
3. **3D Chart Analysis**: Reduced accuracy for complex 3D surface plots
4. **Processing Time**: Analysis takes 60-90 seconds per chart pair due to LLM inference

### Future Improvements
- Fine-tuning VLMs for low-resolution chart images
- Enhanced 3D chart parsing capabilities  
- Faster inference through model optimization
- Support for additional chart types (sankey, treemap, etc.)

## Contributing

We welcome contributions! Please see our contributing guidelines.

### Development Setup
```bash
git clone https://github.com/your-username/charteval.git
cd charteval
pip install -e .
pip install -r requirements-dev.txt
```

### Running Tests
```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use ChartEval in your research, please cite our paper:

```bibtex
@article{goswami2024charteval,
  title={ChartEval: LLM-Driven Chart Generation Evaluation Using Scene Graph Parsing},
  author={Goswami, Kanika and Mathur, Puneet and Rossi, Ryan and Dernoncourt, Franck and Gupta, Vivek and Manocha, Dinesh},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## Support

- **Demo**: chartEval.ai
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

---

**Ready to evaluate your chart generation system?** Get started with our online demo or follow the installation guide above!
