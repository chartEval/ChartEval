import gradio as gr
import json
import os
import tempfile
import traceback
from typing import Dict, Any, Union, Optional, List, Tuple
import base64
import pandas as pd
import numpy as np
from PIL import Image
import io
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 🔑 CONFIGURATION - SET YOUR API KEYS HERE
# =============================================================================
# Replace these with your actual API keys
# Claude API key from: https://console.anthropic.com/
CLAUDE_API_KEY = ""

# OpenAI API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY = ""  # Add your OpenAI API key here

# ⚠️ SECURITY WARNING: Do not share this script with your API keys exposed!
# For production use, consider using environment variables instead:
# CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# =============================================================================

# Import the necessary libraries (install these if not already installed)
try:
    import anthropic
    import openai
    from sentence_transformers import SentenceTransformer
    import networkx as nx
    from sklearn.metrics.pairwise import cosine_similarity
    import matplotlib.pyplot as plt
    import mimetypes
    from collections import defaultdict
    import itertools
    
    # Example chart pairs - Replace these paths with your actual example images
    EXAMPLE_CHART_PAIRS = {
        "Example 1: Maternal Mortality": {
            "ground_truth": "examples/ex_1/ground_truth.png",
            "predicted": "examples/ex_1/output.png",
            "description": "Line chart showing maternal mortality rate over 4 years"
        },
        "Example 2: Main Cooking Fuel": {
            "ground_truth": "examples/ex_2/ground_truth.png", 
            "predicted": "examples/ex_2/output.png",
            "description": "Line chart showing main cooking fuel used by households"
        },
        "Example 3: Distribution of Website Users": {
            "ground_truth": "examples/ex_3/ground_truth.png",
            "predicted": "examples/ex_3/output.png", 
            "description": "Pie chart showing distribution of website users by websites"
        },
        "Example 4: Relation between Latitude and Daylight": {
            "ground_truth": "examples/ex_4/ground_truth.png",
            "predicted": "examples/ex_4/output.png",
            "description": "Scatter chart showing relation between latitude and daylight duration"
        },
        "Example 5: Market Share": {
            "ground_truth": "examples/ex_5/ground_truth.png",
            "predicted": "examples/ex_5/output.png",
            "description": "Bar chart showing market share of top streaming platforms"
        },
        "Example 6: Roaming Wisps": {
            "ground_truth": "examples/ex_6/ground_truth.png",
            "predicted": "examples/ex_6/output.png",
            "description": "3D chart showing roaming wisps of celestial aurora"
        },
        "Example 7: Function Chart": {
            "ground_truth": "examples/ex_7/ground_truth.png", 
            "predicted": "examples/ex_7/output.png",
            "description": "Function chart of a polynomial function"
        },
        "Example 8: Target vs Prediction": {
            "ground_truth": "examples/ex_8/ground_truth.png",
            "predicted": "examples/ex_8/output.png", 
            "description": "Scatter plot showing target vs prediction"
        },
        "Example 9: Saudi Arabia's Re-export in 1991": {
            "ground_truth": "examples/ex_9/ground_truth.png",
            "predicted": "examples/ex_9/output.png",
            "description": "Line chart showing Saudi Arabia's re-export in 1991"
        },
        "Example 10: Glucose vs Fructose": {
            "ground_truth": "examples/ex_10/ground_truth.png",
            "predicted": "examples/ex_10/output.png",
            "description": "Bar chart showing glucose vs fructose in different fruits"
        }
    }
    
    class ChartEval:
        
        def __init__(self, llm_provider="Claude", api_key=None, model_config=None):
            """
            Initialize ChartEval with configurable LLM provider
            
            Args:
                llm_provider: LLM provider name ("GPT-3.5", "GPT-4", "Claude", etc.)
                api_key: API key for the LLM service
                model_config: Additional model configuration parameters
            """
            self.llm_provider = llm_provider
            self.api_key = api_key or os.getenv('LLM_API_KEY')
            self.model_config = model_config or {}
            
            # Initialize LLM client based on provider
            self._init_llm_client()
            
            # Initialize sentence transformer for GraphBERT scoring
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Warning: Could not load sentence transformer: {e}")
                self.sentence_model = None
        
        def _init_llm_client(self):
            """Initialize the appropriate LLM client based on provider"""
            if self.llm_provider.startswith("GPT"):
                try:
                    self.llm_client = openai.OpenAI(api_key=self.api_key)
                except ImportError:
                    raise ImportError("OpenAI package required for GPT models. Install with: pip install openai")
            
            elif self.llm_provider == "Claude":
                try:
                    self.llm_client = anthropic.Anthropic(api_key=self.api_key)
                except ImportError:
                    raise ImportError("Anthropic package required for Claude. Install with: pip install anthropic")
            
            else:
                # Generic/custom LLM - user needs to provide their own client
                self.llm_client = None
        
        def compare(self, chart1, chart2):
            """
            Compare two charts and return various similarity scores
            
            Args:
                chart1: First chart (ground truth) - image path, base64, image data, or graph dict
                chart2: Second chart (predicted) - image path, base64, image data, or graph dict
            
            Returns:
                Tuple of scores: (graphBertScore, hallucinationScore, omissionScore, graphEditDist)
            """
            # Handle different input types - if already graph dicts, use directly
            if isinstance(chart1, dict) and 'chart_type' in chart1:
                graph1 = chart1
            else:
                vega_dict1 = self.chartToVega(chart1)
                graph1 = self.vegaToGraph(vega_dict1)
            
            if isinstance(chart2, dict) and 'chart_type' in chart2:
                graph2 = chart2
            else:
                vega_dict2 = self.chartToVega(chart2)
                graph2 = self.vegaToGraph(vega_dict2)
            
            # Calculate all evaluation metrics
            graphBertScore = self.calculate_graphBert_score(graph1, graph2)
            hallucinationScore = self.calculate_hallucination_score(graph1, graph2)
            omissionScore = self.calculate_omission_score(graph1, graph2)
            graphEditDist = self.calculate_GED_score(graph1, graph2)
            
            return graphBertScore, hallucinationScore, omissionScore, graphEditDist
        
        def generate_detailed_explanation(self, graph1, graph2, metrics, chart1_image=None, chart2_image=None):
            """
            Generate a detailed human-readable explanation of chart comparison results
            
            Args:
                graph1: Ground truth graph structure
                graph2: Predicted graph structure
                metrics: Dictionary containing calculated metrics
                chart1_image: Base64 encoded ground truth chart image (optional)
                chart2_image: Base64 encoded predicted chart image (optional)
            
            Returns:
                String containing detailed explanation
            """
            try:
                # Create comprehensive prompt for detailed analysis
                prompt = self._create_detailed_analysis_prompt(graph1, graph2, metrics)
                
                # Prepare images if available
                image_inputs = []
                if chart1_image and chart2_image:
                    image_inputs = [
                        {"type": "image", "data": chart1_image, "label": "Ground Truth Chart"},
                        {"type": "image", "data": chart2_image, "label": "Predicted Chart"}
                    ]
                
                # Call LLM for detailed analysis
                if image_inputs:
                    explanation = self._call_llm_with_images_for_explanation(prompt, image_inputs)
                else:
                    explanation = self._call_llm_text_only(prompt)
                
                return explanation
                
            except Exception as e:
                return f"Error generating detailed explanation: {str(e)}"
        
        def _create_detailed_analysis_prompt(self, graph1, graph2, metrics):
            """Create a comprehensive prompt for detailed chart analysis"""
            
            # Extract key information from graphs
            gt_info = self._extract_graph_summary(graph1, "Ground Truth")
            pred_info = self._extract_graph_summary(graph2, "Predicted")
            
            # Format metrics for inclusion
            bert_score = metrics.get('bert_score', {})
            hall_score = metrics.get('hallucination_score', {})
            omis_score = metrics.get('omission_score', {})
            ged_score = metrics.get('ged_score', {})
            
            prompt = f"""
You are an expert data analyst tasked with providing a comprehensive, human-readable comparison between two charts. Your analysis should be accessible to non-technical stakeholders while being detailed and actionable.

## CHART INFORMATION:

### Ground Truth Chart (Reference):
{gt_info}

### Predicted Chart (Generated):
{pred_info}

## COMPUTED METRICS:
- GraphBERT F1 Score: {bert_score.get('f1', 0):.3f} (Semantic similarity - higher is better)
- Hallucination Rate: {hall_score.get('hallucination_rate', 0):.3f} (False information - lower is better) 
- Omission Rate: {omis_score.get('omission_rate', 0):.3f} (Missing information - lower is better)
- Normalized Graph Edit Distance: {ged_score.get('normalized_ged', 0):.3f} (Structural difference - lower is better)

## DETAILED ISSUES FOUND:

### Hallucinations (False Information):
{self._format_issues_list(hall_score.get('hallucinations', []))}

### Omissions (Missing Information):
{self._format_issues_list(omis_score.get('omissions', []))}

## TASK:
Provide a detailed analysis in the following structure. Use specific examples from the charts and reference actual data points, labels, and values wherever possible.

## REQUIRED OUTPUT FORMAT:

### 📊 EXECUTIVE SUMMARY
[2-3 sentence high-level assessment of how well the predicted chart matches the ground truth]

### 🎯 OVERALL PERFORMANCE ASSESSMENT
**Accuracy Score: [X/10]**
[Brief justification based on metrics]

**Key Strengths:**
- [Specific examples of what the predicted chart got right]
- [Reference actual data points, labels, axis titles, etc.]

**Critical Issues:**
- [Specific examples of major problems with concrete details]
- [Point to exact discrepancies in data values, missing elements, etc.]

### 🔍 DETAILED BREAKDOWN BY CHART ELEMENTS

**Title and Labels:**
- Ground Truth: [Specific title/labels from GT chart]
- Predicted: [Specific title/labels from predicted chart]
- Assessment: [What matches, what differs, impact on understanding]

**Data Accuracy:**
- [Compare specific data points with exact values]
- [Highlight any missing or incorrect data series]
- [Discuss trends and patterns - are they preserved?]

**Visual Design:**
- [Compare chart types, colors, layout]
- [Assess if visual encoding effectively represents the data]

### ⚠️ SPECIFIC ERRORS WITH EXAMPLES

**Data Errors:**
- [List each incorrect data point with: "Ground truth shows X, but predicted shows Y"]
- [Quantify the magnitude of errors where applicable]

**Missing Elements:**
- [List each missing element: "The predicted chart is missing [specific element] which shows [importance]"]

**Added Elements (Hallucinations):**
- [List each incorrectly added element: "The predicted chart incorrectly includes [specific element] which doesn't exist in the ground truth"]

### 💡 ACTIONABLE RECOMMENDATIONS

**Immediate Fixes:**
1. [Specific correction needed with exact details]
2. [Another specific fix with concrete steps]

**Improvement Suggestions:**
1. [Suggestion for better data accuracy]
2. [Suggestion for better visual representation]

**Quality Assurance:**
- [Recommend specific validation checks]
- [Suggest verification steps for similar charts]

### 📈 IMPACT ASSESSMENT
[Explain how the identified issues would affect:]
- Data interpretation by end users
- Decision-making based on this chart
- Overall credibility and trust

### 🏆 CONCLUSION
[Final verdict with specific confidence level and key takeaway message]

## INSTRUCTIONS:
1. Be specific - always reference actual data points, labels, and values from the charts
2. Use concrete examples rather than general statements
3. Explain the business/analytical impact of each issue
4. Provide actionable recommendations with clear steps
5. Use a tone that's professional but accessible to non-technical audiences
6. Focus on the most impactful differences first
7. If charts are very similar, still provide constructive analysis
8. Include specific numerical references wherever possible
"""
            
            return prompt
        
        def _extract_graph_summary(self, graph, label):
            """Extract key information from graph structure for prompt"""
            if not isinstance(graph, dict):
                return f"{label}: Unable to parse graph structure"
            
            summary = [f"{label}:"]
            summary.append(f"- Chart Type: {graph.get('chart_type', 'Unknown')}")
            summary.append(f"- Title: '{graph.get('title', 'No title')}'")
            
            # Extract axis information
            axes = graph.get('axes', {})
            x_axis = axes.get('x_axis', {})
            y_axis = axes.get('y_axis', {})
            
            if x_axis.get('title'):
                summary.append(f"- X-axis: {x_axis['title']}")
            if y_axis.get('title'):
                summary.append(f"- Y-axis: {y_axis['title']}")
            
            # Extract data points summary
            data_points = graph.get('data_points', [])
            summary.append(f"- Data Points: {len(data_points)} points")
            
            if graph.get('chart_type') == 'pie':
                # For pie charts, show segment breakdown
                segments = []
                for point in data_points[:5]:  # Show first 5 segments
                    if 'label' in point and 'value' in point:
                        segments.append(f"{point['label']}: {point['value']}%")
                if segments:
                    summary.append(f"- Segments: {', '.join(segments)}")
                    if len(data_points) > 5:
                        summary.append(f"  (... and {len(data_points) - 5} more)")
            else:
                # For other charts, show data range
                if data_points:
                    x_values = [p.get('data_x') for p in data_points if p.get('data_x') is not None]
                    y_values = [p.get('data_y') for p in data_points if p.get('data_y') is not None]
                    
                    if x_values and y_values:
                        summary.append(f"- X range: {min(x_values)} to {max(x_values)}")
                        summary.append(f"- Y range: {min(y_values)} to {max(y_values)}")
            
            # Add semantic content if available
            semantic = graph.get('semantic_content', {})
            if semantic.get('data_trend'):
                summary.append(f"- Data Trend: {semantic['data_trend']}")
            
            return '\n'.join(summary)
        
        def _format_issues_list(self, issues):
            """Format list of issues for prompt"""
            if not issues:
                return "None detected"
            
            formatted = []
            for i, issue in enumerate(issues[:10], 1):  # Show first 10 issues
                issue_type = issue.get('type', 'Unknown')
                content = issue.get('content', 'Unknown')
                reason = issue.get('reason', 'No reason provided')
                formatted.append(f"{i}. {issue_type}: {content} ({reason})")
            
            if len(issues) > 10:
                formatted.append(f"... and {len(issues) - 10} more issues")
            
            return '\n'.join(formatted) if formatted else "None detected"
        
        def _call_llm_with_images_for_explanation(self, prompt, image_inputs):
            """Call LLM with both text prompt and images for detailed explanation"""
            try:
                if self.llm_provider == "Claude":
                    # Prepare message content with images and text
                    content = []
                    
                    # Add images first
                    for img_input in image_inputs:
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",  # Assume JPEG for simplicity
                                "data": img_input["data"]
                            }
                        })
                    
                    # Add text prompt
                    content.append({
                        "type": "text",
                        "text": prompt
                    })
                    
                    message = self.llm_client.messages.create(
                        model=self.model_config.get("model", "claude-3-5-sonnet-20241022"),
                        max_tokens=self.model_config.get("max_tokens", 4000),
                        temperature=self.model_config.get("temperature", 0.1),
                        messages=[{
                            "role": "user",
                            "content": content
                        }]
                    )
                    return message.content[0].text
                    
                elif self.llm_provider.startswith("GPT"):
                    # For GPT-4 with vision
                    content = [{"type": "text", "text": prompt}]
                    
                    # Add images
                    for img_input in image_inputs:
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_input['data']}"
                            }
                        })
                    
                    response = self.llm_client.chat.completions.create(
                        model=self.model_config.get("model", "gpt-4-vision-preview"),
                        messages=[{
                            "role": "user",
                            "content": content
                        }],
                        max_tokens=self.model_config.get("max_tokens", 4000),
                        temperature=self.model_config.get("temperature", 0.1)
                    )
                    return response.choices[0].message.content
                else:
                    return "Detailed explanation with images not supported for this LLM provider"
                    
            except Exception as e:
                return f"Error generating explanation with images: {str(e)}"
        
        def _call_llm_text_only(self, prompt):
            """Call LLM with text-only prompt for explanation"""
            try:
                if self.llm_provider == "Claude":
                    message = self.llm_client.messages.create(
                        model=self.model_config.get("model", "claude-3-5-sonnet-20241022"),
                        max_tokens=self.model_config.get("max_tokens", 4000),
                        temperature=self.model_config.get("temperature", 0.1),
                        messages=[{
                            "role": "user",
                            "content": prompt
                        }]
                    )
                    return message.content[0].text
                    
                elif self.llm_provider.startswith("GPT"):
                    response = self.llm_client.chat.completions.create(
                        model=self.model_config.get("model", "gpt-4"),
                        messages=[{
                            "role": "user",
                            "content": prompt
                        }],
                        max_tokens=self.model_config.get("max_tokens", 4000),
                        temperature=self.model_config.get("temperature", 0.1)
                    )
                    return response.choices[0].message.content
                else:
                    return "Detailed explanation not supported for this LLM provider"
                    
            except Exception as e:
                return f"Error generating text-only explanation: {str(e)}"
        
        def calculate_graphBert_score(self, graph1, graph2):
            """Calculate GraphBERT similarity score between two chart graphs."""
            if self.sentence_model is None:
                return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'error': 'Sentence model not available'}
            
            # Extract semantic elements from both graphs as sentences
            sentences1 = self._graph_to_sentences(graph1)
            sentences2 = self._graph_to_sentences(graph2)
            
            if not sentences1 or not sentences2:
                return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            
            # Get embeddings for all sentences
            embeddings1 = self.sentence_model.encode(sentences1)
            embeddings2 = self.sentence_model.encode(sentences2)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(embeddings1, embeddings2)
            
            # Calculate BERT-style precision and recall
            recall_scores = []
            for i in range(len(sentences1)):
                max_sim = np.max(similarity_matrix[i])
                recall_scores.append(max_sim)
            
            precision_scores = []
            for j in range(len(sentences2)):
                max_sim = np.max(similarity_matrix[:, j])
                precision_scores.append(max_sim)
            
            # Calculate final metrics
            recall = np.mean(recall_scores)
            precision = np.mean(precision_scores)
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            
            return {
                'precision': float(precision),
                'recall': float(recall), 
                'f1': float(f1),
                'sentences1_count': len(sentences1),
                'sentences2_count': len(sentences2)
            }
        
        def _graph_to_sentences(self, graph):
            """Convert graph elements to natural language sentences for BERTScore comparison."""
            sentences = []
            
            if not isinstance(graph, dict):
                return sentences
                
            # Add title as sentence
            title = graph.get('title', '')
            if title:
                sentences.append(f"Chart title: {title}")
            
            # Add chart type
            chart_type = graph.get('chart_type', '')
            if chart_type:
                sentences.append(f"Chart type: {chart_type}")
            
            # Handle different chart types differently
            if chart_type == 'pie':
                # For pie charts, focus on segments and their values
                data_points = graph.get('data_points', [])
                for point in data_points:
                    if 'label' in point and 'value' in point:
                        sentences.append(f"{point['label']} accounts for {point['value']}% of the total")
                    elif 'description' in point and point['description']:
                        sentences.append(point['description'])
                
                # Add total validation sentence
                total_percentage = sum(point.get('value', 0) for point in data_points if 'value' in point)
                if abs(total_percentage - 100) < 1:  # Allow small rounding errors
                    sentences.append("All segments sum to 100 percent")
                
            else:
                # For line/bar/scatter charts, use axis information
                axes = graph.get('axes', {})
                x_axis = axes.get('x_axis', {})
                y_axis = axes.get('y_axis', {})
                
                if x_axis.get('title'):
                    sentences.append(f"X-axis represents: {x_axis['title']}")
                if y_axis.get('title'):
                    sentences.append(f"Y-axis represents: {y_axis['title']}")
                
                # Add data points as sentences
                data_points = graph.get('data_points', [])
                for point in data_points:
                    if 'description' in point and point['description']:
                        sentences.append(point['description'])
                    elif 'data_x' in point and 'data_y' in point:
                        sentences.append(f"Data point at x={point['data_x']}, y={point['data_y']}")
            
            # Add semantic content
            semantic = graph.get('semantic_content', {})
            if semantic.get('data_trend'):
                sentences.append(f"Data trend is {semantic['data_trend']}")
            
            if semantic.get('temporal_extent'):
                temp = semantic['temporal_extent']
                if 'start_year' in temp and 'end_year' in temp:
                    sentences.append(f"Time period from {temp['start_year']} to {temp['end_year']}")
            
            # Add visual properties
            visual = graph.get('visual_properties', {})
            if visual.get('stroke'):
                sentences.append(f"Line color: {visual['stroke']}")
            
            return sentences
        
        def calculate_hallucination_score(self, graph1, graph2):
            """Calculate hallucination score - elements present in predicted graph but absent in ground truth."""
            # Extract comparable elements from both graphs
            elements1 = self._extract_graph_elements(graph1)
            elements2 = self._extract_graph_elements(graph2)
            
            # Find elements in graph2 that are not in graph1 (hallucinations)
            hallucinations = []
            
            for element_type, element_data in elements2.items():
                ground_truth_data = elements1.get(element_type, set())
                
                if isinstance(element_data, set):
                    hallucinated_items = element_data - ground_truth_data
                    for item in hallucinated_items:
                        hallucinations.append({
                            'type': element_type,
                            'content': item,
                            'reason': f'{element_type} not present in ground truth'
                        })
                elif isinstance(element_data, (str, int, float)):
                    if element_data != elements1.get(element_type):
                        hallucinations.append({
                            'type': element_type,
                            'content': element_data,
                            'expected': elements1.get(element_type),
                            'reason': f'{element_type} differs from ground truth'
                        })
            
            # Calculate hallucination rate
            total_elements = sum(len(v) if isinstance(v, set) else 1 for v in elements2.values())
            hallucination_count = len(hallucinations)
            
            hallucination_rate = hallucination_count / max(total_elements, 1)
            
            return {
                'hallucination_rate': float(hallucination_rate),
                'hallucination_count': hallucination_count,
                'total_predicted_elements': total_elements,
                'hallucinations': hallucinations
            }
        
        def calculate_omission_score(self, graph1, graph2):
            """Calculate omission score - elements present in ground truth but missing in predicted graph."""
            # Extract comparable elements from both graphs
            elements1 = self._extract_graph_elements(graph1)
            elements2 = self._extract_graph_elements(graph2)
            
            # Find elements in graph1 that are not in graph2 (omissions)
            omissions = []
            
            for element_type, element_data in elements1.items():
                predicted_data = elements2.get(element_type, set())
                
                if isinstance(element_data, set):
                    omitted_items = element_data - predicted_data
                    for item in omitted_items:
                        omissions.append({
                            'type': element_type,
                            'content': item,
                            'reason': f'{element_type} missing from prediction'
                        })
                elif isinstance(element_data, (str, int, float)):
                    if element_data != elements2.get(element_type):
                        omissions.append({
                            'type': element_type,
                            'content': element_data,
                            'predicted': elements2.get(element_type),
                            'reason': f'{element_type} not correctly predicted'
                        })
            
            # Calculate omission rate
            total_elements = sum(len(v) if isinstance(v, set) else 1 for v in elements1.values())
            omission_count = len(omissions)
            
            omission_rate = omission_count / max(total_elements, 1)
            
            return {
                'omission_rate': float(omission_rate),
                'omission_count': omission_count,
                'total_ground_truth_elements': total_elements,
                'omissions': omissions
            }
        
        def _extract_graph_elements(self, graph):
            """Extract comparable elements from a chart graph for hallucination/omission analysis."""
            elements = {}
            
            if not isinstance(graph, dict):
                return elements
            
            # Extract title
            if graph.get('title'):
                elements['title'] = graph['title']
            
            # Extract chart type
            if graph.get('chart_type'):
                elements['chart_type'] = graph['chart_type']
            
            chart_type = graph.get('chart_type', '')
            
            if chart_type == 'pie':
                # For pie charts, extract segment data as label-value pairs
                pie_segments = set()
                data_points = graph.get('data_points', [])
                
                for point in data_points:
                    if 'label' in point and 'value' in point:
                        # Round percentage values to 1 decimal place to handle minor variations
                        label = point['label'].strip()
                        value = round(float(point['value']), 1)
                        pie_segments.add((label, value))
                    elif 'description' in point:
                        # Try to parse label and value from description
                        parsed_segment = self._parse_pie_segment_from_description(point['description'])
                        if parsed_segment:
                            pie_segments.add(parsed_segment)
                
                elements['pie_segments'] = pie_segments
                
            else:
                # For other chart types, use existing logic
                # Extract axis titles
                axes = graph.get('axes', {})
                if axes.get('x_axis', {}).get('title'):
                    elements['x_axis_title'] = axes['x_axis']['title']
                if axes.get('y_axis', {}).get('title'):
                    elements['y_axis_title'] = axes['y_axis']['title']
                
                # Extract data points (rounded to avoid floating point precision issues)
                data_points = set()
                for point in graph.get('data_points', []):
                    if 'data_x' in point and 'data_y' in point:
                        x_val = round(point['data_x'], 2) if isinstance(point['data_x'], (int, float)) else point['data_x']
                        y_val = round(point['data_y'], 2) if isinstance(point['data_y'], (int, float)) else point['data_y']
                        data_points.add((x_val, y_val))
                elements['data_points'] = data_points
                
                # Extract axis labels
                x_labels = set()
                y_labels = set()
                
                if 'x_axis' in axes and 'labels' in axes['x_axis']:
                    for label in axes['x_axis']['labels']:
                        if isinstance(label, dict) and 'text' in label:
                            x_labels.add(label['text'])
                
                if 'y_axis' in axes and 'labels' in axes['y_axis']:
                    for label in axes['y_axis']['labels']:
                        if isinstance(label, dict) and 'text' in label:
                            y_labels.add(label['text'])
                
                if x_labels:
                    elements['x_axis_labels'] = x_labels
                if y_labels:
                    elements['y_axis_labels'] = y_labels
            
            # Extract semantic information (common for all chart types)
            semantic = graph.get('semantic_content', {})
            if semantic.get('data_trend'):
                elements['data_trend'] = semantic['data_trend']
            
            return elements
        
        def _parse_pie_segment_from_description(self, description):
            """Parse pie chart segment information from description text"""
            import re
            
            # Look for patterns like "Label accounts for X% of the total" or "Label: X%"
            patterns = [
                r'(.+?)\s+accounts\s+for\s+([\d.]+)%',
                r'(.+?):\s*([\d.]+)%',
                r'(.+?)\s+([\d.]+)%',
                r'(.+?)\s*-\s*([\d.]+)%'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, description, re.IGNORECASE)
                if match:
                    label = match.group(1).strip()
                    try:
                        value = round(float(match.group(2)), 1)
                        return (label, value)
                    except ValueError:
                        continue
            
            return None
        
        def calculate_GED_score(self, graph1, graph2):
            """Calculate Graph Edit Distance (GED) score between two chart graphs."""
            # Convert graphs to NetworkX format for GED calculation
            nx_graph1 = self._convert_to_networkx(graph1, "ground_truth")
            nx_graph2 = self._convert_to_networkx(graph2, "predicted")
            
            # Calculate edit operations
            edit_ops = self._calculate_edit_operations(graph1, graph2)
            
            # Simple GED approximation based on element differences
            ged_distance = (
                edit_ops['node_insertions'] + 
                edit_ops['node_deletions'] + 
                edit_ops['node_substitutions'] + 
                edit_ops['edge_insertions'] + 
                edit_ops['edge_deletions'] + 
                edit_ops['edge_substitutions']
            )
            
            # Normalize by the maximum possible operations
            max_nodes = max(nx_graph1.number_of_nodes(), nx_graph2.number_of_nodes())
            max_edges = max(nx_graph1.number_of_edges(), nx_graph2.number_of_edges())
            max_operations = max_nodes + max_edges
            
            normalized_ged = ged_distance / max(max_operations, 1)
            
            return {
                'ged_distance': ged_distance,
                'normalized_ged': float(normalized_ged),
                'edit_operations': edit_ops,
                'graph1_nodes': nx_graph1.number_of_nodes(),
                'graph1_edges': nx_graph1.number_of_edges(),
                'graph2_nodes': nx_graph2.number_of_nodes(),
                'graph2_edges': nx_graph2.number_of_edges()
            }
        
        def _convert_to_networkx(self, graph, graph_name="graph"):
            """Convert chart graph to NetworkX graph for GED calculation."""
            G = nx.DiGraph()
            
            if not isinstance(graph, dict):
                return G
                
            # Add nodes for different graph elements
            node_id = 0
            
            # Add title node
            if graph.get('title'):
                G.add_node(f"title_{node_id}", type="title", content=graph['title'])
                node_id += 1
            
            # Add chart type node
            if graph.get('chart_type'):
                G.add_node(f"chart_type_{node_id}", type="chart_type", content=graph['chart_type'])
                node_id += 1
            
            chart_type = graph.get('chart_type', '')
            
            if chart_type == 'pie':
                # For pie charts, create nodes for each segment
                pie_center_node = f"pie_center_{node_id}"
                G.add_node(pie_center_node, type="pie_center")
                node_id += 1
                
                data_points = graph.get('data_points', [])
                segment_nodes = []
                
                for i, point in enumerate(data_points):
                    segment_node = f"pie_segment_{node_id}"
                    G.add_node(segment_node, type="pie_segment", 
                              label=point.get('label', f'Segment {i+1}'),
                              value=point.get('value', 0),
                              percentage=point.get('value', 0))
                    segment_nodes.append(segment_node)
                    node_id += 1
                    
                    # Connect segment to pie center
                    G.add_edge(pie_center_node, segment_node, type="contains_segment")
                
                # Connect adjacent segments (circular structure)
                for i in range(len(segment_nodes)):
                    next_i = (i + 1) % len(segment_nodes)
                    G.add_edge(segment_nodes[i], segment_nodes[next_i], type="adjacent_segment")
                    
            else:
                # For line/bar/scatter charts, use existing logic
                # Add axis nodes
                axes = graph.get('axes', {})
                x_axis_node = None
                y_axis_node = None
                
                if axes.get('x_axis', {}).get('title'):
                    x_axis_node = f"x_axis_{node_id}"
                    G.add_node(x_axis_node, type="x_axis", title=axes['x_axis']['title'])
                    node_id += 1
                    
                if axes.get('y_axis', {}).get('title'):
                    y_axis_node = f"y_axis_{node_id}"  
                    G.add_node(y_axis_node, type="y_axis", title=axes['y_axis']['title'])
                    node_id += 1
                
                # Add data point nodes
                data_nodes = []
                for i, point in enumerate(graph.get('data_points', [])):
                    point_node = f"data_point_{node_id}"
                    G.add_node(point_node, type="data_point", 
                              x=point.get('data_x'), y=point.get('data_y'),
                              description=point.get('description', ''))
                    data_nodes.append(point_node)
                    node_id += 1
                    
                    # Connect data points to axes
                    if x_axis_node:
                        G.add_edge(point_node, x_axis_node, type="uses_x_axis")
                    if y_axis_node:
                        G.add_edge(point_node, y_axis_node, type="uses_y_axis")
                
                # Connect consecutive data points (for line charts)
                if graph.get('chart_type') == 'line' and len(data_nodes) > 1:
                    for i in range(len(data_nodes) - 1):
                        G.add_edge(data_nodes[i], data_nodes[i+1], type="sequence")
            
            return G
        
        def _calculate_edit_operations(self, graph1, graph2):
            """Calculate the edit operations needed to transform graph1 into graph2."""
            elements1 = self._extract_graph_elements(graph1)
            elements2 = self._extract_graph_elements(graph2)
            
            operations = {
                'node_insertions': 0,
                'node_deletions': 0, 
                'node_substitutions': 0,
                'edge_insertions': 0,
                'edge_deletions': 0,
                'edge_substitutions': 0
            }
            
            # Compare each element type
            all_keys = set(elements1.keys()) | set(elements2.keys())
            
            for key in all_keys:
                val1 = elements1.get(key)
                val2 = elements2.get(key)
                
                if val1 is None and val2 is not None:
                    # Insertion
                    if isinstance(val2, set):
                        operations['node_insertions'] += len(val2)
                    else:
                        operations['node_insertions'] += 1
                elif val1 is not None and val2 is None:
                    # Deletion
                    if isinstance(val1, set):
                        operations['node_deletions'] += len(val1)
                    else:
                        operations['node_deletions'] += 1
                elif val1 != val2:
                    # Substitution
                    if isinstance(val1, set) and isinstance(val2, set):
                        # Calculate set differences
                        inserted = val2 - val1
                        deleted = val1 - val2
                        operations['node_insertions'] += len(inserted)
                        operations['node_deletions'] += len(deleted)
                    else:
                        operations['node_substitutions'] += 1
            
            return operations
        
        def chartToVega(self, chart_input):
            """Convert chart image to Vega-Lite specification using LLM"""
            try:
                # Prepare image for LLM
                image_data, media_type = self._prepare_image(chart_input)
                
                # Create prompt for LLM
                prompt = self._create_chart_analysis_prompt()
                
                # Get LLM response
                llm_response = self._call_llm(prompt, image_data, media_type)
                
                # Validate LLM response
                if llm_response is None or not llm_response.strip():
                    raise ValueError("LLM returned empty or None response")
                
                # Parse LLM response to Vega-Lite format
                vega_spec = self._parse_llm_response_to_vega(llm_response)
                
                return vega_spec
                
            except Exception as e:
                print(f"Error in chartToVega: {str(e)}")
                # Return a safe fallback structure
                return {
                    "marktype": "group",
                    "name": "root",
                    "role": "frame",
                    "interactive": True,
                    "clip": False,
                    "items": [],
                    "zindex": 0,
                    "_chart_analysis_error": str(e)
                }
        
        def _detect_image_format(self, file_path):
            """Detect image format from file path or content"""
            # First try to get from file extension
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type and mime_type.startswith('image/'):
                return mime_type
            
            # Fallback: try to detect from file content
            try:
                with Image.open(file_path) as img:
                    format_map = {
                        'JPEG': 'image/jpeg',
                        'PNG': 'image/png',
                        'GIF': 'image/gif',
                        'WebP': 'image/webp',
                        'BMP': 'image/bmp'
                    }
                    return format_map.get(img.format, 'image/jpeg')
            except Exception:
                # Default fallback
                return 'image/jpeg'
        
        def _prepare_image(self, chart_input):
            """Prepare image data for LLM input"""
            if isinstance(chart_input, str):
                if os.path.isfile(chart_input):
                    # File path
                    media_type = self._detect_image_format(chart_input)
                    with open(chart_input, "rb") as image_file:
                        image_bytes = image_file.read()
                    return base64.b64encode(image_bytes).decode('utf-8'), media_type
                elif chart_input.startswith('data:image'):
                    # Data URL - extract media type
                    header, data = chart_input.split(',', 1)
                    media_type = header.split(':')[1].split(';')[0]
                    return data, media_type
                elif len(chart_input) > 100:
                    # Assume it's base64 - default to JPEG
                    return chart_input, 'image/jpeg'
                else:
                    raise ValueError("Invalid image input: not a valid file path or base64 string")
            
            elif isinstance(chart_input, bytes):
                # Raw bytes - try to detect format
                try:
                    img = Image.open(io.BytesIO(chart_input))
                    format_map = {
                        'JPEG': 'image/jpeg',
                        'PNG': 'image/png',
                        'GIF': 'image/gif',
                        'WebP': 'image/webp',
                        'BMP': 'image/bmp'
                    }
                    media_type = format_map.get(img.format, 'image/jpeg')
                except Exception:
                    media_type = 'image/jpeg'
                
                return base64.b64encode(chart_input).decode('utf-8'), media_type
            
            else:
                raise ValueError("Chart input must be file path, base64 string, or bytes")
        
        def _create_chart_analysis_prompt(self):
            """Create a comprehensive prompt for chart analysis that handles multiple chart types"""
            return """
            Analyze this chart image and extract ALL data points, axis information, and visual elements with PRECISE values. 
            The chart could be a line chart, bar chart, pie chart, scatter plot, or other visualization type.
            
            Please provide a detailed analysis in the following JSON format:

            {
                "title": "Exact chart title text",
                "description": "Brief description of what the chart shows",
                "chart_type": "line|bar|scatter|pie|area|donut|etc.",
                "data": [
                    // For line/bar/scatter charts:
                    {"x": exact_value, "y": exact_value, "label": "optional_label", "description": "point description"},
                    
                    // For pie/donut charts:
                    {"label": "segment_name", "value": percentage_value, "description": "segment description"},
                    
                    // Include ALL data points/segments visible in the chart
                    ...
                ],
                "x_axis": {
                    "title": "Exact X-axis title (for non-pie charts)",
                    "type": "quantitative|temporal|ordinal|nominal",
                    "domain": [min_value, max_value],
                    "ticks": [list_of_tick_values],
                    "tick_labels": ["label1", "label2", ...]
                },
                "y_axis": {
                    "title": "Exact Y-axis title (for non-pie charts)", 
                    "type": "quantitative|temporal|ordinal|nominal",
                    "domain": [min_value, max_value],
                    "ticks": [list_of_tick_values],
                    "tick_labels": ["label1", "label2", ...]
                },
                "chart_dimensions": {
                    "width": estimated_width,
                    "height": estimated_height
                },
                "styling": {
                    "primary_color": "#color",
                    "line_width": width_in_pixels,
                    "grid_lines": true/false,
                    "background_color": "#color"
                }
            }

            CRITICAL REQUIREMENTS:
            - Extract EVERY visible data point with exact values
            - For PIE CHARTS: Extract each segment's label and percentage value (ensure they sum to ~100%)
            - For LINE/BAR CHARTS: Extract exact X and Y coordinates for every data point
            - Read ALL axis labels and tick values precisely
            - Include the complete chart title exactly as shown
            - For temporal data (years/dates), extract exact years/dates
            - For numerical axes, read exact tick values and ranges
            - Include descriptive text for each data point/segment
            - Identify chart type correctly (pie, line, bar, scatter, etc.)
            """
        
        def _call_llm(self, prompt, image_data, media_type):
            """Call the configured LLM with the prompt and image"""
            try:
                if self.llm_provider.startswith("GPT"):
                    return self._call_openai_llm(prompt, image_data)
                elif self.llm_provider == "Claude":
                    return self._call_claude_llm(prompt, image_data, media_type)
                else:
                    raise NotImplementedError(f"LLM provider {self.llm_provider} not implemented")
            except Exception as e:
                print(f"LLM call failed: {str(e)}")
                return None
        
        def _call_openai_llm(self, prompt, image_data):
            """Call OpenAI GPT with vision capabilities"""
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model_config.get("model", "gpt-4-vision-preview"),
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=self.model_config.get("max_tokens", 2000),
                    temperature=self.model_config.get("temperature", 0.1)
                )
                
                # Validate response structure
                if not response or not response.choices or not response.choices[0].message:
                    raise Exception("Invalid response structure from OpenAI API")
                    
                content = response.choices[0].message.content
                if not content:
                    raise Exception("Empty content in OpenAI API response")
                    
                return content
                
            except Exception as e:
                raise Exception(f"OpenAI API call failed: {str(e)}")
        
        def _call_claude_llm(self, prompt, image_data, media_type):
            """Call Anthropic Claude with vision capabilities"""
            try:
                message = self.llm_client.messages.create(
                    model=self.model_config.get("model", "claude-3-5-sonnet-20241022"),
                    max_tokens=self.model_config.get("max_tokens", 2000),
                    temperature=self.model_config.get("temperature", 0.1),
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": image_data
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                )
                
                # Validate response structure
                if not message or not message.content or not message.content[0]:
                    raise Exception("Invalid response structure from Claude API")
                    
                content = message.content[0].text
                if not content:
                    raise Exception("Empty content in Claude API response")
                    
                return content
                
            except Exception as e:
                raise Exception(f"Claude API call failed: {str(e)}")
        
        def _parse_llm_response_to_vega(self, llm_response):
            """Parse LLM response and convert to full Vega specification (not Vega-Lite)"""
            try:
                # Validate input
                if not llm_response or not llm_response.strip():
                    raise ValueError("Empty or None LLM response")
                
                # Try to extract JSON from the response
                json_start = llm_response.find('{')
                json_end = llm_response.rfind('}') + 1
                
                if json_start != -1 and json_end != -1:
                    json_str = llm_response[json_start:json_end]
                    chart_data = json.loads(json_str)
                else:
                    raise ValueError("No valid JSON found in LLM response")
                
                # Convert to full Vega format
                vega_spec = self._build_vega_specification(chart_data)
                
                return vega_spec
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Error parsing LLM response: {str(e)}")
                # Fallback: create basic structure if parsing fails
                return {
                    "marktype": "group",
                    "name": "root",
                    "role": "frame",
                    "interactive": True,
                    "clip": False,
                    "items": [],
                    "zindex": 0,
                    "_parse_error": f"Error parsing LLM response: {str(e)}"
                }
        
        def _build_vega_specification(self, chart_data):
            """Build complete Vega specification from chart data"""
            
            # Validate input
            if not isinstance(chart_data, dict):
                chart_data = {}
            
            # Extract basic info with safe defaults
            title = chart_data.get("title", "Chart")
            data_points = chart_data.get("data", [])
            if not isinstance(data_points, list):
                data_points = []
                
            x_axis = chart_data.get("x_axis", {})
            if not isinstance(x_axis, dict):
                x_axis = {}
                
            y_axis = chart_data.get("y_axis", {})
            if not isinstance(y_axis, dict):
                y_axis = {}
                
            chart_type = self._normalize_chart_type(chart_data.get("chart_type", "line"))
            
            dimensions = chart_data.get("chart_dimensions", {"width": 200, "height": 200})
            if not isinstance(dimensions, dict):
                dimensions = {"width": 200, "height": 200}
                
            styling = chart_data.get("styling", {})
            if not isinstance(styling, dict):
                styling = {}
            
            width = dimensions.get("width", 200)
            height = dimensions.get("height", 200)
            
            # Build main frame
            vega_spec = {
                "marktype": "group",
                "name": "root", 
                "role": "frame",
                "interactive": True,
                "clip": False,
                "items": [],
                "zindex": 0
            }
            
            # Main chart area
            main_group = {
                "items": [],
                "x": 0,
                "y": 0,
                "width": width,
                "height": height,
                "fill": "transparent",
                "stroke": "#ddd"
            }
            
            try:
                if chart_type == "pie":
                    # Add pie chart marks
                    main_group["items"].append(self._create_pie_marks(data_points, width, height, styling))
                else:
                    # Add grid lines and axes for non-pie charts
                    axes_items = self._create_axes(x_axis, y_axis, width, height)
                    if axes_items:
                        main_group["items"].extend(axes_items)
                    
                    # Add data marks based on chart type
                    if chart_type == "line":
                        main_group["items"].append(self._create_line_marks(data_points, x_axis, y_axis, width, height, styling))
                    elif chart_type in ["scatter", "point"]:
                        main_group["items"].append(self._create_point_marks(data_points, x_axis, y_axis, width, height, styling))
                    elif chart_type == "bar":
                        main_group["items"].append(self._create_bar_marks(data_points, x_axis, y_axis, width, height, styling))
                    # Add other chart type handling here as needed
                
                # Add title
                if title:
                    main_group["items"].append(self._create_title(title, width))
                    
            except Exception as e:
                print(f"Error creating chart marks: {str(e)}")
                # Add error information to the structure
                main_group["items"].append({
                    "marktype": "text",
                    "role": "error",
                    "text": f"Error creating chart: {str(e)}",
                    "x": width / 2,
                    "y": height / 2
                })
            
            vega_spec["items"] = [main_group]
            
            return vega_spec
        
        def _normalize_chart_type(self, chart_type):
            """Normalize chart type names returned by the LLM"""
            if not chart_type:
                return "line"
                
            ct = str(chart_type).strip().lower()
            aliases = {
                "donut": "pie",
                "doughnut": "pie",
                "bubble": "scatter",
                "scatterplot": "scatter",
                "points": "point",
                "line chart": "line",
                "bar chart": "bar",
            }
            return aliases.get(ct, ct if ct in ["pie", "line", "bar", "scatter", "point", "area"] else "line")
        
        def _create_point_marks(self, data_points, x_axis, y_axis, width, height, styling):
            """Create point marks (scatter plot) from data points with robust type handling"""
            if not data_points or not isinstance(data_points, list):
                return {"marktype": "point", "items": []}
            
            numeric_points = []
            for point in data_points:
                if not isinstance(point, dict):
                    continue
                    
                try:
                    x_val = float(point.get("x", point.get("data_x", 0)))
                except (ValueError, TypeError):
                    x_val = point.get("x", point.get("data_x"))
                try:
                    y_val = float(point.get("y", point.get("data_y", 0)))
                except (ValueError, TypeError):
                    y_val = point.get("y", point.get("data_y"))
                    
                if isinstance(x_val, (int, float)) and isinstance(y_val, (int, float)):
                    numeric_points.append({
                        "x": float(x_val),
                        "y": float(y_val),
                        "description": point.get("description", f"X: {x_val}, Y: {y_val}")
                    })
                    
            if not numeric_points:
                return {"marktype": "point", "items": []}
            
            x_values = [p["x"] for p in numeric_points]
            y_values = [p["y"] for p in numeric_points]
            x_domain = x_axis.get("domain", [min(x_values), max(x_values)])
            y_domain = y_axis.get("domain", [min(y_values), max(y_values)])
            
            try:
                x_domain = [float(x_domain[0]), float(x_domain[1])]
            except (Exception, IndexError):
                x_domain = [min(x_values), max(x_values)]
                
            try:
                y_domain = [float(y_domain[0]), float(y_domain[1])]
            except (Exception, IndexError):
                y_domain = [min(y_values), max(y_values)]
            
            point_color = styling.get("point_color", "#1f77b4")
            point_size = styling.get("point_size", 40)
            
            items = []
            for p in numeric_points:
                items.append({
                    "x": self._scale_value(p["x"], x_domain, [0, width]),
                    "y": self._scale_value(p["y"], y_domain, [height, 0]),
                    "fill": point_color,
                    "size": point_size,
                    "description": p["description"],
                })
                
            return {
                "marktype": "point",
                "name": "marks",
                "role": "mark",
                "interactive": True,
                "clip": False,
                "items": items,
                "zindex": 0
            }
        
        def _create_bar_marks(self, data_points, x_axis, y_axis, width, height, styling):
            """Create bar marks from data points, supporting categorical X"""
            if not data_points or not isinstance(data_points, list):
                return {"marktype": "bar", "items": []}
            
            # Extract categories and values
            categories = []
            values = []
            
            for point in data_points:
                if not isinstance(point, dict):
                    continue
                    
                categories.append(str(point.get("label", point.get("x", ""))))
                try:
                    values.append(float(point.get("y", point.get("value", 0))))
                except (ValueError, TypeError):
                    continue
                    
            if not categories or not values:
                return {"marktype": "bar", "items": []}
            
            # Map categories to index positions
            unique_cats = list(dict.fromkeys(categories))
            x_positions = {cat: idx for idx, cat in enumerate(unique_cats)}
            
            y_domain = y_axis.get("domain", [0, max(values)])
            try:
                y_domain = [float(y_domain[0]), float(y_domain[1])]
            except Exception:
                y_domain = [0, max(values)]
            
            bar_width = max(5, width / max(1, len(unique_cats)) * 0.6)
            items = []
            
            for cat, val in zip(categories, values):
                x_center = self._scale_value(x_positions[cat], [0, max(1, len(unique_cats) - 1)], [0, width])
                y_top = self._scale_value(val, y_domain, [height, 0])
                items.append({
                    "x": x_center - bar_width / 2,
                    "y": y_top,
                    "width": bar_width,
                    "height": height - y_top,
                    "fill": styling.get("bar_color", "#4CAF50"),
                    "description": f"{cat}: {val}",
                })
                
            return {
                "marktype": "bar",
                "name": "marks",
                "role": "mark",
                "interactive": True,
                "clip": False,
                "items": items,
                "zindex": 0
            }
        
        def _create_pie_marks(self, data_points, width, height, styling):
            """Create pie chart marks from data points"""
            if not data_points or not isinstance(data_points, list):
                return {"marktype": "arc", "items": []}
            
            # Extract values and labels
            segments = []
            for point in data_points:
                if not isinstance(point, dict):
                    continue
                    
                if 'label' in point and 'value' in point:
                    try:
                        value = float(point['value'])
                        segments.append({
                            'label': str(point['label']),
                            'value': value,
                            'description': point.get('description', f"{point['label']}: {point['value']}%")
                        })
                    except (ValueError, TypeError):
                        continue
            
            if not segments:
                return {"marktype": "arc", "items": []}
            
            # Calculate angles for pie segments
            total_value = sum(seg['value'] for seg in segments)
            if total_value == 0:
                return {"marktype": "arc", "items": []}
            
            center_x = width / 2
            center_y = height / 2
            radius = min(width, height) / 3
            
            pie_items = []
            current_angle = -90  # Start from top
            
            for segment in segments:
                angle_size = (segment['value'] / total_value) * 360
                
                pie_items.append({
                    "x": center_x,
                    "y": center_y,
                    "startAngle": current_angle,
                    "endAngle": current_angle + angle_size,
                    "innerRadius": 0,
                    "outerRadius": radius,
                    "fill": styling.get("primary_color", "#4CAF50"),
                    "stroke": "#ffffff",
                    "strokeWidth": 2,
                    "label": segment['label'],
                    "value": segment['value'],
                    "description": segment['description']
                })
                
                current_angle += angle_size
            
            return {
                "marktype": "arc",
                "name": "pie_marks",
                "role": "mark", 
                "interactive": True,
                "clip": False,
                "items": pie_items,
                "zindex": 0
            }
        
        def _create_axes(self, x_axis, y_axis, width, height):
            """Create axis groups with grids, ticks, and labels"""
            axes = []
            
            try:
                # X-axis grid
                x_grid = self._create_x_grid(x_axis, width, height)
                if x_grid:
                    axes.append(x_grid)
                
                # Y-axis grid
                y_grid = self._create_y_grid(y_axis, width, height)
                if y_grid:
                    axes.append(y_grid)
                
                # X-axis
                x_axis_group = self._create_x_axis(x_axis, width, height)
                if x_axis_group:
                    axes.append(x_axis_group)
                
                # Y-axis
                y_axis_group = self._create_y_axis(y_axis, width, height)
                if y_axis_group:
                    axes.append(y_axis_group)
                    
            except Exception as e:
                print(f"Error creating axes: {str(e)}")
            
            return axes
        
        def _create_x_grid(self, x_axis, width, height):
            """Create X-axis grid lines with robust type handling"""
            if not isinstance(x_axis, dict):
                return None
                
            ticks = x_axis.get("ticks", [])
            if not ticks or not isinstance(ticks, list):
                return None
            
            # Convert ticks to numeric values, filter out non-numeric ones
            numeric_ticks = []
            for tick in ticks:
                try:
                    numeric_ticks.append(float(tick))
                except (ValueError, TypeError):
                    # Skip non-numeric ticks for grid creation
                    continue
            
            if not numeric_ticks:
                return None
                
            domain = x_axis.get("domain", [min(numeric_ticks), max(numeric_ticks)])
            
            # Ensure domain values are numeric
            try:
                domain = [float(domain[0]), float(domain[1])]
            except (ValueError, TypeError, IndexError):
                domain = [min(numeric_ticks), max(numeric_ticks)]
            
            grid_items = []
            
            for tick in numeric_ticks:
                x_pos = self._scale_value(tick, domain, [0, width])
                grid_items.append({
                    "x": x_pos,
                    "y": -height,
                    "opacity": 1,
                    "stroke": "#ddd",
                    "strokeWidth": 0.2,
                    "y2": 0
                })
            
            return {
                "marktype": "group",
                "role": "axis",
                "interactive": False,
                "clip": False,
                "items": [{
                    "items": [{
                        "marktype": "rule",
                        "role": "axis-grid",
                        "interactive": False,
                        "clip": False,
                        "items": grid_items,
                        "zindex": 0
                    }],
                    "x": 0.5,
                    "y": height + 0.5,
                    "orient": "bottom"
                }],
                "zindex": 0,
                "aria": False
            }
        
        def _create_y_grid(self, y_axis, width, height):
            """Create Y-axis grid lines with robust type handling"""
            if not isinstance(y_axis, dict):
                return None
                
            ticks = y_axis.get("ticks", [])
            if not ticks or not isinstance(ticks, list):
                return None
            
            # Convert ticks to numeric values, filter out non-numeric ones
            numeric_ticks = []
            for tick in ticks:
                try:
                    numeric_ticks.append(float(tick))
                except (ValueError, TypeError):
                    # Skip non-numeric ticks for grid creation
                    continue
            
            if not numeric_ticks:
                return None
                
            domain = y_axis.get("domain", [min(numeric_ticks), max(numeric_ticks)])
            
            # Ensure domain values are numeric
            try:
                domain = [float(domain[0]), float(domain[1])]
            except (ValueError, TypeError, IndexError):
                domain = [min(numeric_ticks), max(numeric_ticks)]
            
            grid_items = []
            
            for tick in numeric_ticks:
                y_pos = self._scale_value(tick, domain, [height, 0])
                grid_items.append({
                    "x": 0,
                    "y": y_pos,
                    "opacity": 1,
                    "stroke": "#ddd",
                    "strokeWidth": 0.2,
                    "x2": width
                })
            
            return {
                "marktype": "group",
                "role": "axis",
                "interactive": False,
                "clip": False,
                "items": [{
                    "items": [{
                        "marktype": "rule",
                        "role": "axis-grid",
                        "interactive": False,
                        "clip": False,
                        "items": grid_items,
                        "zindex": 0
                    }],
                    "x": 0.5,
                    "y": 0.5,
                    "orient": "left"
                }],
                "zindex": 0,
                "aria": False
            }
        
        def _create_x_axis(self, x_axis, width, height):
            """Create X-axis with ticks and labels with robust type handling"""
            if not isinstance(x_axis, dict):
                return None
                
            ticks = x_axis.get("ticks", [])
            tick_labels = x_axis.get("tick_labels", [str(t) for t in ticks] if ticks else [])
            title = x_axis.get("title", "")
            
            if not ticks or not isinstance(ticks, list):
                return None
            
            # Convert ticks to numeric values where possible
            numeric_ticks = []
            valid_labels = []
            
            for i, tick in enumerate(ticks):
                try:
                    numeric_tick = float(tick)
                    numeric_ticks.append(numeric_tick)
                    # Use corresponding label if available, otherwise convert tick to string
                    if i < len(tick_labels):
                        valid_labels.append(str(tick_labels[i]))
                    else:
                        valid_labels.append(str(tick))
                except (ValueError, TypeError):
                    # For non-numeric ticks, use position-based approximation
                    numeric_ticks.append(i)
                    valid_labels.append(str(tick) if i < len(tick_labels) else str(tick))
            
            if not numeric_ticks:
                return None
                
            domain = x_axis.get("domain", [min(numeric_ticks), max(numeric_ticks)])
            
            # Ensure domain values are numeric
            try:
                domain = [float(domain[0]), float(domain[1])]
            except (ValueError, TypeError, IndexError):
                domain = [min(numeric_ticks), max(numeric_ticks)]
            
            # Create simplified axis representation
            return {
                "marktype": "group",
                "role": "axis",
                "items": [],
                "domain": domain,
                "ticks": numeric_ticks,
                "labels": valid_labels,
                "title": title
            }
        
        def _create_y_axis(self, y_axis, width, height):
            """Create Y-axis with ticks and labels with robust type handling"""
            if not isinstance(y_axis, dict):
                return None
                
            ticks = y_axis.get("ticks", [])
            tick_labels = y_axis.get("tick_labels", [str(t) for t in ticks] if ticks else [])
            title = y_axis.get("title", "")
            
            if not ticks or not isinstance(ticks, list):
                return None
            
            # Convert ticks to numeric values where possible
            numeric_ticks = []
            valid_labels = []
            
            for i, tick in enumerate(ticks):
                try:
                    numeric_tick = float(tick)
                    numeric_ticks.append(numeric_tick)
                    # Use corresponding label if available, otherwise convert tick to string
                    if i < len(tick_labels):
                        valid_labels.append(str(tick_labels[i]))
                    else:
                        valid_labels.append(str(tick))
                except (ValueError, TypeError):
                    # For non-numeric ticks, use position-based approximation
                    numeric_ticks.append(i)
                    valid_labels.append(str(tick) if i < len(tick_labels) else str(tick))
            
            if not numeric_ticks:
                return None
                
            domain = y_axis.get("domain", [min(numeric_ticks), max(numeric_ticks)])
            
            # Ensure domain values are numeric
            try:
                domain = [float(domain[0]), float(domain[1])]
            except (ValueError, TypeError, IndexError):
                domain = [min(numeric_ticks), max(numeric_ticks)]
            
            # Create simplified axis representation
            return {
                "marktype": "group",
                "role": "axis",
                "items": [],
                "domain": domain,
                "ticks": numeric_ticks,
                "labels": valid_labels,
                "title": title
            }
        
        def _create_line_marks(self, data_points, x_axis, y_axis, width, height, styling):
            """Create line marks from data points with robust type handling"""
            if not data_points or not isinstance(data_points, list):
                return {"marktype": "line", "items": []}
            
            # Extract and convert data points to numeric values where possible
            numeric_points = []
            for point in data_points:
                if not isinstance(point, dict):
                    continue
                    
                try:
                    x_val = float(point.get("x", 0))
                    y_val = float(point.get("y", 0))
                    numeric_points.append({
                        "x": x_val,
                        "y": y_val,
                        "description": point.get("description", f"X: {x_val}, Y: {y_val}")
                    })
                except (ValueError, TypeError):
                    # Skip points that can't be converted to numeric
                    continue
            
            if not numeric_points:
                return {"marktype": "line", "items": []}
            
            # Determine domains from numeric points
            x_values = [p["x"] for p in numeric_points]
            y_values = [p["y"] for p in numeric_points]
            
            x_domain = x_axis.get("domain", [min(x_values), max(x_values)])
            y_domain = y_axis.get("domain", [min(y_values), max(y_values)])
            
            # Ensure domains are numeric
            try:
                x_domain = [float(x_domain[0]), float(x_domain[1])]
            except (ValueError, TypeError, IndexError):
                x_domain = [min(x_values), max(x_values)]
                
            try:
                y_domain = [float(y_domain[0]), float(y_domain[1])]
            except (ValueError, TypeError, IndexError):
                y_domain = [min(y_values), max(y_values)]
            
            line_color = styling.get("line_color", "#c4c4c4")
            line_width = styling.get("line_width", 2)
            
            line_items = []
            
            for point in numeric_points:
                x_pos = self._scale_value(point["x"], x_domain, [0, width])
                y_pos = self._scale_value(point["y"], y_domain, [height, 0])
                
                line_items.append({
                    "x": x_pos,
                    "y": y_pos,
                    "stroke": line_color,
                    "strokeWidth": line_width,
                    "defined": True,
                    "description": point["description"]
                })
            
            return {
                "marktype": "line",
                "name": "marks",
                "role": "mark",
                "interactive": True,
                "clip": False,
                "items": line_items,
                "zindex": 0
            }
        
        def _create_title(self, title_text, width):
            """Create title group"""
            if not title_text:
                return {"marktype": "group", "role": "title", "content": ""}
                
            # Handle multi-line titles
            if isinstance(title_text, str) and len(title_text) > 60:
                # Try to split long titles into multiple lines
                words = title_text.split()
                lines = []
                current_line = []
                
                for word in words:
                    if len(' '.join(current_line + [word])) > 40:
                        if current_line:
                            lines.append(' '.join(current_line))
                            current_line = [word]
                        else:
                            lines.append(word)
                    else:
                        current_line.append(word)
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                title_content = lines
            else:
                title_content = str(title_text)
            
            return {
                "marktype": "group",
                "role": "title",
                "content": title_content
            }
        
        def _scale_value(self, value, domain, range_vals):
            """Scale a value from domain to range with robust type conversion"""
            try:
                # Convert all values to float for mathematical operations
                value = float(value) if value is not None else 0.0
                domain_0 = float(domain[0]) if domain[0] is not None else 0.0
                domain_1 = float(domain[1]) if domain[1] is not None else 1.0
                
                # Avoid division by zero
                if domain_1 == domain_0:
                    return range_vals[0]
                
                ratio = (value - domain_0) / (domain_1 - domain_0)
                return range_vals[0] + ratio * (range_vals[1] - range_vals[0])
                
            except (ValueError, TypeError, ZeroDivisionError) as e:
                # Fallback: return middle of range if conversion fails
                return (range_vals[0] + range_vals[1]) / 2
        
        def vegaToGraph(self, vega_dict):
            """Convert Vega specification to graph representation for comparison"""
            # Validate input
            if not vega_dict or not isinstance(vega_dict, dict):
                return {
                    'chart_type': 'unknown',
                    'data_points': [],
                    'axes': {'x_axis': {}, 'y_axis': {}},
                    'title': '',
                    'visual_properties': {},
                    'error': 'Invalid or empty Vega specification'
                }
            
            if 'items' not in vega_dict:
                return {
                    'chart_type': 'unknown',
                    'data_points': [],
                    'axes': {'x_axis': {}, 'y_axis': {}},
                    'title': '',
                    'visual_properties': {},
                    'error': 'Missing items in Vega specification'
                }
            
            # Initialize graph structure
            graph = {
                'chart_type': 'unknown',
                'data_points': [],
                'axes': {
                    'x_axis': {},
                    'y_axis': {}
                },
                'title': '',
                'visual_properties': {},
                'structural_elements': [],
                'semantic_content': {}
            }
            
            try:
                # Get main chart group
                main_items = vega_dict.get('items', [])
                if not main_items or not isinstance(main_items, list):
                    return graph
                    
                chart_group = main_items[0]
                if not isinstance(chart_group, dict) or 'items' not in chart_group:
                    return graph
                    
                chart_items = chart_group.get('items', [])
                if not isinstance(chart_items, list):
                    chart_items = []
                
                # Extract chart dimensions
                graph['visual_properties']['width'] = chart_group.get('width', 0)
                graph['visual_properties']['height'] = chart_group.get('height', 0)
                
                # Parse different components
                for item in chart_items:
                    if not isinstance(item, dict):
                        continue
                        
                    role = item.get('role', '')
                    marktype = item.get('marktype', '')
                    
                    if role == 'title':
                        graph['title'] = self._extract_title(item)
                    elif role == 'axis':
                        self._extract_axis_info(item, graph)
                    elif marktype == 'arc':
                        # Handle pie chart arcs
                        self._extract_pie_marks(item, graph)
                        graph['chart_type'] = 'pie'
                    elif role == 'mark' or marktype in ['line', 'bar', 'point', 'area']:
                        self._extract_data_marks(item, graph)
                    
                    # Track structural elements
                    graph['structural_elements'].append({
                        'type': marktype,
                        'role': role,
                        'interactive': item.get('interactive', False)
                    })
                
                # Determine chart type from marks if not already set
                if graph['chart_type'] == 'unknown':
                    graph['chart_type'] = self._determine_chart_type(graph)
                
                # Extract semantic content
                graph['semantic_content'] = self._extract_semantic_content(graph)
                
            except Exception as e:
                graph['error'] = f"Error parsing Vega specification: {str(e)}"
            
            return graph
        
        def _extract_pie_marks(self, pie_item, graph):
            """Extract pie chart segments from pie marks"""
            try:
                pie_marks = pie_item.get('items', [])
                if not isinstance(pie_marks, list):
                    pie_marks = []
                
                for mark in pie_marks:
                    if not isinstance(mark, dict):
                        continue
                        
                    data_point = {
                        'label': mark.get('label', ''),
                        'value': mark.get('value', 0),
                        'startAngle': mark.get('startAngle', 0),
                        'endAngle': mark.get('endAngle', 0),
                        'description': mark.get('description', ''),
                        'mark_type': 'arc'
                    }
                    
                    graph['data_points'].append(data_point)
                
                # Store visual properties from first mark
                if pie_marks:
                    first_mark = pie_marks[0]
                    if isinstance(first_mark, dict):
                        graph['visual_properties'].update({
                            'fill': first_mark.get('fill', ''),
                            'stroke': first_mark.get('stroke', ''),
                            'strokeWidth': first_mark.get('strokeWidth', 0),
                            'innerRadius': first_mark.get('innerRadius', 0),
                            'outerRadius': first_mark.get('outerRadius', 0)
                        })
            except Exception as e:
                print(f"Error extracting pie marks: {str(e)}")
        
        def _extract_title(self, title_item):
            """Extract title text from title item"""
            try:
                if 'content' in title_item:
                    content = title_item['content']
                    if isinstance(content, list):
                        return ' '.join(str(item) for item in content)
                    return str(content)
                
                title_groups = title_item.get('items', [])
                if not isinstance(title_groups, list):
                    return ''
                    
                for group in title_groups:
                    if not isinstance(group, dict):
                        continue
                        
                    text_items = group.get('items', [])
                    if not isinstance(text_items, list):
                        continue
                        
                    for text_item in text_items:
                        if not isinstance(text_item, dict):
                            continue
                            
                        if text_item.get('role') == 'title-text':
                            text_content = text_item.get('items', [])
                            if text_content and isinstance(text_content, list):
                                title_text = text_content[0].get('text', '') if text_content[0] else ''
                                # Handle multi-line titles
                                if isinstance(title_text, list):
                                    return ' '.join(str(item) for item in title_text)
                                return str(title_text)
            except Exception as e:
                print(f"Error extracting title: {str(e)}")
            return ''

        def _extract_axis_info(self, axis_item, graph):
            """Extract axis information from axis item"""
            try:
                # Handle simplified axis representation
                if 'domain' in axis_item and 'ticks' in axis_item:
                    # Determine if this is x or y axis based on position or other indicators
                    # For now, we'll need to make an assumption or add more logic
                    axis_key = 'x_axis'  # Default assumption
                    
                    axis_info = graph['axes'][axis_key]
                    axis_info['domain'] = axis_item.get('domain', [])
                    axis_info['ticks'] = axis_item.get('ticks', [])
                    axis_info['labels'] = [{'text': str(label)} for label in axis_item.get('labels', [])]
                    axis_info['title'] = axis_item.get('title', '')
                    return
                
                axis_groups = axis_item.get('items', [])
                if not isinstance(axis_groups, list):
                    return
                    
                for group in axis_groups:
                    if not isinstance(group, dict):
                        continue
                        
                    orient = group.get('orient', '')
                    axis_components = group.get('items', [])
                    if not isinstance(axis_components, list):
                        continue
                    
                    axis_key = 'x_axis' if orient == 'bottom' else 'y_axis'
                    axis_info = graph['axes'][axis_key]
                    
                    for component in axis_components:
                        if not isinstance(component, dict):
                            continue
                            
                        role = component.get('role', '')
                        
                        if role == 'axis-label':
                            axis_info['labels'] = self._extract_axis_labels(component)
                        elif role == 'axis-title':
                            axis_info['title'] = self._extract_axis_title(component)
                        elif role == 'axis-tick':
                            axis_info['ticks'] = self._extract_axis_ticks(component)
                        elif role == 'axis-domain':
                            axis_info['domain'] = self._extract_axis_domain(component)
                        elif role == 'axis-grid':
                            axis_info['grid'] = self._extract_axis_grid(component)
            except Exception as e:
                print(f"Error extracting axis info: {str(e)}")

        def _extract_axis_labels(self, label_component):
            """Extract axis labels"""
            labels = []
            try:
                label_items = label_component.get('items', [])
                if not isinstance(label_items, list):
                    return labels
                    
                for item in label_items:
                    if not isinstance(item, dict):
                        continue
                        
                    text = item.get('text', '')
                    x = item.get('x', 0)
                    y = item.get('y', 0)
                    labels.append({
                        'text': str(text),
                        'position': {'x': x, 'y': y}
                    })
            except Exception as e:
                print(f"Error extracting axis labels: {str(e)}")
            return labels

        def _extract_axis_title(self, title_component):
            """Extract axis title"""
            try:
                title_items = title_component.get('items', [])
                if title_items and isinstance(title_items, list) and title_items[0]:
                    return str(title_items[0].get('text', ''))
            except Exception as e:
                print(f"Error extracting axis title: {str(e)}")
            return ''

        def _extract_axis_ticks(self, tick_component):
            """Extract axis tick positions"""
            ticks = []
            try:
                tick_items = tick_component.get('items', [])
                if not isinstance(tick_items, list):
                    return ticks
                    
                for item in tick_items:
                    if not isinstance(item, dict):
                        continue
                        
                    x = item.get('x', 0)
                    y = item.get('y', 0)
                    ticks.append({'x': x, 'y': y})
            except Exception as e:
                print(f"Error extracting axis ticks: {str(e)}")
            return ticks

        def _extract_axis_domain(self, domain_component):
            """Extract axis domain line"""
            try:
                domain_items = domain_component.get('items', [])
                if domain_items and isinstance(domain_items, list) and domain_items[0]:
                    item = domain_items[0]
                    if isinstance(item, dict):
                        return {
                            'x1': item.get('x', 0),
                            'y1': item.get('y', 0),
                            'x2': item.get('x2', 0),
                            'y2': item.get('y2', 0),
                            'stroke': item.get('stroke', ''),
                            'strokeWidth': item.get('strokeWidth', 0)
                        }
            except Exception as e:
                print(f"Error extracting axis domain: {str(e)}")
            return {}

        def _extract_axis_grid(self, grid_component):
            """Extract axis grid lines"""
            grid_lines = []
            try:
                grid_items = grid_component.get('items', [])
                if not isinstance(grid_items, list):
                    return grid_lines
                    
                for item in grid_items:
                    if not isinstance(item, dict):
                        continue
                        
                    grid_lines.append({
                        'x1': item.get('x', 0),
                        'y1': item.get('y', 0),
                        'x2': item.get('x2', 0),
                        'y2': item.get('y2', 0),
                        'stroke': item.get('stroke', ''),
                        'strokeWidth': item.get('strokeWidth', 0)
                    })
            except Exception as e:
                print(f"Error extracting axis grid: {str(e)}")
            return grid_lines

        def _extract_data_marks(self, mark_item, graph):
            """Extract data points from mark items"""
            try:
                mark_type = mark_item.get('marktype', '')
                mark_items = mark_item.get('items', [])
                if not isinstance(mark_items, list):
                    return
                
                # Store visual properties
                if mark_items:
                    first_item = mark_items[0]
                    if isinstance(first_item, dict):
                        graph['visual_properties'].update({
                            'stroke': first_item.get('stroke', ''),
                            'strokeWidth': first_item.get('strokeWidth', 0),
                            'fill': first_item.get('fill', ''),
                            'opacity': first_item.get('opacity', 1)
                        })
                
                # Extract data points
                for item in mark_items:
                    if not isinstance(item, dict):
                        continue
                        
                    data_point = {
                        'x': item.get('x', 0),
                        'y': item.get('y', 0),
                        'description': item.get('description', ''),
                        'mark_type': mark_type
                    }
                    
                    # Extract value information from description if available
                    desc = data_point['description']
                    if desc:
                        # Try to parse values from description
                        parsed_values = self._parse_description_values(desc)
                        data_point.update(parsed_values)
                    
                    graph['data_points'].append(data_point)
            except Exception as e:
                print(f"Error extracting data marks: {str(e)}")

        def _parse_description_values(self, description):
            """Parse actual data values from description text"""
            values = {}
            try:
                if not description or not isinstance(description, str):
                    return values
                    
                # Look for patterns like "X: value, Y: value" or "Year: value, Price: value"
                import re
                
                # Pattern for year
                year_match = re.search(r'(\d{4})', description)
                if year_match:
                    values['data_x'] = int(year_match.group(1))
                
                # Pattern for dollar amounts
                price_match = re.search(r'\$(\d+\.?\d*)', description)
                if price_match:
                    values['data_y'] = float(price_match.group(1))
                
                # Pattern for generic X: value, Y: value
                x_match = re.search(r'X:\s*([^\s,]+)', description)
                y_match = re.search(r'Y:\s*([^\s,]+)', description)
                
                if x_match and 'data_x' not in values:
                    try:
                        values['data_x'] = float(x_match.group(1))
                    except:
                        values['data_x'] = x_match.group(1)
                
                if y_match and 'data_y' not in values:
                    try:
                        values['data_y'] = float(y_match.group(1))
                    except:
                        values['data_y'] = y_match.group(1)
                        
            except Exception as e:
                print(f"Error parsing description values: {str(e)}")
            
            return values

        def _determine_chart_type(self, graph):
            """Determine chart type from structural elements"""
            try:
                structural_elements = graph.get('structural_elements', [])
                mark_types = [elem.get('type', '') for elem in structural_elements 
                            if elem.get('type', '') in ['line', 'bar', 'point', 'area', 'pie', 'arc']]
                
                if 'arc' in mark_types:
                    return 'pie'
                elif 'line' in mark_types:
                    return 'line'
                elif 'bar' in mark_types:
                    return 'bar'
                elif 'point' in mark_types:
                    return 'scatter'
                elif 'area' in mark_types:
                    return 'area'
                else:
                    return 'unknown'
            except Exception as e:
                print(f"Error determining chart type: {str(e)}")
                return 'unknown'

        def _extract_semantic_content(self, graph):
            """Extract high-level semantic content for comparison"""
            semantic = {
                'data_trend': 'unknown',
                'data_range': {},
                'temporal_extent': {},
                'value_distribution': {},
                'key_statistics': {}
            }
            
            try:
                data_points = graph.get('data_points', [])
                if not data_points or not isinstance(data_points, list):
                    return semantic
                
                chart_type = graph.get('chart_type', '')
                
                if chart_type == 'pie':
                    # For pie charts, extract segment statistics
                    segment_values = []
                    for p in data_points:
                        if isinstance(p, dict) and 'value' in p:
                            try:
                                segment_values.append(float(p['value']))
                            except (ValueError, TypeError):
                                continue
                                
                    if segment_values:
                        semantic['value_distribution'] = {
                            'total_segments': len(segment_values),
                            'largest_segment': max(segment_values),
                            'smallest_segment': min(segment_values),
                            'total_percentage': sum(segment_values)
                        }
                        
                        # Check if percentages sum to approximately 100
                        if abs(sum(segment_values) - 100) < 2:
                            semantic['data_integrity'] = 'valid_percentages'
                        else:
                            semantic['data_integrity'] = 'invalid_percentages'
                
                else:
                    # For other chart types, extract x and y values
                    x_values = []
                    y_values = []
                    
                    for p in data_points:
                        if isinstance(p, dict):
                            if p.get('data_x') is not None:
                                try:
                                    x_values.append(float(p['data_x']))
                                except (ValueError, TypeError):
                                    pass
                            if p.get('data_y') is not None:
                                try:
                                    y_values.append(float(p['data_y']))
                                except (ValueError, TypeError):
                                    pass
                    
                    if x_values and y_values:
                        # Data range
                        semantic['data_range'] = {
                            'x_min': min(x_values),
                            'x_max': max(x_values),
                            'y_min': min(y_values),
                            'y_max': max(y_values)
                        }
                        
                        # Temporal extent (if x values look like years)
                        if all(isinstance(x, (int, float)) and 1900 <= x <= 2100 for x in x_values):
                            semantic['temporal_extent'] = {
                                'start_year': min(x_values),
                                'end_year': max(x_values),
                                'duration': max(x_values) - min(x_values)
                            }
                        
                        # Data trend (simple linear trend)
                        if len(y_values) >= 2:
                            first_y, last_y = y_values[0], y_values[-1]
                            if last_y > first_y * 1.1:
                                semantic['data_trend'] = 'increasing'
                            elif last_y < first_y * 0.9:
                                semantic['data_trend'] = 'decreasing'
                            else:
                                semantic['data_trend'] = 'stable'
                        
                        # Key statistics
                        semantic['key_statistics'] = {
                            'num_points': len(y_values),
                            'y_mean': sum(y_values) / len(y_values),
                            'y_std': (sum((y - sum(y_values)/len(y_values))**2 for y in y_values) / len(y_values))**0.5
                        }
            
            except Exception as e:
                print(f"Error extracting semantic content: {str(e)}")
            
            return semantic

    DEPENDENCIES_AVAILABLE = True
    
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please install required packages:")
    print("pip install anthropic openai sentence-transformers networkx scikit-learn matplotlib Pillow")
    DEPENDENCIES_AVAILABLE = False


def get_api_key_status(api_key):
    """Check if API key is properly configured"""
    if not api_key or len(api_key.strip()) < 10:
        return "❌ Not Set"
    elif api_key.startswith("sk-ant-") or api_key.startswith("sk-"):
        return "✅ Configured"
    else:
        return "⚠️ Invalid Format"


def safe_evaluate_charts(chart1_image, chart2_image, llm_provider, claude_api_key, openai_api_key, progress=gr.Progress()):
    """
    Enhanced wrapper for chart evaluation that includes detailed human-readable explanations
    
    Args:
        chart1_image: PIL Image object for ground truth chart
        chart2_image: PIL Image object for predicted chart
        llm_provider: Selected LLM provider ("Claude" or "GPT-4")
        claude_api_key: Claude API key
        openai_api_key: OpenAI API key
        progress: Gradio progress tracker
        
    Returns:
        ALWAYS returns exactly 4 values: (success_message, error_message, results_dataframe, detailed_explanation)
    """
    
    # Initialize default return values
    default_success = ""
    default_error = ""
    default_df = pd.DataFrame([
        ["Status", "Not Evaluated", "Please check inputs and try again"]
    ], columns=["Metric", "Score", "Description"])
    default_explanation = "No detailed explanation available. Please check inputs and try again."
    
    try:
        # Check dependencies first
        if not DEPENDENCIES_AVAILABLE:
            error_msg = """
            ❌ **Missing Dependencies**
            
            Please install required packages:
            ```
            pip install anthropic openai sentence-transformers networkx scikit-learn matplotlib Pillow pandas numpy
            ```
            """
            return default_success, error_msg, default_df, default_explanation
        
        # Determine which API key to use
        if llm_provider == "Claude":
            api_key = claude_api_key or CLAUDE_API_KEY
            if not api_key or len(api_key.strip()) < 10:
                return default_success, "❌ **Error**: Please set your Claude API key.", default_df, default_explanation
        elif llm_provider.startswith("GPT"):
            api_key = openai_api_key or OPENAI_API_KEY
            if not api_key or len(api_key.strip()) < 10:
                return default_success, "❌ **Error**: Please set your OpenAI API key.", default_df, default_explanation
        else:
            return default_success, f"❌ **Error**: Unsupported LLM provider: {llm_provider}", default_df, default_explanation
        
        # Progress update with error handling
        try:
            if progress:
                progress(0.1, desc="Validating inputs...")
        except:
            pass
        
        # Validate inputs
        if chart1_image is None or chart2_image is None:
            return default_success, "❌ **Error**: Please upload both chart images.", default_df, default_explanation
        
        # Initialize evaluator
        try:
            if progress:
                progress(0.2, desc=f"Initializing {llm_provider} evaluator...")
        except:
            pass
        
        # Set model configuration based on provider
        if llm_provider == "Claude":
            model_config = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 4000,
                "temperature": 0.1
            }
        elif llm_provider == "GPT-4":
            model_config = {
                "model": "gpt-4-vision-preview",
                "max_tokens": 4000,
                "temperature": 0.1
            }
        else:
            model_config = {}
        
        evaluator = ChartEval(
            llm_provider=llm_provider,
            api_key=api_key.strip(),
            model_config=model_config
        )
        
        # Convert PIL images to temporary files and base64
        try:
            if progress:
                progress(0.3, desc="Converting images to temporary files...")
        except:
            pass
        
        chart1_path = None
        chart2_path = None
        chart1_b64 = None
        chart2_b64 = None
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp1:
                chart1_image.save(tmp1.name, 'PNG')
                chart1_path = tmp1.name
                
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp2:
                chart2_image.save(tmp2.name, 'PNG')
                chart2_path = tmp2.name
                
            # Convert to base64 for detailed explanation
            import io
            chart1_buffer = io.BytesIO()
            chart1_image.save(chart1_buffer, format='PNG')
            chart1_b64 = base64.b64encode(chart1_buffer.getvalue()).decode('utf-8')
            
            chart2_buffer = io.BytesIO()
            chart2_image.save(chart2_buffer, format='PNG')
            chart2_b64 = base64.b64encode(chart2_buffer.getvalue()).decode('utf-8')
            
        except Exception as e:
            return default_success, f"❌ **Error**: Failed to process uploaded images: {str(e)}", default_df, default_explanation
        
        try:
            # Analyze Chart 1
            try:
                if progress:
                    progress(0.4, desc=f"Analyzing Chart 1 with {llm_provider}...")
            except:
                pass
            
            try:
                vega1 = evaluator.chartToVega(chart1_path)
                graph1 = evaluator.vegaToGraph(vega1)
            except Exception as e:
                print(f"Chart 1 analysis error: {e}")
                graph1 = {
                    'chart_type': 'unknown',
                    'data_points': [],
                    'axes': {'x_axis': {}, 'y_axis': {}},
                    'title': 'Chart 1 (Analysis Error)',
                    'visual_properties': {},
                    'semantic_content': {},
                    'parse_error': str(e)
                }
            
            # Analyze Chart 2
            try:
                if progress:
                    progress(0.6, desc=f"Analyzing Chart 2 with {llm_provider}...")
            except:
                pass
            
            try:
                vega2 = evaluator.chartToVega(chart2_path)
                graph2 = evaluator.vegaToGraph(vega2)
            except Exception as e:
                print(f"Chart 2 analysis error: {e}")
                graph2 = {
                    'chart_type': 'unknown',
                    'data_points': [],
                    'axes': {'x_axis': {}, 'y_axis': {}},
                    'title': 'Chart 2 (Analysis Error)',
                    'visual_properties': {},
                    'semantic_content': {},
                    'parse_error': str(e)
                }
            
            # Run evaluation metrics
            try:
                if progress:
                    progress(0.8, desc="Running evaluation metrics...")
            except:
                pass
            
            bert_score, hall_score, omis_score, ged_score = evaluator.compare(graph1, graph2)
            
            # Generate detailed explanation
            try:
                if progress:
                    progress(0.9, desc="Generating detailed analysis...")
            except:
                pass
            
            metrics_for_explanation = {
                'bert_score': bert_score,
                'hallucination_score': hall_score,
                'omission_score': omis_score,
                'ged_score': ged_score
            }
            
            detailed_explanation = evaluator.generate_detailed_explanation(
                graph1, graph2, metrics_for_explanation, chart1_b64, chart2_b64
            )
            
            # Format results
            try:
                if progress:
                    progress(0.95, desc="Formatting results...")
            except:
                pass
            
            success_message = f"""
            ## ✅ **Evaluation Completed Successfully!**
            
            ### 🤖 **LLM Provider**: {llm_provider}
            
            ### 📊 **Chart Analysis Summary**
            - **Chart 1**: {graph1.get('chart_type', 'unknown')} chart with {len(graph1.get('data_points', []))} data points
            - **Chart 2**: {graph2.get('chart_type', 'unknown')} chart with {len(graph2.get('data_points', []))} data points
            
            ### 🏆 **Overall Scores**
            - **Semantic Similarity (F1)**: {bert_score.get('f1', 0):.3f}
            - **Hallucination Rate**: {hall_score.get('hallucination_rate', 0):.3f} (lower is better)
            - **Omission Rate**: {omis_score.get('omission_rate', 0):.3f} (lower is better)  
            - **Structural Difference**: {ged_score.get('normalized_ged', 0):.3f} (lower is better)
            """
            
            # Create detailed results DataFrame
            results_data = [
                ["LLM Provider", llm_provider, f"Chart analysis performed using {llm_provider}"],
                ["GraphBERT Correctness", f"{bert_score.get('precision', 0):.3f}", "Semantic similarity precision"],
                ["GraphBERT Completeness", f"{bert_score.get('recall', 0):.3f}", "Semantic similarity recall"],
                ["GraphBERT F1", f"{bert_score.get('f1', 0):.3f}", "Overall semantic similarity"],
                ["Hallucination Rate", f"{hall_score.get('hallucination_rate', 0):.3f}", "False information rate"],
                ["Hallucination Count", str(hall_score.get('hallucination_count', 0)), "Number of hallucinated elements"],
                ["Omission Rate", f"{omis_score.get('omission_rate', 0):.3f}", "Missing information rate"],
                ["Omission Count", str(omis_score.get('omission_count', 0)), "Number of omitted elements"],
                ["Graph Edit Distance", f"{ged_score.get('ged_distance', 0)}", "Raw structural differences"],
                ["Normalized GED", f"{ged_score.get('normalized_ged', 0):.3f}", "Normalized structural similarity"]
            ]
            
            try:
                results_df = pd.DataFrame(results_data, columns=["Metric", "Score", "Description"])
            except Exception as e:
                print(f"DataFrame creation error: {e}")
                results_df = default_df
            
            try:
                if progress:
                    progress(1.0, desc="Complete!")
            except:
                pass
            
            return success_message, default_error, results_df, detailed_explanation
            
        finally:
            # Clean up temporary files
            try:
                if chart1_path and os.path.exists(chart1_path):
                    os.unlink(chart1_path)
                if chart2_path and os.path.exists(chart2_path):
                    os.unlink(chart2_path)
            except:
                pass
                
    except Exception as e:
        error_msg = f"""
        ❌ **Evaluation Failed**
        
        **Error**: {str(e)}
        
        **Common issues:**
        - API key not configured or invalid
        - Network connection problems  
        - Image processing errors
        - API rate limits exceeded
        
        **Troubleshooting:**
        - Set your API key for the selected LLM provider
        - Verify your API key is correct and active
        - Ensure images are valid chart images (PNG, JPG, etc.)
        - Check your internet connection
        - Wait a moment and try again
        """
        
        print(f"Full error traceback: {traceback.format_exc()}")
        return default_success, error_msg, default_df, default_explanation


def load_example_charts(example_name):
    """
    Load example chart images based on selection
    
    Args:
        example_name: Name of the selected example
        
    Returns:
        Tuple of (ground_truth_image, predicted_image, info_message)
    """
    if example_name == "Select an example...":
        return None, None, ""
    
    if example_name not in EXAMPLE_CHART_PAIRS:
        return None, None, "❌ Example not found"
    
    example_data = EXAMPLE_CHART_PAIRS[example_name]
    gt_path = example_data["ground_truth"]
    pred_path = example_data["predicted"]
    description = example_data["description"]
    
    try:
        # Check if files exist
        if not os.path.exists(gt_path):
            # Create a placeholder image if file doesn't exist
            gt_image = create_placeholder_image(f"Ground Truth\n{example_name}", (400, 300))
        else:
            gt_image = Image.open(gt_path)
        
        if not os.path.exists(pred_path):
            # Create a placeholder image if file doesn't exist  
            pred_image = create_placeholder_image(f"Predicted\n{example_name}", (400, 300))
        else:
            pred_image = Image.open(pred_path)
        
        info_message = f"""
        ### 📋 **Example Loaded: {example_name}**
        
        **Description**: {description}
        
        **Files**:
        - Ground Truth: `{gt_path}`
        - Predicted: `{pred_path}`
        
        ℹ️ *If you see placeholder images, replace the file paths in the code with your actual example images.*
        """
        
        return gt_image, pred_image, info_message
        
    except Exception as e:
        error_msg = f"❌ Error loading example images: {str(e)}"
        # Return placeholder images on error
        gt_placeholder = create_placeholder_image(f"Error loading\nGround Truth", (400, 300))
        pred_placeholder = create_placeholder_image(f"Error loading\nPredicted", (400, 300))
        return gt_placeholder, pred_placeholder, error_msg


def create_placeholder_image(text, size=(400, 300)):
    """
    Create a placeholder image with text
    
    Args:
        text: Text to display on the image
        size: Tuple of (width, height)
        
    Returns:
        PIL Image object
    """
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a new image with white background
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Calculate text position (center)
    if font:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    else:
        text_width = len(text) * 8  # Rough estimate
        text_height = 16
    
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    # Draw the text
    draw.text((x, y), text, fill='black', font=font)
    
    # Draw a border
    draw.rectangle([0, 0, size[0]-1, size[1]-1], outline='gray', width=2)
    
    return img


def create_demo():
    """Create the enhanced Gradio interface with detailed explanations"""
    
    # Define the interface
    with gr.Blocks(
        title="Enhanced Chart Evaluation System",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px;
            margin: auto;
        }
        .metric-card {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
        }
        .explanation-box {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 10px 0;
            max-height: 600px;
            overflow-y: auto;
        }
        """
    ) as demo:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>📊 Enhanced Chart Evaluation System</h1>
            <p style="font-size: 18px; color: #666;">
                Compare two chart images using advanced evaluation metrics with detailed human-readable explanations.
                Get GraphBERT Score, Hallucination Detection, Omission Analysis, Graph Edit Distance, 
                and comprehensive data analyst insights.
            </p>
            <p style="font-size: 16px; color: #888;">
                🎯 Ready to use with Claude or GPT-4! Now includes detailed explanations pointing to specific chart elements
            </p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h3>🔧 Configuration</h3>")
                
                # LLM Provider Selection
                llm_provider = gr.Dropdown(
                    choices=["Claude", "GPT-4"],
                    value="Claude",
                    label="🤖 LLM Provider",
                    info="Select the AI model to analyze your charts"
                )
                
                # API Key inputs
                claude_api_key_input = gr.Textbox(
                    label="🔑 Claude API Key",
                    type="password",
                    placeholder="Enter your Claude API key (or leave blank to use configured key)",
                    value="",
                    visible=True
                )
                
                openai_api_key_input = gr.Textbox(
                    label="🔑 OpenAI API Key", 
                    type="password",
                    placeholder="Enter your OpenAI API key (or leave blank to use configured key)",
                    value="",
                    visible=False
                )
                
                # API Key Status Display
                claude_status = get_api_key_status(CLAUDE_API_KEY)
                openai_status = get_api_key_status(OPENAI_API_KEY)
                    
                api_status_display = gr.HTML(f"""
                    <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0; border: 1px solid #dee2e6;">
                        <strong>🔑 API Key Status:</strong><br>
                        <span style="display: block; margin: 5px 0;">Claude: {claude_status}</span>
                        <span style="display: block; margin: 5px 0;">OpenAI: {openai_status}</span>
                        <small style="color: #6c757d;">Configure API keys in the script or enter them above</small>
                    </div>
                    """)
                    
                # Function to toggle API key visibility
                def toggle_api_key_fields(provider):
                        if provider == "Claude":
                            return gr.update(visible=True), gr.update(visible=False)
                        elif provider == "GPT-4":
                            return gr.update(visible=True), gr.update(visible=True)
                        else:
                            return gr.update(visible=True), gr.update(visible=False)
                    
                llm_provider.change(
                        fn=toggle_api_key_fields,
                        inputs=[llm_provider],
                        outputs=[claude_api_key_input, openai_api_key_input]
                    )
                    
                gr.HTML("""
                    <div style="background: #f0f8ff; padding: 10px; border-radius: 5px; margin: 10px 0;">
                        <strong>📝 How to use:</strong><br>
                        1. Select your preferred LLM provider (Claude or GPT-4)<br>
                        2. Enter API key if not configured in script<br>
                        3. Either:<br>
                        &nbsp;&nbsp;&nbsp;• Select a pre-loaded example from the dropdown, OR<br>
                        &nbsp;&nbsp;&nbsp;• Upload your own ground truth chart (Chart 1)<br>
                        &nbsp;&nbsp;&nbsp;• Upload your own predicted/generated chart (Chart 2)<br>
                        4. Click "Evaluate Charts" to run the analysis
                    </div>
                    """)
                    
                evaluate_btn = gr.Button(
                        "🚀 Evaluate Charts",
                        variant="primary",
                        size="lg"
                        )
                    
                gr.HTML("""
                    <div style="background: #fff8e1; padding: 10px; border-radius: 5px; margin: 10px 0;">
                        <strong>📊 Metrics Explained:</strong><br>
                        • <strong>GraphBERT F1</strong>: Semantic similarity (higher = better)<br>
                        • <strong>Hallucination Rate</strong>: False information (lower = better)<br>
                        • <strong>Omission Rate</strong>: Missing information (lower = better)<br>
                        • <strong>Normalized GED</strong>: Structural differences (lower = better)<br>
                        • <strong>Detailed Explanation</strong>: Human-readable analysis with specific examples
                    </div>
                    """)
            
            with gr.Column(scale=1):
                gr.HTML("<h3>📈 Chart Images</h3>")
                
                # Example selection dropdown
                gr.HTML("<h4>🎯 Quick Examples</h4>")
                example_dropdown = gr.Dropdown(
                    choices=["Select an example..."] + list(EXAMPLE_CHART_PAIRS.keys()),
                    value="Select an example...",
                    label="Choose from pre-loaded examples",
                    info="Select an example to automatically load both ground truth and predicted charts"
                )
                
                example_info = gr.Markdown(
                    value="Select an example above to see details and load chart images automatically.",
                    visible=True
                )
                
                gr.HTML("<h4>📤 Or Upload Your Own</h4>")
                
                chart1_input = gr.Image(
                    label="Chart 1 (Ground Truth)",
                    type="pil",
                    height=300
                )
                
                chart2_input = gr.Image(
                    label="Chart 2 (Predicted/Generated)",
                    type="pil", 
                    height=300
                )
        
        gr.HTML("<hr style='margin: 30px 0;'>")
        
        # Results section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h3>📋 Results</h3>")
                
                success_output = gr.Markdown(
                    label="Success Message",
                    visible=True
                )
                
                error_output = gr.Markdown(
                    label="Error Message", 
                    visible=True
                )
                
                results_output = gr.Dataframe(
                    label="Detailed Metrics"
                )
                
        # NEW: Detailed Explanation Section
        gr.HTML("<hr style='margin: 30px 0;'>")
        
        with gr.Row():
            with gr.Column():
                gr.HTML("<h3>🔍 Detailed Analysis & Insights</h3>")
                gr.HTML("""
                <div style="background: #e8f4fd; padding: 10px; border-radius: 5px; margin: 10px 0;">
                    <strong>📋 What you'll get:</strong><br>
                    • Executive summary with accuracy score<br>
                    • Specific examples of what went right and wrong<br>
                    • Element-by-element comparison (titles, data, axes, etc.)<br>
                    • Actionable recommendations for improvement<br>
                    • Impact assessment for decision-making
                </div>
                """)
                
                detailed_explanation_output = gr.Markdown(
                    value="Detailed explanation will appear here after evaluation.",
                    label="Human-Readable Analysis",
                    elem_classes=["explanation-box"]
                )
                
        # Example section
        gr.HTML("<hr style='margin: 30px 0;'>")
            
        with gr.Accordion("📚 Examples & Help", open=False):
            gr.HTML("""
            <div style="padding: 20px;">
                <h4>🔑 API Key Configuration</h4>
                <p>To use this application, you need API keys for your chosen provider:</p>
                
                <h5>Claude API Key:</h5>
                <ol>
                    <li>Get your Claude API key from <a href="https://console.anthropic.com/" target="_blank">console.anthropic.com</a></li>
                    <li>Either enter it in the Claude API Key field above, or</li>
                    <li>Set it permanently in the script by editing the <code>CLAUDE_API_KEY</code> variable</li>
                </ol>
                
                <h5>OpenAI API Key (for GPT-4):</h5>
                <ol>
                    <li>Get your OpenAI API key from <a href="https://platform.openai.com/api-keys" target="_blank">platform.openai.com</a></li>
                    <li>Either enter it in the OpenAI API Key field above, or</li>
                    <li>Set it permanently in the script by editing the <code>OPENAI_API_KEY</code> variable</li>
                </ol>
                
                <h4>🤖 LLM Provider Comparison</h4>
                <ul>
                    <li><strong>Claude</strong>: Excellent at detailed chart analysis, precise data extraction, comprehensive explanations</li>
                    <li><strong>GPT-4</strong>: Good vision capabilities, different analytical perspective, thorough insights</li>
                </ul>
                
                <h4>🎯 Quick Start with Examples</h4>
                <p>Use the dropdown above to try pre-loaded chart examples. Each example includes:</p>
                <ul>
                    <li><strong>Ground Truth Chart</strong>: The reference/correct chart</li>
                    <li><strong>Predicted Chart</strong>: The generated/predicted version to evaluate</li>
                    <li><strong>Description</strong>: Context about what the chart represents</li>
                </ul>
                
                <h4>🏆 What makes a good chart comparison?</h4>
                <ul>
                    <li><strong>High GraphBERT F1 (>0.8)</strong>: Charts convey similar semantic information</li>
                    <li><strong>Low Hallucination Rate (<0.2)</strong>: Predicted chart doesn't add false information</li>
                    <li><strong>Low Omission Rate (<0.2)</strong>: Predicted chart doesn't miss important information</li>
                    <li><strong>Low Normalized GED (<0.3)</strong>: Charts have similar structure</li>
                    <li><strong>Clear Detailed Explanation</strong>: Specific examples of strengths and areas for improvement</li>
                </ul>
                
                <h4>🔍 Understanding the Detailed Analysis</h4>
                <p>The enhanced system now provides:</p>
                <ul>
                    <li><strong>Executive Summary</strong>: High-level assessment with accuracy score</li>
                    <li><strong>Specific Examples</strong>: References to actual data points, labels, and chart elements</li>
                    <li><strong>Element Breakdown</strong>: Detailed comparison of titles, axes, data, and visual design</li>
                    <li><strong>Error Analysis</strong>: Specific data errors, missing elements, and hallucinations</li>
                    <li><strong>Actionable Recommendations</strong>: Concrete steps for improvement</li>
                    <li><strong>Impact Assessment</strong>: How issues affect interpretation and decision-making</li>
                </ul>
                
                <h4>📁 Adding Your Own Examples</h4>
                <p>To add your own example chart pairs:</p>
                <ol>
                    <li>Create an <code>examples/</code> folder in your project directory</li>
                    <li>Add your chart image pairs (ground truth + predicted)</li>
                    <li>Update the <code>EXAMPLE_CHART_PAIRS</code> dictionary in the code</li>
                    <li>Replace the placeholder paths with your actual file paths</li>
                </ol>
                
                <h4>🔧 Troubleshooting</h4>
                <ul>
                    <li><strong>API Key Issues</strong>: Make sure your API key is set and valid for the selected provider</li>
                    <li><strong>Provider Switching</strong>: You can switch between Claude and GPT-4 at any time</li>
                    <li><strong>Image Quality</strong>: Use clear, high-resolution chart images</li>
                    <li><strong>Chart Types</strong>: Works best with line charts, bar charts, pie charts, and scatter plots</li>
                    <li><strong>Processing Time</strong>: Analysis may take 60-90 seconds per chart due to detailed explanation</li>
                    <li><strong>Long Explanations</strong>: Detailed analysis may be lengthy but provides comprehensive insights</li>
                </ul>
                
                <h4>📞 Support</h4>
                <p>For issues or questions, check the console logs for detailed error messages.</p>
            </div>
            """)
        
        # Connect the example dropdown to load example images
        example_dropdown.change(
            fn=load_example_charts,
            inputs=[example_dropdown],
            outputs=[chart1_input, chart2_input, example_info]
        )
        
        # Connect the evaluation function (now with detailed explanation)
        evaluate_btn.click(
            fn=safe_evaluate_charts,
            inputs=[chart1_input, chart2_input, llm_provider, claude_api_key_input, openai_api_key_input],
            outputs=[success_output, error_output, results_output, detailed_explanation_output],
            show_progress=True
        )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )