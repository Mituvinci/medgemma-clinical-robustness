"""
Evaluation Results Analysis Script

Generates comprehensive metrics, tables, and plots from evaluation results.

Usage:
    python scripts/analyze_evaluation_results.py \
        --results logs/evaluation_medgemma-27b-it-vertex_without_options/nejim_evaluation_*.json \
        --groundtruth NEJIM/NEJM_Groundtruth.csv \
        --output logs/analysis_report_27b_it.html
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import re
import csv
from datetime import datetime
from typing import Dict, List, Tuple, Any
import argparse
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class EvaluationAnalyzer:
    """Analyzes evaluation results and generates comprehensive metrics."""

    def __init__(self, groundtruth_path: str):
        """
        Initialize analyzer with ground truth data.

        Args:
            groundtruth_path: Path to CSV file with correct diagnoses
        """
        self.groundtruth = self._load_groundtruth(groundtruth_path)
        self.results_data = []

    def _load_groundtruth(self, path: str) -> Dict[str, str]:
        """
        Load ground truth diagnoses from CSV.

        Format: Date,Ground_Truth
        Example: 1/2/2025,Argyria

        Returns:
            Dict mapping case_id (e.g., "01_02_25") to diagnosis
        """
        groundtruth = {}

        with open(path, 'r', encoding='utf-8-sig') as f:  # utf-8-sig removes BOM
            reader = csv.DictReader(f)
            for row in reader:
                date_str = row['Date'].strip()
                diagnosis = row['Ground_Truth'].strip()

                # Parse date and convert to case_id format
                # "1/2/2025" → "01_02_25" (NEJM format)
                # "01_01_23" → already in case_id format (JDCR format)
                if '/' in date_str:
                    try:
                        date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                        case_id = date_obj.strftime('%m_%d_%y')
                        groundtruth[case_id] = diagnosis
                    except ValueError:
                        print(f"Warning: Could not parse date: {date_str}")
                        continue
                else:
                    groundtruth[date_str] = diagnosis

        print(f"Loaded {len(groundtruth)} ground truth diagnoses")
        return groundtruth

    def load_evaluation_results(self, results_path: str):
        """
        Load evaluation results from JSON file.

        Args:
            results_path: Path to evaluation results JSON
        """
        with open(results_path, 'r') as f:
            data = json.load(f)

        self.metadata = data.get('metadata', {})
        self.results_data = data.get('results', [])

        print(f"Loaded {len(self.results_data)} evaluation results")
        print(f"Metadata: {self.metadata}")

    def extract_diagnosis(self, response_text: str) -> Tuple[str, float]:
        """
        Extract primary diagnosis from SOAP note response.

        Args:
            response_text: Full agent response with SOAP note

        Returns:
            (diagnosis, confidence) tuple
        """
        # Helper to extract confidence from nearby text
        def _extract_confidence(text: str) -> float:
            for cp in [
                r'Confidence(?:\s+Score)?[:\s]+\*?\*?([\d.]+)',
                r'\(Confidence[:\s]+([\d.]+)\)',
            ]:
                cm = re.search(cp, text, re.IGNORECASE)
                if cm:
                    try:
                        return float(cm.group(1))
                    except ValueError:
                        pass
            return 0.5

        # Helper to clean diagnosis text
        def _clean(diag: str) -> str:
            diag = re.sub(r'\*\*', '', diag)  # remove bold markdown
            diag = re.sub(r'\*', '', diag)    # remove italic markdown
            diag = re.sub(r'^\s*[*\-•]\s*', '', diag)  # remove leading bullets
            diag = re.sub(r'^\s*\d+[.)]\s*', '', diag)  # remove leading numbers
            # Remove MCQ option letter prefix: A) B) (A) (B) A. B. etc.
            # Requires delimiter after letter to avoid eating first char of diagnosis
            diag = re.sub(r'^\(?[A-Ea-e][.):\)]\s*', '', diag)
            # Remove trailing confidence text
            diag = re.sub(r'\s*[-–]\s*Confidence.*$', '', diag, flags=re.IGNORECASE)
            diag = re.sub(r'\s*\(Confidence[^)]*\)\s*$', '', diag, flags=re.IGNORECASE)
            # Remove trailing colon
            diag = diag.rstrip(':').strip()
            return diag

        # Pattern 1: **Primary Diagnosis:** on same line or next line with bullet
        # Handles: **Primary Diagnosis:** **Text** and **Primary Diagnosis:**\n*   **Text**
        pattern1 = r'\*\*Primary Diagnosis:\*\*\s*(?:\n\s*[*\-1-9.]*\s*)?(?:\*\*)?(.+?)(?:\*\*)?(?:\s*[-–]\s*Confidence|\s*\(Confidence|\n|$)'
        match = re.search(pattern1, response_text, re.IGNORECASE)
        if match:
            diagnosis = _clean(match.group(1))
            if diagnosis and diagnosis.lower() != 'unknown':
                confidence = _extract_confidence(response_text[match.start():match.start()+500])
                return diagnosis, confidence

        # Pattern 2: Primary Diagnosis: Text (no bold markers)
        pattern2 = r'Primary Diagnosis:\s*(?:\n\s*[*\-1-9.]*\s*)?(.+?)(?:\s*[-–]\s*Confidence|\s*\(Confidence|\n|$)'
        match = re.search(pattern2, response_text, re.IGNORECASE)
        if match:
            diagnosis = _clean(match.group(1))
            if diagnosis and diagnosis.lower() != 'unknown':
                confidence = _extract_confidence(response_text[match.start():match.start()+500])
                return diagnosis, confidence

        # Pattern 2b: Most Likely / Final / Presumptive Diagnosis (bold)
        pattern2b = r'\*\*(?:Most Likely|Final|Presumptive)\s+Diagnosis:\*\*\s*(?:\n\s*[*\-1-9.]*\s*)?(?:\*\*)?(.+?)(?:\*\*)?(?:\s*[-–]\s*Confidence|\s*\(Confidence|\n|$)'
        match = re.search(pattern2b, response_text, re.IGNORECASE)
        if match:
            diagnosis = _clean(match.group(1))
            if diagnosis and diagnosis.lower() != 'unknown':
                confidence = _extract_confidence(response_text[match.start():match.start()+500])
                return diagnosis, confidence

        # Pattern 2c: Most Likely / Final / Presumptive Diagnosis (no bold)
        pattern2c = r'(?:Most Likely|Final|Presumptive)\s+Diagnosis:\s*(?:\n\s*[*\-1-9.]*\s*)?(.+?)(?:\s*[-–]\s*Confidence|\s*\(Confidence|\n|$)'
        match = re.search(pattern2c, response_text, re.IGNORECASE)
        if match:
            diagnosis = _clean(match.group(1))
            if diagnosis and diagnosis.lower() != 'unknown':
                confidence = _extract_confidence(response_text[match.start():match.start()+500])
                return diagnosis, confidence

        # Pattern 3: Assessment section - look for first bold diagnosis (allow up to 3 intervening lines)
        pattern3 = r'Assessment \(A\):[^\n]*\n(?:[^\n]*\n){0,3}?\*\*([^*\n]+)\*\*'
        match = re.search(pattern3, response_text, re.IGNORECASE | re.DOTALL)
        if match:
            diagnosis = _clean(match.group(1))
            if diagnosis:
                return diagnosis, 0.3

        # Pattern 4: Assessment section - any capitalized diagnosis phrase
        pattern4 = r'Assessment \(A\):[^\n]*\n[^\n]*?([A-Z][a-z]+(?:\s+[a-z]+){0,5})'
        match = re.search(pattern4, response_text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip(), 0.3

        # Pattern 5: "Conclusion" section with bold diagnosis
        pattern5 = r'\*\*Conclusion\*\*:?\s*.*?(?:most likely|diagnosis is)\s*\*\*([^*\n]+)\*\*'
        match = re.search(pattern5, response_text, re.IGNORECASE | re.DOTALL)
        if match:
            diagnosis = _clean(match.group(1))
            if diagnosis:
                return diagnosis, 0.3

        # No diagnosis found
        return "Unknown", 0.0

    def extract_all_diagnoses(self, response_text: str) -> List[Tuple[str, float]]:
        """
        Extract primary + all differential diagnoses from SOAP note response.

        Returns:
            List of (diagnosis, confidence) tuples, primary first then differentials.
        """
        diagnoses = []

        # Helper to clean diagnosis text (same as in extract_diagnosis)
        def _clean(diag: str) -> str:
            diag = re.sub(r'\*\*', '', diag)
            diag = re.sub(r'\*', '', diag)
            diag = re.sub(r'^\s*[*\-•]\s*', '', diag)
            diag = re.sub(r'^\s*\d+[.)]\s*', '', diag)
            # Remove MCQ option letter prefix: A) B) (A) (B) A. B. etc.
            # Requires delimiter after letter to avoid eating first char of diagnosis
            diag = re.sub(r'^\(?[A-Ea-e][.):\)]\s*', '', diag)
            diag = re.sub(r'\s*[-–]\s*Confidence.*$', '', diag, flags=re.IGNORECASE)
            diag = re.sub(r'\s*\(Confidence[^)]*\)\s*$', '', diag, flags=re.IGNORECASE)
            diag = diag.rstrip(':').strip()
            return diag

        def _extract_confidence(text: str) -> float:
            for cp in [
                r'Confidence(?:\s+Score)?[:\s]+\*?\*?([\d.]+)',
                r'\(Confidence[:\s]+([\d.]+)\)',
            ]:
                cm = re.search(cp, text, re.IGNORECASE)
                if cm:
                    try:
                        return float(cm.group(1))
                    except ValueError:
                        pass
            return 0.5

        # Get primary diagnosis first
        primary, primary_conf = self.extract_diagnosis(response_text)
        if primary != "Unknown":
            diagnoses.append((primary, primary_conf))

        # Extract differential diagnoses
        # Pattern: **Diagnosis Name** - Confidence: 0.XX or (Confidence: 0.XX)
        # These appear after "Differential Diagnos" section
        diff_section = re.search(
            r'Differential Diagnos[ei]s.*?(?=###\s*Plan|##\s*Plan|\*\*P[:\s]|### P\b|\Z)',
            response_text, re.IGNORECASE | re.DOTALL
        )
        if diff_section:
            section_text = diff_section.group(0)
            # Find each differential: numbered bold text with confidence score nearby
            # Pattern: 1. **Diagnosis Name** or *  **Diagnosis Name**
            diff_pattern = r'(?:^|\n)\s*(?:\d+[.)]\s+|\*\s+)\*\*([A-Z][^*\n]{3,80}?)\*\*'
            for m in re.finditer(diff_pattern, section_text):
                diag = _clean(m.group(1))
                # Skip section headers and non-diagnosis text
                skip_words = {
                    'evidence', 'rationale', 'supporting evidence', 'confidence',
                    'confidence score', 'plan', 'differential diagnoses',
                    'differential diagnosis', 'relevant guidelines', 'soap note',
                    'subjective', 'objective', 'assessment', 'diagnostic',
                    'treatment', 'follow-up', 'citations', 'reasoning',
                    'key diagnostic criteria', 'clinical features',
                    'summary', 'recommendations', 'patient', 'case id',
                    'primary diagnosis', 'specific citations', 'monitoring',
                    'referral', 'diagnostic tests', 'medication review',
                    'final diagnosis', 'p: plan',
                    # SOAP section headers that slip through
                    'justification', 'onset after vaccination', 'morphology',
                    'distribution', 'symptoms', 'skin', 'imaging',
                    'biopsy', 'biopsy findings', 'laboratory findings',
                    'cultures', 'physical examination', 'chief complaint',
                    'history of present illness', 'past medical history',
                    'family history', 'detailed physical examination',
                    'cutaneous examination', 'patient age',
                    'duration of symptoms', 'confidence score',
                }
                if diag and diag.lower() not in skip_words and len(diag) > 3:
                    conf = _extract_confidence(section_text[m.start():m.start()+300])
                    # Avoid duplicating primary
                    if not any(self.fuzzy_match(diag, d[0], 0.85) for d in diagnoses):
                        diagnoses.append((diag, conf))
                        # Limit to 4 differentials max
                        if len(diagnoses) >= 5:
                            break

        return diagnoses

    def fuzzy_match(self, pred: str, true: str, threshold: float = 0.7) -> bool:
        """
        Fuzzy match two diagnosis strings.

        Args:
            pred: Predicted diagnosis
            true: Ground truth diagnosis
            threshold: Similarity threshold (0-1)

        Returns:
            True if match, False otherwise
        """
        # Normalize both strings
        pred_norm = pred.lower().strip()
        true_norm = true.lower().strip()

        # Exact match
        if pred_norm == true_norm:
            return True

        # Check if one is substring of other
        if pred_norm in true_norm or true_norm in pred_norm:
            return True

        # Fuzzy similarity
        similarity = SequenceMatcher(None, pred_norm, true_norm).ratio()
        return similarity >= threshold

    def analyze_all_results(self) -> Dict[str, Any]:
        """
        Analyze all evaluation results and calculate metrics.

        Returns:
            Dict with comprehensive metrics
        """
        metrics = {
            'by_variant': {},
            'by_model': {},
            'overall': {},
            'detailed_results': []
        }

        variants = ['original', 'history_only', 'image_only', 'exam_only', 'exam_restricted']

        for variant in variants:
            metrics['by_variant'][variant] = {
                'total': 0,
                'correct': 0,
                'top3_correct': 0,
                'top4_correct': 0,
                'paused': 0,
                'errors': 0,
                'avg_confidence': 0.0,
                'avg_execution_time_ms': 0.0,
                'diagnoses': []
            }

        # Process each result
        for result in self.results_data:
            case_id = result['case_id']
            variant = result['variant']

            # Extract case number (e.g., "NEJIM_01_02_25_original" → "01_02_25")
            case_match = re.search(r'(\d{2}_\d{2}_\d{2})', case_id)
            if not case_match:
                continue
            case_num = case_match.group(1)

            # Get ground truth
            true_diagnosis = self.groundtruth.get(case_num, "Unknown")

            # Extract predicted diagnosis
            response_text = result.get('response_text', '')
            pred_diagnosis, confidence = self.extract_diagnosis(response_text)

            # Check if correct (Top-1)
            is_correct = self.fuzzy_match(pred_diagnosis, true_diagnosis)

            # Extract all diagnoses for Top-3 and Top-4 accuracy
            all_diagnoses = self.extract_all_diagnoses(response_text)
            top3_diagnoses = all_diagnoses[:3]
            top4_diagnoses = all_diagnoses[:4]
            is_top3_correct = any(self.fuzzy_match(d[0], true_diagnosis) for d in top3_diagnoses)
            is_top4_correct = any(self.fuzzy_match(d[0], true_diagnosis) for d in top4_diagnoses)

            # Metrics
            paused = result.get('agentic_pause_triggered', False)
            error = result.get('error') is not None
            exec_time = result.get('execution_time_ms', 0)

            # Update variant metrics
            if variant in metrics['by_variant']:
                var_metrics = metrics['by_variant'][variant]
                var_metrics['total'] += 1
                if is_correct:
                    var_metrics['correct'] += 1
                if is_top3_correct:
                    var_metrics['top3_correct'] += 1
                if is_top4_correct:
                    var_metrics['top4_correct'] += 1
                if paused:
                    var_metrics['paused'] += 1
                if error:
                    var_metrics['errors'] += 1
                var_metrics['avg_confidence'] += confidence
                var_metrics['avg_execution_time_ms'] += exec_time
                var_metrics['diagnoses'].append({
                    'case_id': case_num,
                    'predicted': pred_diagnosis,
                    'all_predicted': [d[0] for d in all_diagnoses],
                    'true': true_diagnosis,
                    'correct': is_correct,
                    'top3_correct': is_top3_correct,
                    'top4_correct': is_top4_correct,
                    'confidence': confidence,
                    'paused': paused
                })

            # Store detailed result
            metrics['detailed_results'].append({
                'case_id': case_num,
                'variant': variant,
                'predicted_diagnosis': pred_diagnosis,
                'all_diagnoses': [d[0] for d in all_diagnoses],
                'true_diagnosis': true_diagnosis,
                'correct': is_correct,
                'top3_correct': is_top3_correct,
                'top4_correct': is_top4_correct,
                'confidence': confidence,
                'paused': paused,
                'error': error,
                'execution_time_ms': exec_time
            })

        # Calculate averages and rates
        for variant, data in metrics['by_variant'].items():
            if data['total'] > 0:
                data['accuracy'] = data['correct'] / data['total']
                data['top3_accuracy'] = data['top3_correct'] / data['total']
                data['top4_accuracy'] = data['top4_correct'] / data['total']
                data['pause_rate'] = data['paused'] / data['total']
                data['error_rate'] = data['errors'] / data['total']
                data['avg_confidence'] /= data['total']
                data['avg_execution_time_ms'] /= data['total']

        # Overall metrics
        total_cases = len(metrics['detailed_results'])
        if total_cases > 0:
            metrics['overall'] = {
                'total_evaluations': total_cases,
                'overall_accuracy': sum(1 for r in metrics['detailed_results'] if r['correct']) / total_cases,
                'overall_top3_accuracy': sum(1 for r in metrics['detailed_results'] if r['top3_correct']) / total_cases,
                'overall_top4_accuracy': sum(1 for r in metrics['detailed_results'] if r['top4_correct']) / total_cases,
                'overall_pause_rate': sum(1 for r in metrics['detailed_results'] if r['paused']) / total_cases,
                'overall_error_rate': sum(1 for r in metrics['detailed_results'] if r['error']) / total_cases,
                'avg_confidence': sum(r['confidence'] for r in metrics['detailed_results']) / total_cases,
                'avg_execution_time_ms': sum(r['execution_time_ms'] for r in metrics['detailed_results']) / total_cases
            }

        return metrics

    def generate_tables(self, metrics: Dict, output_dir: str, prefix: str = ''):
        """Generate markdown and CSV tables from metrics."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Table 1: Metrics by Variant
        variant_df = pd.DataFrame([
            {
                'Variant': variant,
                'Total Cases': data['total'],
                'Top-1 Acc (%)': f"{data['accuracy']*100:.1f}",
                'Top-3 Acc (%)': f"{data['top3_accuracy']*100:.1f}",
                'Top-4 Acc (%)': f"{data['top4_accuracy']*100:.1f}",
                'Pause Rate (%)': f"{data['pause_rate']*100:.1f}",
                'Avg Confidence': f"{data['avg_confidence']:.2f}",
                'Avg Time (s)': f"{data['avg_execution_time_ms']/1000:.1f}",
                'Error Rate (%)': f"{data['error_rate']*100:.1f}"
            }
            for variant, data in metrics['by_variant'].items()
        ])

        # Save CSV
        variant_df.to_csv(output_path / f'{prefix}metrics_by_variant.csv', index=False)

        # Save Markdown
        with open(output_path / f'{prefix}metrics_by_variant.md', 'w') as f:
            f.write("# Evaluation Metrics by Context Variant\n\n")
            f.write(variant_df.to_markdown(index=False))
            f.write("\n")

        print(f"✓ Saved metrics table to {output_path}/{prefix}metrics_by_variant.csv")

        # Table 2: Detailed Results
        detailed_df = pd.DataFrame(metrics['detailed_results'])
        detailed_df.to_csv(output_path / f'{prefix}detailed_results.csv', index=False)
        print(f"✓ Saved detailed results to {output_path}/{prefix}detailed_results.csv")

        return variant_df

    def generate_plots(self, metrics: Dict, output_dir: str, prefix: str = ''):
        """Generate visualization plots."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        variants = list(metrics['by_variant'].keys())

        # Plot 1: Accuracy by Variant (Top-1, Top-3, Top-4 grouped bars)
        fig, ax = plt.subplots(figsize=(12, 7))
        top1_accs = [metrics['by_variant'][v]['accuracy'] * 100 for v in variants]
        top3_accs = [metrics['by_variant'][v]['top3_accuracy'] * 100 for v in variants]
        top4_accs = [metrics['by_variant'][v]['top4_accuracy'] * 100 for v in variants]

        x = np.arange(len(variants))
        width = 0.25

        bars1 = ax.bar(x - width, top1_accs, width, label='Top-1', alpha=0.8, color='#e74c3c', edgecolor='black')
        bars2 = ax.bar(x, top3_accs, width, label='Top-3', alpha=0.8, color='#f39c12', edgecolor='black')
        bars3 = ax.bar(x + width, top4_accs, width, label='Top-4', alpha=0.8, color='#27ae60', edgecolor='black')

        ax.set_xlabel('Context Variant', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Diagnostic Accuracy by Context Variant (Top-1 / Top-3 / Top-4)', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        for bar_group in [bars1, bars2, bars3]:
            for bar in bar_group:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{height:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path / f'{prefix}accuracy_by_variant.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved plot to {output_path}/{prefix}accuracy_by_variant.png")

        # Plot 2: Pause Rate by Variant
        fig, ax = plt.subplots(figsize=(10, 6))
        pause_rates = [metrics['by_variant'][v]['pause_rate'] * 100 for v in variants]

        # Expected pause rates (incomplete variants should pause more)
        expected_pauses = {'original': 0, 'history_only': 80, 'image_only': 80,
                          'exam_only': 80, 'exam_restricted': 90}
        expected = [expected_pauses.get(v, 50) for v in variants]

        x = range(len(variants))
        width = 0.35
        ax.bar([i - width/2 for i in x], pause_rates, width, label='Actual', alpha=0.7, color='steelblue')
        ax.bar([i + width/2 for i in x], expected, width, label='Expected', alpha=0.7, color='lightcoral')

        ax.set_xlabel('Context Variant', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pause Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Agentic Pause Rate by Context Variant', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=45, ha='right')
        ax.set_ylim(0, 100)
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path / f'{prefix}pause_rate_by_variant.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved plot to {output_path}/{prefix}pause_rate_by_variant.png")

        # Plot 3: Confidence vs Accuracy Scatter
        fig, ax = plt.subplots(figsize=(10, 6))

        for variant in variants:
            data = metrics['by_variant'][variant]['diagnoses']
            confidences = [d['confidence'] for d in data]
            corrects = [1 if d['correct'] else 0 for d in data]
            ax.scatter(confidences, corrects, alpha=0.6, label=variant, s=100)

        ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Correct (1) vs Incorrect (0)', fontsize=12, fontweight='bold')
        ax.set_title('Confidence vs Correctness', fontsize=14, fontweight='bold')
        ax.set_ylim(-0.1, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / f'{prefix}confidence_vs_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved plot to {output_path}/{prefix}confidence_vs_accuracy.png")

        # Plot 4: Execution Time by Variant
        fig, ax = plt.subplots(figsize=(10, 6))
        exec_times = [metrics['by_variant'][v]['avg_execution_time_ms'] / 1000 for v in variants]

        ax.bar(variants, exec_times, color='teal', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Context Variant', fontsize=12, fontweight='bold')
        ax.set_ylabel('Avg Execution Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Average Execution Time by Context Variant', fontsize=14, fontweight='bold')

        for i, v in enumerate(exec_times):
            ax.text(i, v + 1, f'{v:.1f}s', ha='center', fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path / f'{prefix}execution_time_by_variant.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved plot to {output_path}/{prefix}execution_time_by_variant.png")

    def generate_html_report(self, metrics: Dict, output_path: str):
        """Generate comprehensive HTML report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>MedGemma Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metric-box {{
            background: white;
            padding: 20px;
            margin: 15px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-row {{ display: flex; justify-content: space-between; margin: 10px 0; }}
        .metric-label {{ font-weight: bold; color: #7f8c8d; }}
        .metric-value {{ color: #2c3e50; font-size: 1.2em; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            background: white;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{ background: #3498db; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ecf0f1; }}
        tr:hover {{ background: #f8f9fa; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }}
        .success {{ color: green; font-weight: bold; }}
        .warning {{ color: orange; font-weight: bold; }}
        .error {{ color: red; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>📊 MedGemma Evaluation Analysis Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Model:</strong> {metrics.get('metadata', {}).get('agent_model', 'Unknown')}</p>

    <div class="metric-box">
        <h2>🎯 Overall Performance</h2>
        <div class="metric-row">
            <span class="metric-label">Total Evaluations:</span>
            <span class="metric-value">{metrics['overall'].get('total_evaluations', 0)}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Top-1 Accuracy:</span>
            <span class="metric-value {'success' if metrics['overall'].get('overall_accuracy', 0) > 0.7 else 'warning' if metrics['overall'].get('overall_accuracy', 0) > 0.5 else 'error'}">
                {metrics['overall'].get('overall_accuracy', 0)*100:.1f}%
            </span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Top-3 Accuracy:</span>
            <span class="metric-value {'success' if metrics['overall'].get('overall_top3_accuracy', 0) > 0.7 else 'warning' if metrics['overall'].get('overall_top3_accuracy', 0) > 0.5 else 'error'}">
                {metrics['overall'].get('overall_top3_accuracy', 0)*100:.1f}%
            </span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Top-4 Accuracy:</span>
            <span class="metric-value {'success' if metrics['overall'].get('overall_top4_accuracy', 0) > 0.7 else 'warning' if metrics['overall'].get('overall_top4_accuracy', 0) > 0.5 else 'error'}">
                {metrics['overall'].get('overall_top4_accuracy', 0)*100:.1f}%
            </span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Pause Rate:</span>
            <span class="metric-value">{metrics['overall'].get('overall_pause_rate', 0)*100:.1f}%</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Avg Confidence:</span>
            <span class="metric-value">{metrics['overall'].get('avg_confidence', 0):.2f}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Avg Time:</span>
            <span class="metric-value">{metrics['overall'].get('avg_execution_time_ms', 0)/1000:.1f}s</span>
        </div>
    </div>

    <h2>📈 Visualizations</h2>
    <img src="accuracy_by_variant.png" alt="Accuracy by Variant">
    <img src="pause_rate_by_variant.png" alt="Pause Rate by Variant">
    <img src="confidence_vs_accuracy.png" alt="Confidence vs Accuracy">
    <img src="execution_time_by_variant.png" alt="Execution Time">

    <h2>📋 Metrics by Variant</h2>
    <table>
        <tr>
            <th>Variant</th>
            <th>Top-1 Acc</th>
            <th>Top-3 Acc</th>
            <th>Top-4 Acc</th>
            <th>Pause Rate</th>
            <th>Avg Confidence</th>
            <th>Avg Time (s)</th>
        </tr>
"""

        for variant, data in metrics['by_variant'].items():
            accuracy_class = 'success' if data['accuracy'] > 0.7 else 'warning' if data['accuracy'] > 0.5 else 'error'
            top3_class = 'success' if data['top3_accuracy'] > 0.7 else 'warning' if data['top3_accuracy'] > 0.5 else 'error'
            top4_class = 'success' if data['top4_accuracy'] > 0.7 else 'warning' if data['top4_accuracy'] > 0.5 else 'error'
            html += f"""
        <tr>
            <td><strong>{variant}</strong></td>
            <td class="{accuracy_class}">{data['accuracy']*100:.1f}%</td>
            <td class="{top3_class}">{data['top3_accuracy']*100:.1f}%</td>
            <td class="{top4_class}">{data['top4_accuracy']*100:.1f}%</td>
            <td>{data['pause_rate']*100:.1f}%</td>
            <td>{data['avg_confidence']:.2f}</td>
            <td>{data['avg_execution_time_ms']/1000:.1f}s</td>
        </tr>
"""

        html += """
    </table>

    <h2>✅ Key Findings</h2>
    <div class="metric-box">
        <ul>
"""

        # Generate findings
        original_acc = metrics['by_variant']['original']['accuracy']
        original_pause = metrics['by_variant']['original']['pause_rate']

        original_top3 = metrics['by_variant']['original']['top3_accuracy']
        original_top4 = metrics['by_variant']['original']['top4_accuracy']

        html += f"<li><strong>Complete Cases (Original):</strong> Top-1: {original_acc*100:.1f}%, "
        html += f"Top-3: {original_top3*100:.1f}%, Top-4: {original_top4*100:.1f}% accuracy, "
        html += f"{original_pause*100:.1f}% false positive pause rate</li>"

        incomplete_variants = ['history_only', 'image_only', 'exam_only', 'exam_restricted']
        avg_incomplete_pause = sum(metrics['by_variant'][v]['pause_rate'] for v in incomplete_variants) / len(incomplete_variants)
        html += f"<li><strong>Incomplete Cases:</strong> {avg_incomplete_pause*100:.1f}% average pause rate (safety mechanism working)</li>"

        html += f"<li><strong>Robustness:</strong> Top-1: {metrics['overall']['overall_accuracy']*100:.1f}%, "
        html += f"Top-3: {metrics['overall']['overall_top3_accuracy']*100:.1f}%, "
        html += f"Top-4: {metrics['overall']['overall_top4_accuracy']*100:.1f}% accuracy across all context levels</li>"

        html += """
        </ul>
    </div>

    <p style="margin-top: 40px; text-align: center; color: #95a5a6;">
        Generated by MedGemma Evaluation Analysis Script
    </p>
</body>
</html>
"""

        with open(output_path, 'w') as f:
            f.write(html)

        print(f"✓ Saved HTML report to {output_path}")


    def generate_side_by_side_csv(self, output_dir: str, prefix: str = ''):
        """
        Generate a CSV with all 5 variants side-by-side per case for manual verification.

        Format: Case_ID | Ground_Truth | original | original_full_response | history_only | ... | exam_restricted_full_response
        - If model paused → "PAUSED"
        - If error → "ERROR"
        - Otherwise → extracted diagnosis text
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        variants = ['original', 'history_only', 'image_only', 'exam_only', 'exam_restricted']

        # Group results by case_id
        cases = {}
        for result in self.results_data:
            case_id = result['case_id']
            variant = result['variant']

            # Extract base case_id (e.g., "JDCR_01_01_23_original" → "01_01_23")
            case_match = re.search(r'(\d{2}_\d{2}_\d{2})', case_id)
            if not case_match:
                continue
            case_num = case_match.group(1)

            if case_num not in cases:
                cases[case_num] = {}

            response_text = result.get('response_text', '')
            paused = result.get('agentic_pause_triggered', False)
            error = result.get('error') is not None

            # Determine diagnosis cell value
            if error:
                diagnosis_cell = "ERROR"
                all_diag_cell = "ERROR"
            elif paused:
                diagnosis_cell = "PAUSED"
                all_diag_cell = "PAUSED"
            else:
                all_diags = self.extract_all_diagnoses(response_text)
                pred = all_diags[0][0] if all_diags else "Unknown"
                if pred != "Unknown":
                    diagnosis_cell = pred
                elif 'SOAP' in response_text or 'Assessment' in response_text:
                    # Has SOAP structure but regex couldn't extract — show first 200 chars
                    diagnosis_cell = response_text[:200]
                elif response_text:
                    # Response exists but no SOAP note (incomplete workflow)
                    diagnosis_cell = "INCOMPLETE (no SOAP)"
                else:
                    diagnosis_cell = "Unknown"
                # Format all diagnoses as numbered list
                all_diag_cell = " | ".join(f"{i+1}. {d[0]}" for i, d in enumerate(all_diags)) if all_diags else diagnosis_cell

            cases[case_num][variant] = {
                'diagnosis': diagnosis_cell,
                'all_diagnoses': all_diag_cell,
                'full_response': response_text
            }

        # Build CSV rows
        headers = ['Case_ID', 'Ground_Truth']
        for v in variants:
            headers.append(v)
            headers.append(f'{v}_all_diagnoses')
            headers.append(f'{v}_full_response')

        rows = []
        for case_num in sorted(cases.keys()):
            true_diagnosis = self.groundtruth.get(case_num, "Unknown")
            row = [case_num, true_diagnosis]
            for v in variants:
                if v in cases[case_num]:
                    row.append(cases[case_num][v]['diagnosis'])
                    row.append(cases[case_num][v]['all_diagnoses'])
                    row.append(cases[case_num][v]['full_response'])
                else:
                    row.append("N/A")
                    row.append("")
                    row.append("")
            rows.append(row)

        # Write CSV
        csv_path = output_path / f'{prefix}side_by_side_results.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

        print(f"✓ Saved side-by-side results to {csv_path}")
        print(f"  {len(rows)} cases x {len(variants)} variants")


def main():
    parser = argparse.ArgumentParser(description='Analyze MedGemma evaluation results')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to evaluation results JSON file')
    parser.add_argument('--groundtruth', type=str, required=True,
                       help='Path to ground truth CSV file')
    parser.add_argument('--output-dir', type=str, default='logs/analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--prefix', type=str, default='',
                       help='Prefix for output file names (e.g., nejm_medgemma-27b-it_without_options)')

    args = parser.parse_args()

    print("="*70)
    print("  MedGemma Evaluation Analysis")
    print("="*70)
    print()

    # Initialize analyzer
    analyzer = EvaluationAnalyzer(args.groundtruth)

    # Load results
    analyzer.load_evaluation_results(args.results)

    # Analyze
    print("\nAnalyzing results...")
    metrics = analyzer.analyze_all_results()

    # Build file prefix
    prefix = args.prefix + '_' if args.prefix else ''

    # Generate outputs
    print("\nGenerating tables...")
    analyzer.generate_tables(metrics, args.output_dir, prefix)

    print("\nGenerating side-by-side CSV...")
    analyzer.generate_side_by_side_csv(args.output_dir, prefix)

    print("\nGenerating plots...")
    analyzer.generate_plots(metrics, args.output_dir, prefix)

    print("\nGenerating HTML report...")
    html_path = Path(args.output_dir) / f'{prefix}evaluation_report.html'
    analyzer.generate_html_report(metrics, str(html_path))

    print("\n" + "="*70)
    print("  ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Open HTML report: {html_path}")
    print()

    # Print summary
    print("SUMMARY:")
    print(f"  Total Evaluations: {metrics['overall']['total_evaluations']}")
    print(f"  Top-1 Accuracy:    {metrics['overall']['overall_accuracy']*100:.1f}%")
    print(f"  Top-3 Accuracy:    {metrics['overall']['overall_top3_accuracy']*100:.1f}%")
    print(f"  Top-4 Accuracy:    {metrics['overall']['overall_top4_accuracy']*100:.1f}%")
    print(f"  Pause Rate:        {metrics['overall']['overall_pause_rate']*100:.1f}%")
    print(f"  Avg Confidence:    {metrics['overall']['avg_confidence']:.2f}")
    print()


if __name__ == "__main__":
    main()
