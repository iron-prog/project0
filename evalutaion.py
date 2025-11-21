"""
evaluation.py - RAG Evaluation Module using RAGAS
==================================================
This module evaluates the RAG system's performance using RAGAS metrics,
focusing on "Faithfulness" - how well the answers stick to the provided context.

RAGAS Metrics:
- Faithfulness: Does the answer only contain info from the context?
- Answer Relevancy: Is the answer relevant to the question?
- Context Precision: Are the retrieved docs relevant?
- Context Recall: Were all necessary docs retrieved?

Author: Senior AI Engineer
"""

import os
import json
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# RAGAS for evaluation
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# Datasets library for RAGAS compatibility
from datasets import Dataset

# LangChain for LLM
from langchain_openai import ChatOpenAI

# Import our modules
from src.chain import RAGChain, create_rag_chain
from src.retrieval import RAGRetriever

# Load environment variables
load_dotenv()


# ============================================
# Evaluation Data Structures
# ============================================

class EvaluationSample:
    """
    A single evaluation sample containing:
    - question: The user's question
    - answer: The RAG system's answer
    - contexts: Retrieved context documents
    - ground_truth: (Optional) The correct answer for comparison
    """
    
    def __init__(
        self,
        question: str,
        answer: str = None,
        contexts: List[str] = None,
        ground_truth: str = None
    ):
        self.question = question
        self.answer = answer
        self.contexts = contexts or []
        self.ground_truth = ground_truth
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for RAGAS."""
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "ground_truth": self.ground_truth
        }


# ============================================
# RAG Evaluator Class
# ============================================

class RAGEvaluator:
    """
    Evaluator class for assessing RAG system performance.
    
    Uses RAGAS metrics to measure:
    - Faithfulness: How faithful is the answer to the context?
    - Answer Relevancy: How relevant is the answer to the question?
    - Context Precision: How precise are the retrieved contexts?
    - Context Recall: How complete is the context retrieval?
    """
    
    def __init__(
        self,
        rag_chain: RAGChain = None,
        metrics: List = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            rag_chain: The RAG chain to evaluate
            metrics: List of RAGAS metrics to use (default: all)
        """
        self.rag_chain = rag_chain
        
        # Default to all standard metrics
        # Faithfulness is the most important for "Clean RAG"
        self.metrics = metrics or [
            faithfulness,      # CRITICAL: Does answer stick to context?
            answer_relevancy,  # Is answer relevant to question?
            context_precision, # Are retrieved docs relevant?
            context_recall,    # Were all necessary docs retrieved?
        ]
        
        print(f"✅ RAG Evaluator initialized")
        print(f"   Metrics: {[m.name for m in self.metrics]}")
    
    def generate_answer(self, question: str) -> EvaluationSample:
        """
        Generate an answer using the RAG chain and capture all data.
        
        Args:
            question: The question to answer
            
        Returns:
            EvaluationSample with question, answer, and contexts
        """
        if not self.rag_chain:
            raise ValueError("RAG chain not initialized")
        
        # Get response from RAG chain
        response = self.rag_chain.invoke(question)
        
        # Extract contexts from source documents
        contexts = [doc.page_content for doc in response.sources]
        
        return EvaluationSample(
            question=question,
            answer=response.answer,
            contexts=contexts
        )
    
    def evaluate_samples(
        self,
        samples: List[EvaluationSample]
    ) -> Dict[str, Any]:
        """
        Evaluate a list of samples using RAGAS.
        
        Args:
            samples: List of EvaluationSample objects
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"🔍 Evaluating {len(samples)} samples...")
        
        # Convert samples to RAGAS format
        eval_data = {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
            "ground_truth": [s.ground_truth or "" for s in samples]
        }
        
        # Create HuggingFace Dataset
        dataset = Dataset.from_dict(eval_data)
        
        # Run RAGAS evaluation
        results = evaluate(
            dataset=dataset,
            metrics=self.metrics
        )
        
        return results
    
    def evaluate_questions(
        self,
        questions: List[str],
        ground_truths: List[str] = None
    ) -> Dict[str, Any]:
        """
        Convenience method: Generate answers and evaluate in one step.
        
        Args:
            questions: List of questions to evaluate
            ground_truths: Optional list of correct answers
            
        Returns:
            Evaluation results
        """
        print(f"📝 Processing {len(questions)} questions...")
        
        samples = []
        for i, question in enumerate(questions):
            print(f"   [{i+1}/{len(questions)}] {question[:50]}...")
            
            # Generate answer
            sample = self.generate_answer(question)
            
            # Add ground truth if provided
            if ground_truths and i < len(ground_truths):
                sample.ground_truth = ground_truths[i]
            
            samples.append(sample)
        
        # Evaluate all samples
        return self.evaluate_samples(samples)
    
    def evaluate_faithfulness_only(
        self,
        questions: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate only faithfulness - the key metric for "Clean RAG".
        
        Faithfulness measures whether the answer contains ONLY
        information from the provided context (no hallucination).
        
        Args:
            questions: List of questions to evaluate
            
        Returns:
            Faithfulness scores
        """
        # Temporarily set only faithfulness metric
        original_metrics = self.metrics
        self.metrics = [faithfulness]
        
        try:
            results = self.evaluate_questions(questions)
            return results
        finally:
            # Restore original metrics
            self.metrics = original_metrics


# ============================================
# Evaluation Report Generator
# ============================================

def generate_evaluation_report(
    results: Dict[str, Any],
    output_path: str = None
) -> str:
    """
    Generate a human-readable evaluation report.
    
    Args:
        results: RAGAS evaluation results
        output_path: Optional path to save the report
        
    Returns:
        Formatted report string
    """
    report_lines = [
        "=" * 60,
        "RAG EVALUATION REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        ""
    ]
    
    # Overall scores
    report_lines.append("OVERALL SCORES")
    report_lines.append("-" * 40)
    
    for metric, score in results.items():
        if isinstance(score, (int, float)):
            # Format score as percentage
            report_lines.append(f"  {metric}: {score:.2%}")
    
    report_lines.append("")
    
    # Interpretation
    report_lines.append("INTERPRETATION")
    report_lines.append("-" * 40)
    
    # Faithfulness interpretation
    if "faithfulness" in results:
        faith_score = results["faithfulness"]
        if faith_score >= 0.9:
            interpretation = "Excellent - Answers are highly faithful to context"
        elif faith_score >= 0.7:
            interpretation = "Good - Most answers stick to context"
        elif faith_score >= 0.5:
            interpretation = "Fair - Some hallucination detected"
        else:
            interpretation = "Poor - Significant hallucination issues"
        
        report_lines.append(f"  Faithfulness: {interpretation}")
    
    # Answer relevancy interpretation
    if "answer_relevancy" in results:
        rel_score = results["answer_relevancy"]
        if rel_score >= 0.9:
            interpretation = "Excellent - Answers are highly relevant"
        elif rel_score >= 0.7:
            interpretation = "Good - Answers are mostly relevant"
        else:
            interpretation = "Needs improvement - Answers may miss the point"
        
        report_lines.append(f"  Answer Relevancy: {interpretation}")
    
    report_lines.append("")
    report_lines.append("=" * 60)
    
    report = "\n".join(report_lines)
    
    # Save to file if path provided
    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"📄 Report saved to: {output_path}")
    
    return report


# ============================================
# Sample Evaluation Test Set
# ============================================

SAMPLE_TEST_QUESTIONS = [
    "What is the main topic of this document?",
    "What are the key findings or conclusions?",
    "Are there any numerical values or statistics mentioned?",
    "What data is presented in the tables?",
    "Who are the main entities or people mentioned?",
]


# ============================================
# Main Evaluation Script
# ============================================

def run_evaluation(
    questions: List[str] = None,
    ground_truths: List[str] = None,
    output_path: str = "evaluation_report.txt"
) -> Dict[str, Any]:
    """
    Run a full evaluation of the RAG system.
    
    Args:
        questions: Questions to evaluate (default: sample questions)
        ground_truths: Optional ground truth answers
        output_path: Path to save the report
        
    Returns:
        Evaluation results
    """
    print("🚀 Starting RAG Evaluation")
    print("=" * 50)
    
    # Use sample questions if none provided
    questions = questions or SAMPLE_TEST_QUESTIONS
    
    # Initialize RAG chain
    print("\n1. Initializing RAG chain...")
    rag_chain = create_rag_chain()
    
    # Initialize evaluator
    print("\n2. Initializing evaluator...")
    evaluator = RAGEvaluator(rag_chain=rag_chain)
    
    # Run evaluation
    print("\n3. Running evaluation...")
    results = evaluator.evaluate_questions(
        questions=questions,
        ground_truths=ground_truths
    )
    
    # Generate report
    print("\n4. Generating report...")
    report = generate_evaluation_report(results, output_path)
    
    # Print report
    print("\n" + report)
    
    return results


# ============================================
# Faithfulness-Only Quick Evaluation
# ============================================

def evaluate_faithfulness(
    questions: List[str] = None
) -> float:
    """
    Quick evaluation of faithfulness only.
    
    This is the key metric for "Clean RAG" - ensuring
    answers don't contain hallucinated information.
    
    Args:
        questions: Questions to test
        
    Returns:
        Faithfulness score (0-1)
    """
    questions = questions or SAMPLE_TEST_QUESTIONS[:3]  # Use fewer for speed
    
    print("🎯 Evaluating Faithfulness...")
    
    rag_chain = create_rag_chain()
    evaluator = RAGEvaluator(rag_chain=rag_chain)
    
    results = evaluator.evaluate_faithfulness_only(questions)
    
    faith_score = results.get("faithfulness", 0)
    
    print(f"\n📊 Faithfulness Score: {faith_score:.2%}")
    
    if faith_score >= 0.9:
        print("✅ Excellent! Answers are faithful to the context.")
    elif faith_score >= 0.7:
        print("👍 Good. Most answers stick to the context.")
    else:
        print("⚠️ Warning: Potential hallucination detected.")
    
    return faith_score


# ============================================
# Entry Point
# ============================================

if __name__ == "__main__":
    import sys
    
    print("""
╔══════════════════════════════════════════════════════════╗
║         Clean RAG Evaluation Script (RAGAS)              ║
╠══════════════════════════════════════════════════════════╣
║  This script evaluates the RAG system using RAGAS        ║
║  metrics, focusing on Faithfulness.                      ║
║                                                          ║
║  Prerequisites:                                          ║
║  1. Documents must be ingested first (run ingestion.py)  ║
║  2. OPENAI_API_KEY must be set in .env                   ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            # Quick faithfulness-only evaluation
            evaluate_faithfulness()
        elif sys.argv[1] == "--full":
            # Full evaluation with all metrics
            run_evaluation()
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Usage: python evaluation.py [--quick|--full]")
    else:
        # Default: run full evaluation
        print("Running full evaluation (use --quick for faster faithfulness-only test)")
        run_evaluation()