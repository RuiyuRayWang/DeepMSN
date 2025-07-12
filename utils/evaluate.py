import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support, 
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_multilabel_model(y_true, y_pred_probs, threshold=0.5, topic_names=None):
    """
    Comprehensive evaluation for multi-label classification
    
    Args:
        y_true: Ground truth labels (n_samples, n_topics)
        y_pred_probs: Predicted probabilities (n_samples, n_topics)
        threshold: Decision threshold for binary predictions
        topic_names: List of topic names (optional)
    """
    
    n_samples, n_topics = y_true.shape
    
    if topic_names is None:
        topic_names = [f'Topic_{i+1}' for i in range(n_topics)]
    
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred_probs > threshold).astype(int)
    
    # Initialize results storage
    results = {
        'topic': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'auroc': [],
        'aupr': [],
        'support': [],  # Number of positive samples
        'predictions': [],  # Number of positive predictions
        'true_positives': [],
        'false_positives': [],
        'false_negatives': [],
        'true_negatives': []
    }
    
    print("="*80)
    print("PER-TOPIC EVALUATION RESULTS")
    print("="*80)
    print(f"{'Topic':<12} {'Precision':<10} {'Recall':<8} {'F1':<8} {'AUROC':<8} {'AUPR':<8} {'Support':<8} {'Preds':<6}")
    print("-"*80)
    
    # Evaluate each topic individually
    for topic_idx in range(n_topics):
        topic_name = topic_names[topic_idx]
        
        # Get predictions and targets for this topic
        y_true_topic = y_true[:, topic_idx]
        y_pred_topic = y_pred_binary[:, topic_idx]
        y_prob_topic = y_pred_probs[:, topic_idx]
        
        # Calculate confusion matrix components
        tp = np.sum((y_true_topic == 1) & (y_pred_topic == 1))
        fp = np.sum((y_true_topic == 0) & (y_pred_topic == 1))
        fn = np.sum((y_true_topic == 1) & (y_pred_topic == 0))
        tn = np.sum((y_true_topic == 0) & (y_pred_topic == 0))
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate AUROC and AUPR
        try:
            auroc = roc_auc_score(y_true_topic, y_prob_topic)
        except:
            auroc = 0.0  # If all samples are one class
            
        try:
            aupr = average_precision_score(y_true_topic, y_prob_topic)
        except:
            aupr = 0.0
        
        support = np.sum(y_true_topic)
        predictions = np.sum(y_pred_topic)
        
        # Store results
        results['topic'].append(topic_name)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1_score'].append(f1)
        results['auroc'].append(auroc)
        results['aupr'].append(aupr)
        results['support'].append(support)
        results['predictions'].append(predictions)
        results['true_positives'].append(tp)
        results['false_positives'].append(fp)
        results['false_negatives'].append(fn)
        results['true_negatives'].append(tn)
        
        # Print results
        print(f"{topic_name:<12} {precision:<10.3f} {recall:<8.3f} {f1:<8.3f} {auroc:<8.3f} {aupr:<8.3f} {support:<8} {predictions:<6}")
    
    return pd.DataFrame(results)

# # Usage with your data:
# results_df = evaluate_multilabel_model(
#     y_true=y_test_np,
#     y_pred_probs=y_pred_np,
#     threshold=0.5,
#     topic_names=[f'Topic_{i}' for i in range(y_test_np.shape[1])]
# )

def plot_per_topic_metrics(results_df, save_path=None):
    """Plot per-topic performance metrics"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # F1 Score
    axes[0, 0].bar(range(len(results_df)), results_df['f1_score'])
    axes[0, 0].set_title('F1 Score per Topic')
    axes[0, 0].set_xlabel('Topic Index')
    axes[0, 0].set_ylabel('F1 Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Precision vs Recall
    axes[0, 1].scatter(results_df['recall'], results_df['precision'], alpha=0.7)
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision vs Recall per Topic')
    
    # AUROC
    axes[1, 0].bar(range(len(results_df)), results_df['auroc'])
    axes[1, 0].set_title('AUROC per Topic')
    axes[1, 0].set_xlabel('Topic Index')
    axes[1, 0].set_ylabel('AUROC')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Support vs Performance
    axes[1, 1].scatter(results_df['support'], results_df['f1_score'], alpha=0.7)
    axes[1, 1].set_xlabel('Support (# Positive Samples)')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('F1 Score vs Support')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_topic_difficulty(results_df, n_top=10):
    """Analyze which topics are hardest to predict"""
    
    # Sort by F1 score
    sorted_df = results_df.sort_values('f1_score', ascending=False)
    
    print("\n" + "="*60)
    print("TOPIC DIFFICULTY ANALYSIS")
    print("="*60)
    
    print(f"\n🏆 TOP {n_top} BEST PERFORMING TOPICS:")
    print(sorted_df[['topic', 'f1_score', 'precision', 'recall', 'support']].head(n_top).to_string(index=False))
    
    print(f"\n❌ TOP {n_top} WORST PERFORMING TOPICS:")
    print(sorted_df[['topic', 'f1_score', 'precision', 'recall', 'support']].tail(n_top).to_string(index=False))
    
    # Analyze correlation between support and performance
    correlation = results_df['support'].corr(results_df['f1_score'])
    print(f"\n📊 Correlation between Support and F1 Score: {correlation:.3f}")
    
    return sorted_df

def calculate_overall_metrics(y_true, y_pred_probs, threshold=0.5):
    """Calculate overall multi-label metrics"""
    
    y_pred_binary = (y_pred_probs > threshold).astype(int)
    
    # Micro-averaged metrics (treat each sample-label pair as separate)
    micro_precision = np.sum(y_true * y_pred_binary) / np.sum(y_pred_binary) if np.sum(y_pred_binary) > 0 else 0
    micro_recall = np.sum(y_true * y_pred_binary) / np.sum(y_true) if np.sum(y_true) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    # Macro-averaged metrics (average across topics)
    per_topic_precision = []
    per_topic_recall = []
    per_topic_f1 = []
    
    for topic_idx in range(y_true.shape[1]):
        y_true_topic = y_true[:, topic_idx]
        y_pred_topic = y_pred_binary[:, topic_idx]
        
        tp = np.sum((y_true_topic == 1) & (y_pred_topic == 1))
        fp = np.sum((y_true_topic == 0) & (y_pred_topic == 1))
        fn = np.sum((y_true_topic == 1) & (y_pred_topic == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_topic_precision.append(precision)
        per_topic_recall.append(recall)
        per_topic_f1.append(f1)
    
    macro_precision = np.mean(per_topic_precision)
    macro_recall = np.mean(per_topic_recall)
    macro_f1 = np.mean(per_topic_f1)
    
    # Additional metrics
    hamming_accuracy = np.mean(y_true == y_pred_binary)
    exact_match_ratio = np.mean(np.all(y_true == y_pred_binary, axis=1))
    
    print("\n" + "="*50)
    print("OVERALL MULTI-LABEL METRICS")
    print("="*50)
    print(f"Micro-averaged Precision: {micro_precision:.4f}")
    print(f"Micro-averaged Recall:    {micro_recall:.4f}")
    print(f"Micro-averaged F1:        {micro_f1:.4f}")
    print("-"*50)
    print(f"Macro-averaged Precision: {macro_precision:.4f}")
    print(f"Macro-averaged Recall:    {macro_recall:.4f}")
    print(f"Macro-averaged F1:        {macro_f1:.4f}")
    print("-"*50)
    print(f"Hamming Accuracy:         {hamming_accuracy:.4f}")
    print(f"Exact Match Ratio:        {exact_match_ratio:.4f}")
    
    return {
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'hamming_accuracy': hamming_accuracy,
        'exact_match_ratio': exact_match_ratio
    }

# # Run the complete analysis
# print("Running comprehensive multi-label evaluation...")

# # Per-topic evaluation
# results_df = evaluate_multilabel_model(y_test_np, y_pred_np, threshold=0.5)

# # Overall metrics
# overall_metrics = calculate_overall_metrics(y_test_np, y_pred_np, threshold=0.5)

# # Topic difficulty analysis
# sorted_results = analyze_topic_difficulty(results_df)

# # Plot results
# plot_per_topic_metrics(results_df, save_path='topic_performance.png')

def optimize_thresholds_per_topic(y_true, y_pred_probs, metric='f1'):
    """Find optimal threshold for each topic"""
    
    n_topics = y_true.shape[1]
    optimal_thresholds = []
    optimal_scores = []
    
    print("\n" + "="*60)
    print("THRESHOLD OPTIMIZATION PER TOPIC")
    print("="*60)
    print(f"{'Topic':<8} {'Best Thresh':<12} {'Best F1':<10} {'Precision':<10} {'Recall':<8}")
    print("-"*60)
    
    for topic_idx in range(n_topics):
        y_true_topic = y_true[:, topic_idx]
        y_prob_topic = y_pred_probs[:, topic_idx]
        
        best_threshold = 0.5
        best_score = 0.0
        best_precision = 0.0
        best_recall = 0.0
        
        # Test different thresholds
        for threshold in np.arange(0.1, 0.95, 0.05):
            y_pred_topic = (y_prob_topic > threshold).astype(int)
            
            tp = np.sum((y_true_topic == 1) & (y_pred_topic == 1))
            fp = np.sum((y_true_topic == 0) & (y_pred_topic == 1))
            fn = np.sum((y_true_topic == 1) & (y_pred_topic == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            if f1 > best_score:
                best_score = f1
                best_threshold = threshold
                best_precision = precision
                best_recall = recall
        
        optimal_thresholds.append(best_threshold)
        optimal_scores.append(best_score)
        
        print(f"Topic_{topic_idx+1:<3} {best_threshold:<12.2f} {best_score:<10.3f} {best_precision:<10.3f} {best_recall:<8.3f}")
    
    return optimal_thresholds, optimal_scores

# # Find optimal thresholds
# optimal_thresholds, optimal_f1_scores = optimize_thresholds_per_topic(y_test_np, y_pred_np)

# # Evaluate with optimal thresholds
# print("\n" + "="*60)
# print("PERFORMANCE WITH OPTIMIZED THRESHOLDS")
# print("="*60)

# # Apply different thresholds per topic
# y_pred_optimized = np.zeros_like(y_pred_np)
# for topic_idx in range(y_test_np.shape[1]):
#     threshold = optimal_thresholds[topic_idx]
#     y_pred_optimized[:, topic_idx] = (y_pred_np[:, topic_idx] > threshold).astype(int)

# # Calculate metrics with optimized thresholds
# optimized_overall = calculate_overall_metrics(y_test_np, y_pred_np, threshold=None)  # Custom threshold application