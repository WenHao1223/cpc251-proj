import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluate a model using multiple metrics and classification report
    """
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1 Score': f1_score(y_true, y_pred, average='weighted')
    }
    
    # Generate classification report
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    return metrics, class_report

def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Plot confusion matrix for a model
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def compare_models(y_true, model1_pred, model2_pred, model3_pred, 
                  model1_name="Model 1", model2_name="Model 2", model3_name="Model 3"):
    """
    Compare three models using various metrics and classification reports
    """
    # Calculate metrics and classification reports for each model
    metrics1, report1 = evaluate_model(y_true, model1_pred, model1_name)
    metrics2, report2 = evaluate_model(y_true, model2_pred, model2_name)
    metrics3, report3 = evaluate_model(y_true, model3_pred, model3_name)
    
    # Create comparison DataFrame for overall metrics
    comparison_df = pd.DataFrame([metrics1, metrics2, metrics3])
    
    # Convert classification reports to DataFrames
    report1_df = pd.DataFrame(report1).transpose()
    report2_df = pd.DataFrame(report2).transpose()
    report3_df = pd.DataFrame(report3).transpose()
    
    # Plot confusion matrices
    plot_confusion_matrix(y_true, model1_pred, model1_name)
    plot_confusion_matrix(y_true, model2_pred, model2_name)
    plot_confusion_matrix(y_true, model3_pred, model3_name)
    
    # Plot comparison bar chart
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    plt.bar(x - width, comparison_df[metrics_to_plot].iloc[0], width, label=model1_name)
    plt.bar(x, comparison_df[metrics_to_plot].iloc[1], width, label=model2_name)
    plt.bar(x + width, comparison_df[metrics_to_plot].iloc[2], width, label=model3_name)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.xticks(x, metrics_to_plot)
    plt.legend()
    plt.show()
    
    # Print detailed classification reports
    print(f"\nClassification Report for {model1_name}:")
    print(classification_report(y_true, model1_pred))
    print(f"\nClassification Report for {model2_name}:")
    print(classification_report(y_true, model2_pred))
    print(f"\nClassification Report for {model3_name}:")
    print(classification_report(y_true, model3_pred))
    
    return comparison_df, report1_df, report2_df, report3_df

# Example usage:
if __name__ == "__main__":
    # Replace these with your actual data
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    model1_pred = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])  # Example predictions
    model2_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 0])  # Example predictions
    model3_pred = np.array([1, 0, 1, 1, 1, 1, 0, 0, 1, 0])  # Example predictions
    
    # Compare models
    results, report1, report2, report3 = compare_models(
        y_true, 
        model1_pred, 
        model2_pred, 
        model3_pred,
        model1_name="Random Forest",
        model2_name="SVM",
        model3_name="Neural Network"
    )
    
    # Print detailed results
    print("\nDetailed Model Comparison:")
    print(results.to_string(index=False))
    
    # Print detailed classification reports as DataFrames
    print("\nDetailed Classification Reports:")
    print("\nRandom Forest Classification Report:")
    print(report1)
    print("\nSVM Classification Report:")
    print(report2)
    print("\nNeural Network Classification Report:")
    print(report3) 