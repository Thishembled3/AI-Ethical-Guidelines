import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

class BiasDetector:
    """
    A toolkit for detecting and mitigating bias in machine learning models.
    This class demonstrates how to use AIF360 for bias detection.
    """
    def __init__(self, protected_attribute, privileged_groups, unprivileged_groups):
        self.protected_attribute = protected_attribute
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        print(f"BiasDetector initialized for protected attribute: {protected_attribute}")

    def load_and_prepare_data(self, filepath, label_name, favorable_label, features_to_drop=None):
        """
        Loads data and prepares it for bias detection using AIF360.
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        if features_to_drop:
            df = df.drop(columns=features_to_drop)

        # Convert to AIF360 StandardDataset
        dataset = StandardDataset(df, 
                                  label_name=label_name, 
                                  favorable_classes=[favorable_label], 
                                  protected_attribute_names=[self.protected_attribute], 
                                  privileged_classes=self.privileged_groups)
        print("Data loaded and converted to AIF360 dataset.")
        return dataset

    def train_model(self, dataset):
        """
        Trains a simple logistic regression model on the dataset.
        """
        print("Training a Logistic Regression model...")
        X = dataset.features
        y = dataset.labels.ravel()

        model = LogisticRegression(solver=\'liblinear\', random_state=0)
        model.fit(X, y)
        print("Model training complete.")
        return model

    def evaluate_bias(self, original_dataset, trained_model):
        """
        Evaluates bias in the trained model using various metrics.
        """
        print("Evaluating bias in the model...")
        # Make predictions
        y_pred = trained_model.predict(original_dataset.features)
        pred_dataset = original_dataset.copy()
        pred_dataset.labels = y_pred

        metric = ClassificationMetric(original_dataset, 
                                      pred_dataset, 
                                      unprivileged_groups=self.unprivileged_groups, 
                                      privileged_groups=self.privileged_groups)
        
        print(f"Disparate Impact: {metric.disparate_impact():.4f}")
        print(f"Statistical Parity Difference: {metric.statistical_parity_difference():.4f}")
        print(f"Equal Opportunity Difference: {metric.equal_opportunity_difference():.4f}")
        print("Bias evaluation complete.")
        return metric

    def mitigate_bias_reweighing(self, original_dataset):
        """
        Applies reweighing preprocessing technique to mitigate bias.
        """
        print("Applying Reweighing bias mitigation technique...")
        RW = Reweighing(unprivileged_groups=self.unprivileged_groups, 
                        privileged_groups=self.privileged_groups)
        dataset_transf = RW.fit_transform(original_dataset)
        print("Reweighing applied. Transformed dataset created.")
        return dataset_transf

# Example Usage:
if __name__ == "__main__":
    # Create a dummy dataset for demonstration
    data = {
        'age': [25, 30, 35, 40, 28, 32, 38, 45, 22, 33],
        'education': [12, 16, 18, 14, 12, 16, 18, 14, 12, 16],
        'income': [30000, 50000, 70000, 60000, 35000, 55000, 75000, 65000, 28000, 52000],
        'gender': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1], # 0 for female, 1 for male
        'loan_approved': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] # 0 for denied, 1 for approved
    }
    df_dummy = pd.DataFrame(data)
    df_dummy.to_csv("dummy_loan_data.csv", index=False)

    # Define protected attribute and groups
    protected_attribute_name = 'gender'
    privileged_groups = [{'gender': 1}]
    unprivileged_groups = [{'gender': 0}]

    bias_detector = BiasDetector(protected_attribute_name, privileged_groups, unprivileged_groups)

    # Load and prepare data
    dataset_orig = bias_detector.load_and_prepare_data(
        filepath="dummy_loan_data.csv",
        label_name='loan_approved',
        favorable_label=1.0
    )

    # Train a model
    model_orig = bias_detector.train_model(dataset_orig)

    # Evaluate bias in the original model
    bias_detector.evaluate_bias(dataset_orig, model_orig)

    # Mitigate bias using reweighing
    dataset_reweighed = bias_detector.mitigate_bias_reweighing(dataset_orig)

    # Train a new model on the reweighed dataset
    model_reweighed = bias_detector.train_model(dataset_reweighed)

    # Evaluate bias in the reweighed model
    bias_detector.evaluate_bias(dataset_reweighed, model_reweighed)

    # Clean up dummy data
    os.remove("dummy_loan_data.csv")
