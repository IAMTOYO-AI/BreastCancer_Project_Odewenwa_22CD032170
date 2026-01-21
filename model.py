import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os


class BreastCancerModel:
    """
    Breast Cancer Prediction Model
    Predicts whether a tumor is Benign or Malignant based on cell characteristics
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def load_data(self, file_path=None):
        """
        Load and preprocess breast cancer data
        Returns: X (features), y (diagnosis labels)
        """
        if file_path is None:
            # Use absolute path
            base_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(base_dir, 'data', 'breast_cancer.csv')

        try:
            df = pd.read_csv(file_path)
            print(f"✓ Loaded {len(df)} patient records")

            # Separate features and target
            # diagnosis: 0 = malignant (cancerous), 1 = benign (non-cancerous)
            X = df.drop('diagnosis', axis=1)
            y = df['diagnosis']

            # Store feature names
            self.feature_names = X.columns.tolist()

            print("\nDataset Statistics:")
            print(f"  Total Samples: {len(df)}")
            print(f"  Malignant (Cancerous): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
            print(f"  Benign (Non-cancerous): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")
            print(f"  Number of Features: {len(self.feature_names)}")

            return X, y

        except FileNotFoundError:
            print(f"✗ Error: {file_path} not found")
            return None, None

    def train(self, X, y):
        """
        Train the cancer prediction model
        Uses Random Forest Classifier with feature scaling
        """
        print("\n--- Training Model ---")

        # Split data: 80% training, 20% testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features (important for medical data)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Random Forest Classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.model.fit(X_train_scaled, y_train)

        # Evaluate model
        train_predictions = self.model.predict(X_train_scaled)
        test_predictions = self.model.predict(X_test_scaled)

        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)

        print(f"\nModel Performance:")
        print(f"  Training Accuracy: {train_accuracy * 100:.2f}%")
        print(f"  Testing Accuracy: {test_accuracy * 100:.2f}%")

        # Detailed classification report
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, test_predictions,
                                    target_names=['Malignant', 'Benign']))

        # Confusion Matrix
        cm = confusion_matrix(y_test, test_predictions)
        print("\nConfusion Matrix:")
        print(f"  True Negatives (Correctly predicted Malignant): {cm[0][0]}")
        print(f"  False Positives (Wrongly predicted Benign): {cm[0][1]}")
        print(f"  False Negatives (Wrongly predicted Malignant): {cm[1][0]}")
        print(f"  True Positives (Correctly predicted Benign): {cm[1][1]}")

        # Feature importance (top 10)
        self._show_feature_importance()

        return test_accuracy

    def _show_feature_importance(self):
        """Display which features matter most for diagnosis"""
        if self.model and self.feature_names:
            importance = self.model.feature_importances_
            feature_imp = sorted(zip(self.feature_names, importance),
                                 key=lambda x: x[1], reverse=True)

            print("\nTop 10 Most Important Features:")
            for i, (name, imp) in enumerate(feature_imp[:10], 1):
                print(f"  {i}. {name}: {imp:.4f}")

    def predict(self, features):
        """
        Predict cancer diagnosis based on cell characteristics

        Parameters:
        - features: dict or array of 30 feature values

        Returns: (prediction, probability)
        - prediction: 0 (malignant) or 1 (benign)
        - probability: Probability of being benign (0-1)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Convert features dict to array if needed
        if isinstance(features, dict):
            features_array = np.array([[features[name] for name in self.feature_names]])
        else:
            features_array = np.array([features])

        # Scale features
        features_scaled = self.scaler.transform(features_array)

        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]

        # Return prediction and benign probability
        return prediction, probability[1]

    def save_model(self, model_dir='model_files'):
        """Save trained model and scaler to disk"""
        # Use absolute path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, model_dir)
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, 'cancer_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        features_path = os.path.join(model_dir, 'feature_names.pkl')

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.feature_names, features_path)

        print(f"\n✓ Model saved to {model_path}")
        print(f"✓ Scaler saved to {scaler_path}")
        print(f"✓ Feature names saved to {features_path}")

    def load_model(self, model_dir='model_files'):
        """Load pre-trained model and scaler from disk"""
        # Use absolute path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, model_dir)

        model_path = os.path.join(model_dir, 'cancer_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        features_path = os.path.join(model_dir, 'feature_names.pkl')

        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_names = joblib.load(features_path)
            print(f"✓ Model loaded from {model_path}")
            return True
        except FileNotFoundError:
            print(f"✗ Model files not found in {model_dir}")
            return False


def train_and_save_model():
    """
    Main training function - run this to create the model
    """
    print("=" * 60)
    print("BREAST CANCER PREDICTION MODEL TRAINING")
    print("=" * 60)

    # Initialize model
    cancer_model = BreastCancerModel()

    # Load data
    X, y = cancer_model.load_data()
    if X is None:
        return

    # Train model
    cancer_model.train(X, y)

    # Save trained model
    cancer_model.save_model()

    # Test predictions with sample data
    print("\n--- Testing Sample Predictions ---")

    # Get first sample from dataset (known benign case)
    sample_benign = X.iloc[0].to_dict()
    pred, prob = cancer_model.predict(sample_benign)
    print(f"Sample 1 (Known Benign): {'Benign ✓' if pred == 1 else 'Malignant ⚠️'} "
          f"(Confidence: {prob * 100:.1f}%)")

    # Get a malignant sample
    sample_malignant = X.iloc[y[y == 0].index[0]].to_dict()
    pred, prob = cancer_model.predict(sample_malignant)
    print(f"Sample 2 (Known Malignant): {'Benign ✓' if pred == 1 else 'Malignant ⚠️'} "
          f"(Confidence: {(1 - prob) * 100:.1f}%)")

    print("\n" + "=" * 60)
    print("✓ Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    train_and_save_model()