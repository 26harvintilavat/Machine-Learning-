# Building a simple linear regression from scratch

import numpy as np
import matplotlib.pyplot as plt

class SimpleLinearRegression:
    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        mean_x = np.mean(X)
        mean_y = np.mean(y)

        numerator = np.sum((X-mean_x) * (y-mean_y))
        denominator = np.sum((X-mean_x) ** 2)
        self.slope = numerator / denominator

        self.intercept = mean_y - self.slope * mean_x

        print(f"Model trained successfully!")
        print(f"Slope (m): {self.slope:.4f}")
        print(f"Intercept (b): {self.intercept:.4f}")
        print(f"Equation: y = {self.slope:.4f}x + {self.intercept:.4f}")

    def predict(self, X):
        if self.slope is None or self.intercept is None:
            raise ValueError("Model must be fitted before predictions!")
        
        X = np.array(X)
        return self.slope * X + self.intercept
    
    def score(self, X, y):
        y = np.array(y)
        y_pred = self.predict(X)

        SS_residual = np.sum((y-y_pred) ** 2)
        SS_total = np.sum((y-np.mean(y)) ** 2)
        r_squared = 1 - (SS_residual / SS_total)

        return r_squared
    
    def mean_squared_error(self, X, y):
        y = np.array(y)
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)

        return mse
    
if __name__ == "__main__":
    np.random.seed(42)
    hours_studied = np.array([1,2,3,4,5,6,7,8,9,10])
    exam_scores = 30 + hours_studied * 5 + np.random.randn(10) * 5

    model = SimpleLinearRegression()
    model.fit(hours_studied, exam_scores)

    predictions = model.predict(hours_studied)

    r2 = model.score(hours_studied, exam_scores)
    mse = model.mean_squared_error(hours_studied, exam_scores)

    print(f"\nModel Performance:")
    print(f"R-squared: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(hours_studied, exam_scores, c='blue', label="Actual data", s=100)
    plt.plot(hours_studied, predictions, c='red', linewidth=2, label="Regression Line")
    plt.xlabel('Hours Studied', fontsize=12)
    plt.ylabel('Exam Score', fontsize=12)
    plt.title('Simple Linear Regression: Hours Studied vs Exam Score', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    new_hours = np.array([5.5, 8.5])
    new_predictions = model.predict(new_hours)
    print("\nPredictions for new data:")
    for hours, score in zip(new_hours, new_predictions):
        print(f"  {hours} hours → predicted score: {score:.2f}")
