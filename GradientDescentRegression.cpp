// Simple Linear Regression Model with Gradient Descent
// by MK, 2024

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

// This class does the heavy lifting for our linear regression
class LinearRegression {
private:
    std::vector<double> weights; // our model parameters
    double learning_rate = 0.01; // how fast we adjust weights
    int iterations = 1000;       // how many times we tweak the weights

    // Helper to calculate mean of a vector
    double mean(const std::vector<double>& vec) const {
        double sum = 0.0;
        for (double val : vec) sum += val;
        return sum / vec.size();
    }

    // Helper to calculate variance
    double variance(const std::vector<double>& vec) const {
        double m = mean(vec);
        double accum = 0.0;
        for (double d : vec) {
            accum += (d - m) * (d - m);
        }
        return accum / (vec.size() - 1);
    }

    // Standard deviation is just sqrt of variance
    double std_dev(const std::vector<double>& vec) const {
        return std::sqrt(variance(vec));
    }

    // Normalize the data to help our model perform better
    std::vector<double> normalize(const std::vector<double>& features) const {
        double m = mean(features);
        double sd = std_dev(features);
        std::vector<double> normalized;
        
        // Apply the normalization formula
        for (double feature : features) {
            normalized.push_back((feature - m) / sd);
        }
        return normalized;
    }

public:
    // Constructor - we need to know how many features we're dealing with
    LinearRegression(size_t num_features) {
        // We have one weight for each feature plus one for the bias term
        weights.resize(num_features + 1, 0.0);
    }

    // This is where we train our model using gradient descent
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
        // First, we need to normalize our features
        std::vector<std::vector<double>> normalized_X;
        for (size_t i = 0; i < X[0].size(); ++i) {
            std::vector<double> feature_column;
            for (const auto& row : X) {
                feature_column.push_back(row[i]);
            }
            normalized_X.push_back(normalize(feature_column));
        }

        // Now let's iterate and adjust our weights
        for (int iter = 0; iter < iterations; ++iter) {
            double total_error = 0.0;
            std::vector<double> gradients(weights.size(), 0.0);

            for (size_t i = 0; i < X.size(); ++i) {
                // Predict using current weights
                double predicted = weights[0];  // Bias term
                for (size_t j = 0; j < X[i].size(); ++j) {
                    predicted += weights[j + 1] * normalized_X[j][i];
                }
                
                // Calculate error and gradients
                double error = predicted - y[i];
                total_error += error * error;
                gradients[0] += error;  // Bias gradient
                for (size_t j = 0; j < X[i].size(); ++j) {
                    gradients[j + 1] += error * normalized_X[j][i];
                }
            }

            // Update weights based on gradients
            for (size_t i = 0; i < weights.size(); ++i) {
                weights[i] -= learning_rate * (gradients[i] / X.size());
            }

            // Log progress every 100 iterations
            if (iter % 100 == 0) {
                std::cout << "Iteration " << iter << ": MSE = " << (total_error / X.size()) << std::endl;
            }
        }
    }

    // Use the trained model to make predictions
    double predict(const std::vector<double>& sample) const {
        std::vector<double> normalized_sample = normalize(sample);
        double prediction = weights[0];  // Bias
        for (size_t i = 0; i < sample.size(); ++i) {
            prediction += weights[i + 1] * normalized_sample[i];
        }
        return prediction;
    }
};

// Quick function to generate some fake data for testing
void generate_dataset(std::vector<std::vector<double>>& X, std::vector<double>& y, int num_samples, int num_features) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0, 0.1);
    
    // Some made-up coefficients for our data
    std::vector<double> true_coeffs = {2.5};
    for (int i = 0; i < num_features; ++i) {
        true_coeffs.push_back((i + 1) * (i % 2 ? -1 : 1) * 0.5);  // Adding some variety
    }
    
    for (int i = 0; i < num_samples; ++i) {
        std::vector<double> sample;
        double prediction = true_coeffs[0];  // Bias
        
        for (int j = 0; j < num_features; ++j) {
            std::uniform_real_distribution<> dist(j * 10, (j + 1) * 10);  // Spread out features
            double feature = dist(gen);
            sample.push_back(feature);
            prediction += true_coeffs[j + 1] * feature;
        }
        
        // Add a touch of randomness
        prediction += noise(gen);
        
        X.push_back(sample);
        y.push_back(prediction);
    }
}

int main() {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    generate_dataset(X, y, 100, 2);  // Let's generate 100 samples with 2 features

    LinearRegression model(X[0].size());
    model.fit(X, y);

    std::cout << "Here are some predictions:" << std::endl;
    for (size_t i = 0; i < 5; ++i) {
        double pred = model.predict(X[i]);
        std::cout << "Actual: " << y[i] << ", Predicted: " << pred << std::endl;
    }

    return 0;
}