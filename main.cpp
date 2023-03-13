

#include <bits/stdc++.h> ///<!---Used this header file got lazy importing all individual files for this project---!>

using namespace std;

// Define sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Perform inference on input sequence
vector<vector<double>> inference(vector<vector<double>> x,
                                  vector<vector<double>> Wxh,
                                  vector<vector<double>> Whh,
                                  vector<vector<double>> Why,
                                  vector<double> bh,
                                  vector<double> by,
                                  vector<double> h0) {
    const int sequence_length = x.size();
    const int hidden_size = h0.size();
    const int output_size = by.size();

    // Initialize hidden state
    vector<double> h = h0;

    // Initialize output sequence
    vector<vector<double>> y(sequence_length, vector<double>(output_size, 0.0));

    // Perform forward pass
    for (int t = 0; t < sequence_length; t++) {
        // Compute hidden state
        vector<double> new_h(hidden_size, 0.0);
        for (int i = 0; i < hidden_size; i++) {
            double sum = bh[i];
            for (int j = 0; j < x[t].size(); j++) {
                sum += Wxh[i][j] * x[t][j];
            }
            for (int j = 0; j < hidden_size; j++) {
                sum += Whh[i][j] * h[j];
            }
            new_h[i] = sigmoid(sum);
        }
        h = new_h;

        // Compute output
        for (int i = 0; i < output_size; i++) {
            double sum = by[i];
            for (int j = 0; j < hidden_size; j++) {
                sum += Why[i][j] * h[j];
            }
            y[t][i] = sigmoid(sum);
        }
    }

    return y;
}

int main() {
    // Define network parameters
    const int input_size = 3; // Size of input vector
    const int hidden_size = 4; // Size of hidden layer
    const int output_size = 2; // Size of output vector

    // Initialize weight matrices
    vector<vector<double>> Wxh(hidden_size, vector<double>(input_size, 0.0));
    vector<vector<double>> Whh(hidden_size, vector<double>(hidden_size, 0.0));
    vector<vector<double>> Why(output_size, vector<double>(hidden_size, 0.0));

    // Initialize bias vectors
    vector<double> bh(hidden_size, 0.0);
    vector<double> by(output_size, 0.0);

    // Initialize input sequence
    vector<vector<double>> x = {{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9}};

    // Initialize hidden state
    vector<double> h0(hidden_size, 0.0);

    // Perform inference
    vector<vector<double>> y = inference(x, Wxh, Whh, Why, bh, by, h0);

    // Print output
    for (int t = 0; t < x.size(); t++) {
        cout << "Output at time step " << t << ": [";
        for (int i = 0; i < output_size; i++) {
            cout << y[t][i] << ", ";
        }
             cout << "]" << endl;
    }
}
        /// <!---Justs' Recurrent Neural Network Implementation in C++ ---!>
        
        
        