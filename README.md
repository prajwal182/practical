# practical

dl1

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Loading the dataset manually
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

# Processing the dataset
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]

# Defining the column names
columns = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
]

# Creating DataFrame
data = pd.DataFrame(X, columns=columns)
data['PRICE'] = y

# Checking the data
print(data.info())
print(data.head())

# Distribution of target variable
sns.histplot(data['PRICE'], kde=True)
plt.title('Distribution of House Prices')
plt.show()

# Boxplot of house prices
sns.boxplot(x=data['PRICE'])
plt.title('Boxplot of House Prices')
plt.show()

# Correlation analysis
correlation = data.corr()
print(correlation['PRICE'])

# Heatmap of correlations
plt.figure(figsize=(15, 12))
sns.heatmap(correlation, annot=True, square=True)
plt.title('Correlation Heatmap')
plt.show()

# Scatter plots of important features vs PRICE
plt.figure(figsize=(20, 5))
features = ['LSTAT', 'RM', 'PTRATIO']
for i, col in enumerate(features):
    plt.subplot(1, len(features), i+1)
    plt.scatter(data[col], data['PRICE'], marker='o')
    plt.xlabel(col)
    plt.ylabel('House Prices ($1000)')
    plt.title(f'{col} vs Price')
plt.show()

# Splitting the data into independent and dependent features
X = data.iloc[:, :-1]  # All columns except 'PRICE'
y = data['PRICE']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizing (Standardizing) the data
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred_lr = lr_model.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)

print(f"Linear Regression - RMSE: {rmse_lr:.4f}")
print(f"Linear Regression - R2 Score: {r2_lr:.4f}")
print(f"Linear Regression - MAE: {mae_lr:.4f}")

# Neural Network Model
import keras
from keras.models import Sequential
from keras.layers import Dense

# Build the model
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=13))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Model Visualization (optional, requires ann_visualizer and graphviz installed)
# !pip install ann_visualizer graphviz
# from ann_visualizer.visualize import ann_viz
# ann_viz(model, title="DEMO ANN")

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.05)

# Plotting Training and Validation Loss
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(y=history.history['loss'], name='Train Loss'))
fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
fig.update_layout(title='Model Loss', xaxis_title='Epoch', yaxis_title='Loss')
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(y=history.history['mae'], name='Train MAE'))
fig.add_trace(go.Scatter(y=history.history['val_mae'], name='Validation MAE'))
fig.update_layout(title='Model MAE', xaxis_title='Epoch', yaxis_title='Mean Absolute Error')
fig.show()

# Model Evaluation
mse_nn, mae_nn = model.evaluate(X_test, y_test)
print(f"Neural Network - MSE on test data: {mse_nn:.4f}")
print(f"Neural Network - MAE on test data: {mae_nn:.4f}")

# Neural Network Predictions
y_pred_nn = model.predict(X_test)

# Comparison of Models
print("\n--- Comparison ---")
print(f"Linear Regression RMSE: {rmse_lr:.4f}")
print(f"Neural Network RMSE: {np.sqrt(mse_nn):.4f}")

# Predicting new data
scaler = StandardScaler()
scaler.fit(X)  # Fit scaler on complete original dataset (X)
new_data_raw = [[0.1, 10.0, 5.0, 0, 0.4, 6.0, 50, 6.0, 1, 400, 20, 300, 10]]
new_data_scaled = scaler.transform(new_data_raw)

prediction = model.predict(new_data_scaled)
print(f"Predicted house price for new data: {prediction[0][0]:.2f} ($1000)")




hpc bfs fs


Code : 
#include <iostream> 
#include <queue> 
#include <omp.h> 
#include <iomanip> 
#include <vector> 
 
using namespace std; 
 
// Node structure for tree 
class Node { 
public: 
    int data; 
    Node *left, *right; 
     
    // Constructor 
    Node(int value) { 
        data = value; 
        left = right = nullptr; 
    } 
}; 
 
// BFS Tree traversal class 
class ParallelBFS { 
private: 
    Node* root; 
     
public: 
    ParallelBFS() : root(nullptr) {} 
     
    // Insert a node in level order 
    Node* insert(Node* root, int data) { 
        // If tree is empty, create a new node as root 
        if (!root) { 
            return new Node(data); 
        } 
         
        queue<Node*> q; 
        q.push(root); 
         
        // Level order traversal to find the first vacant position 
        while (!q.empty()) { 
            Node* temp = q.front(); 
            q.pop(); 
             
            // If left child is empty, insert here 
            if (!temp->left) { 
                temp->left = new Node(data); 
                return root; 
            } else { 
                q.push(temp->left); 
            } 
             
            // If right child is empty, insert here 
            if (!temp->right) { 
                temp->right = new Node(data); 
                return root; 
            } else { 
                q.push(temp->right); 
            } 
        } 
        return root; // Should never reach here in a proper binary tree 
    } 
     
    // Breadth-first traversal with parallel processing of each level 
    void bfs(Node* root) { 
        if (!root) { 
            cout << "Tree is empty!" << endl; 
            return; 
        } 
         
        queue<Node*> q; 
        q.push(root); 
         
        int level = 0; 
         
        cout << "\n===== Breadth First Search Traversal (Parallel) =====\n" << endl; 
         
        // Process level by level 
        while (!q.empty()) { 
            int levelSize = q.size(); 
            vector<Node*> currentLevel; 
             
            // Extract all nodes at current level 
            for (int i = 0; i < levelSize; i++) { 
                currentLevel.push_back(q.front()); 
                q.pop(); 
            } 
             
            cout << "Level " << level << ": "; 
             
            // Process nodes at current level in parallel 
            #pragma omp parallel for 
            for (int i = 0; i < levelSize; i++) { 
                Node* current = currentLevel[i]; 
                 
                // Critical section for output to prevent garbled text 
                #pragma omp critical 
                { 
                    cout << setw(4) << current->data << " "; 
                } 
                 
                // Add children to queue (needs to be in critical section to avoid race conditions) 
                #pragma omp critical 
                { 
                    if (current->left) 
                        q.push(current->left); 
                    if (current->right) 
                        q.push(current->right); 
                } 
            } 
             
            cout << endl; 
            level++; 
        } 
         
        cout << "\n====================================================" << endl; 
    } 
     
    // Getter and setter for root 
    Node* getRoot() { return root; } 
    void setRoot(Node* newRoot) { root = newRoot; } 
}; 
 
// Depth-first search implementation (added to complement BFS) 
class ParallelDFS { 
public: 
    // Function to perform parallel DFS traversal 
    void dfs(Node* root) { 
        if (!root) { 
            cout << "Tree is empty!" << endl; 
            return; 
        } 
         
        cout << "\n===== Depth First Search Traversal (Parallel) =====\n" << endl; 
         
        // Using OpenMP tasks for parallelism 
        #pragma omp parallel 
        { 
            #pragma omp single nowait 
            { 
                dfsRecursive(root, 0); 
            } 
        } 
         
        cout << "\n===================================================" << endl; 
    } 
     
private: 
    // Recursive DFS with level tracking 
    void dfsRecursive(Node* node, int level) { 
        if (!node) return; 
         
        // Print current node 
        #pragma omp critical 
        { 
            cout << "Level " << level << ": Node " << node->data << endl; 
        } 
         
        // Process children in parallel using OpenMP tasks 
        #pragma omp task 
        { 
            if (node->left) 
                dfsRecursive(node->left, level + 1); 
        } 
         
        #pragma omp task 
        { 
            if (node->right) 
                dfsRecursive(node->right, level + 1); 
        } 
         
        #pragma omp taskwait 
    } 
}; 
 
int main() { 
    ParallelBFS bfsTree; 
    Node* root = nullptr; 
     
    // Set number of threads for OpenMP 
    int numThreads = 4; // You can adjust this based on your system 
    omp_set_num_threads(numThreads); 
     
    cout << "Parallel BFS and DFS Tree Traversal" << endl; 
    cout << "Using " << numThreads << " OpenMP threads" << endl; 
     
    char choice; 
    do { 
        int data; 
        cout << "\nEnter value to insert: "; 
        cin >> data; 
         
        root = bfsTree.insert(root, data); 
        bfsTree.setRoot(root); 
         
        cout << "Do you want to insert another node? (y/n): "; 
        cin >> choice; 
    } while (choice == 'y' || choice == 'Y'); 
     
    // Perform BFS traversal 
    bfsTree.bfs(root); 
// Perform DFS traversal 
ParallelDFS dfsTree; 
dfsTree.dfs(root); 
return 0; 
} 
OUTPUT 
// PS D:\DEGREE\8th Sem\practicals\HPC> g++ -fopenmp pr1.cpp -o bfs 
// PS D:\DEGREE\8th Sem\practicals\HPC> ./bfs 
// Parallel BFS and DFS Tree Traversal 
// Using 4 OpenMP threads 
// Enter value to insert: 7 
// Do you want to insert another node? (y/n): y 
// Enter value to insert: 3 
// Do you want to insert another node? (y/n): y 
// Enter value to insert: 2 
// Do you want to insert another node? (y/n): y 
// Enter value to insert: 6 
// Do you want to insert another node? (y/n): y 
// Enter value to insert: 8 
// Do you want to insert another node? (y/n): y 
// Enter value to insert: 5 
// Do you want to insert another node? (y/n): 1 
// ===== Breadth First Search Traversal (Parallel) ===== 
// Level 0:    
// Level 1:    
// Level 2:    
7 
2    3 
8    6    5 
// ==================================================== 
// ===== Depth First Search Traversal (Parallel) ===== 
// Level 0: Node 7 
// Level 1: Node 2 
// Level 1: Node 3 
// Level 2: Node 5 
// Level 2: Node 8 
// Level 2: Node 6 
// =================================================== 

hpc matrix multiplication

// matrix_multiplication.cu

#include <iostream>
#include <cstdlib>  // for rand()
#include <cuda_runtime.h>

#define N 512  // Size of the matrix (N x N)

__global__ void matrixMulKernel(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Compute row index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Compute column index

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

void randomInit(float *data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = rand() % 100; // Random numbers between 0 and 99
    }
}

int main() {
    int size = N * N;
    int bytes = size * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    randomInit(h_A, size);
    randomInit(h_B, size);

    float *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_B, bytes);
    cudaMalloc((void **)&d_C, bytes);

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the matrix multiplication kernel
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // (Optional) Print a small part of result
    std::cout << "Result Matrix (partial output):" << std::endl;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}


nvcc matrix_multiplication.cu -o matrix_multiplication

./matrix_multiplication


