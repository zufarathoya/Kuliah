#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

std::vector<std::vector<double>> iris_data;
std::vector<double> w = {0.50000, 0.80000, 0.70000, 0.30000, 0.50000};  
std::vector<double> dw = {0.00000, 0.00000, 0.00000, 0.00000, 0.00000};
double learning_rate = 0.10000;
double prediction = 0;
std::vector<double> predictions;
std::vector<std::vector<double>> accuracies;  
std::vector<std::vector<double>> meanErrors;   
std::vector<std::vector<double>> weightsHistory; 
std::vector<std::vector<double>> deltaWeightsHistory;
std::vector<std::vector<double>> activationsHistory; 

void readIrisData(std::string filename) {
  std::ifstream file(filename);
  std::string line;

  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string token;
    std::vector<double> iris_sample;

    while (std::getline(ss, token, ',')) {
      if (token == "Iris-setosa" || token == "Iris-versicolor" || token == "Iris-virginica") {
        continue;
      } else {
        iris_sample.push_back(std::stod(token));
      }
    }
    iris_data.push_back(iris_sample);
  }
  file.close();
}

double activationFunction(int i) {
  double result = (w[0] * iris_data[i][0]) + (w[1] * iris_data[i][1]) +
                  (w[2] * iris_data[i][2]) + (w[3] * iris_data[i][3]) + w[4];
  result = 1 / (1 + exp(-result));
  return result;
}

void updateDWeight(int i) {
  double activation = activationFunction(i);
  for (int j = 0; j < 4; j++) {
    dw[j] = 2 * (activation - iris_data[i][4]) * activation * (1 - activation) * iris_data[i][j];
  }
  dw[4] = 2 * (activation - iris_data[i][4]) * activation * (1 - activation);
}

void predict(int i) {
  prediction = activationFunction(i);
  prediction = prediction >= 0.5 ? 1 : 0;
  predictions.push_back(prediction);
}

void updateWeight(int i) {
  for (int j = 0; j < w.size(); j++) {
    w[j] = w[j] - (learning_rate * dw[j]);
  }
}

void squareError(int i, std::vector<double>& errors) {
  double error = pow((iris_data[i][4] - activationFunction(i)), 2);
  errors.push_back(error);
}

void meanSquareError(std::vector<double>& errors) {
  double total_error = 0;
  for (double error : errors) {
    total_error += error;
  }
  double mse = total_error / errors.size();
  meanErrors.back().push_back(mse);
}

void accurate() {
  double correct = 0;
  for (int i = 0; i < iris_data.size(); i++) {
    if (predictions[i] == iris_data[i][4]) {
      correct++;
    }
  }
  double accuracy = correct / iris_data.size();
  accuracies.back().push_back(accuracy);
}

void storeWeights() {
  weightsHistory.emplace_back(w); 
}

void storeDeltaWeights() {
  deltaWeightsHistory.emplace_back(dw);
}

void storeActivations(int i) {
  double current_activation = activationFunction(i);
  activationsHistory.push_back({(double)i, current_activation});
}

void saveCombinedCSV(const std::string& filename) {
  std::ofstream file(filename);
  file << "iteration,weight1,weight2,weight3,weight4,weight_bias,dw1,dw2,dw3,dw4,dw_bias,activation\n";
  for (int i = 0; i < weightsHistory.size(); i++) {
    file << i + 1 << ",";
    for (double w_val : weightsHistory[i]) {
      file << w_val << ",";
    }
    for (double dw_val : deltaWeightsHistory[i]) {
      file << dw_val << ",";
    }
    file << activationsHistory[i][1] << "\n"; 
  }
  file.close();
}

void train(int epoch = 10) {
  storeWeights();
  
  for (int i = 0; i < iris_data.size(); i++) {
    storeActivations(i);
  }

  while (epoch > 0) {
    std::vector<double> current_epoch_errors;
    predictions.clear();
    
    for (int i = 0; i < iris_data.size(); i++) {
      updateDWeight(i);
      predict(i);
      updateWeight(i);
      squareError(i, current_epoch_errors);
      storeWeights();       
      storeDeltaWeights();  
      storeActivations(i);  
    }

    accuracies.emplace_back();
    accurate();
    meanErrors.emplace_back();  
    meanSquareError(current_epoch_errors);
    
    epoch--;
  }
}

int main() {
  readIrisData("iris_data.data");

  train();

  saveCombinedCSV("combined_results_train.csv");

  std::ofstream resultFile("result.csv");
  resultFile << "epoch,accuracy,meanError\n";
  for (int epoch = 0; epoch < accuracies.size(); epoch++) {
    resultFile << epoch + 1 << "," << accuracies[epoch][0] << "," << meanErrors[epoch][0] << "\n";
  }
  iris_data.clear();
  deltaWeightsHistory.clear();
  weightsHistory.clear();
  activationsHistory.clear();
  predictions.clear();  
  accuracies.clear();
  meanErrors.clear();

  readIrisData("test.csv");

  train();
  saveCombinedCSV("combined_results_train.csv");
  for (int epoch = 0; epoch < accuracies.size(); epoch++) {
    resultFile << epoch + 1 << "," << accuracies[epoch][0] << "," << meanErrors[epoch][0] << "\n";
  }

  resultFile.close();

  return 0;
}

