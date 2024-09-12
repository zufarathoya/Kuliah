#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

std::vector<std::vector<double>> iris_data;
// using vector
std::vector<double> w = {0.50000, 0.80000, 0.70000, 0.30000, 0.50000};
// std::vector<double> w = {0.20000, 0.20000, 0.20000, 0.20000, 0.20000};
std::vector<double> dw = {0.10000, 0.10000, 0.10000, 0.10000, 0.10000};
// std::vector<double> dw;
double learning_rate = 0.10000;
// double activation = 0;
double prediction = 0;
std::vector<double> errors;
std::vector<double> predictions;
std::vector<double> accuray;
std::vector<double> meanError;

// std::vector<std::vector<double>> readIrisData(const std::string
// &filename) {
void readIrisData(std::string filename) {
  std::ifstream file(filename);
  // std::vector<std::vector<double>> iris_data;
  std::string line;

  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string token;
    std::vector<double> iris_sample;

    while (std::getline(ss, token, ',')) {
      if (token == "Iris-setosa") {
        // iris_sample.push_back(0.0);
        continue;
      } else if (token == "Iris-versicolor") {
        // iris_sample.push_back(2.0);
        continue;
      } else if (token == "Iris-virginica") {
        // iris_sample.push_back(1.0);
      } else {
        iris_sample.push_back(std::stod(token));
        continue;
      }
    }
    // if (iris_sample.size() == 5) {
    iris_data.push_back(iris_sample);
    // }
  }

  file.close();
  // return iris_data;
}

double activationFunction(int i) {
  // for (int i; i < sizeof(iris_data) / sizeof(iris_data[0]); i++) {
  double result = (w[0] * iris_data[i][0]) + (w[1] * iris_data[i][1]) +
                  (w[2] * iris_data[i][2]) + (w[3] * iris_data[i][3]) + w[4];
  result = 1 / (1 + exp(-result));
  return result;
  // }
}

void updateDWeight(int i) {
  // dw[i] = iris_data[0][i] * w[i];
  double activation = activationFunction(i);
  // dw[i] = dw[i] + (iris_data[0][i] * w[i]);
  for (int j = 0; j < 4; j++) {
    dw[j] = 2 * (activation - iris_data[i][4]) * activation * (1 - activation) *
            iris_data[i][j];
  }
  dw[4] = 2 * (activation - iris_data[i][4]) * activation * (1 - activation);
}

void predict(int i) {
  prediction = activationFunction(i);
  if (prediction >= 0.5) {
    prediction = 1;
  } else {
    prediction = 0;
  }
  predictions.push_back(prediction);
}

void updateWeight(int i) {
  for (int j = 0; j < w.size(); j++) {
    w[j] = w[j] - (learning_rate * dw[j]);
  }
}

void squareError(int i) {
  double error = 0;
  // for (int j = 0; j < iris_data[i].size(); j++) {
  error = pow((iris_data[i][4] - activationFunction(i)), 2);
  //}
  errors.push_back(error);
}

void meanSquareError() {
  double total_error = 0;
  for (int i = 0; i < errors.size(); i++) {
    total_error += errors[i];
  }
  double mse = total_error / errors.size();
  meanError.push_back(mse);
}

void accurate() {
  double correct = 0;
  for (int i = 0; i < iris_data.size(); i++) {
    if (predictions[i] == iris_data[i][4]) {
      correct++;
    }
  }
  double accuracy = correct / iris_data.size();
  accuray.push_back(accuracy);
}

// training
void train(int epoch = 10) {
  while (epoch > 0) {
    predictions.clear();
    errors.clear();
    for (int i = 0; i < iris_data.size(); i++) {
      updateDWeight(i);
      updateWeight(i);
      predict(i);
      squareError(i);
    }
    accurate();
    meanSquareError();
    epoch--;
  }
}

int main() {
  readIrisData("iris_data.data");

  train();
  for (int i = 0; i < w.size(); i++) {
    printf("w: %.9f \n", w[i]);
  }
  std::ofstream file("result.csv");

  file << "x,y\n";

  for (int i = 0; i < accuray.size(); i++) {
    printf("Epoch %d: %.2f %c\n", i + 1, accuray[i] * 100, '%');
    file << i + 1 << "," << accuray[i] << "\n";
  }
  for (int i = 0; i < meanError.size(); i++) {
    printf("Epoch %d: %.9f \n", i + 1, meanError[i]);
    file << i + 1 << "," << meanError[i] << "\n";
  }

  accuray.clear();
  meanError.clear();
  iris_data.clear();

  readIrisData("test.csv");

  train();
  for (int i = 0; i < accuray.size(); i++) {
    printf("Epoch %d: %.2f %c\n", i + 1, accuray[i] * 100, '%');
    file << i + 1 << "," << accuray[i] << "\n";
  }
  for (int i = 0; i < meanError.size(); i++) {
    printf("Epoch %d: %.9f \n", i + 1, meanError[i]);
    file << i + 1 << "," << meanError[i] << "\n";
  }

  file.close();

  return 0;
}
