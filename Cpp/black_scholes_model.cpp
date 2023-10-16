#include <iostream>
#include <limits>
#include <cmath>
#include <fstream>
#include <windows.h>
#include <algorithm>
#include <vector>
#include <iomanip> 


double N(double x) {
    return 0.5 * erfc(-x * std::sqrt(0.5));
}

double blackScholesCall(double S, double K, double r, double T, double sigma) {
    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    return S * N(d1) - K * exp(-r * T) * N(d2);
}

double blackScholesPut(double S, double K, double r, double T, double sigma) {
    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    return K * exp(-r * T) * N(-d2) - S * N(-d1);
}

int main() {
    double S;
    double K;
    double r;
    double t;      // In days
    double sigma;  // Volatility annualized

    /*
    Mathematical notations:

    C = Call option price
    S = Current stock price
    K = Strike price
    r = Risk-free interest rate
    t = Time to maturity
    N = Normal Distribution
    
    */

    char option;
    std::cout << "What is the option? (Enter C for call & P for put) - ";
    std::cin >> option;
    std::cout << "What is the current stock price? - ";
    std::cin >> S;
    std::cout << "What is the strike price? - ";
    std::cin >> K;
    std::cout << "What is the risk free rate? (if 1%, then enter 0.01) - ";
    std::cin >> r;
    std::cout << "How many days till maturity? - ";
    std::cin >> t;
    std::cout << "What is the annualized volatility? (if 10%, then enter 0.1) - ";
    std::cin >> sigma;

    double T = t / 365; // Convert days to years
    double optionPrice;

    if (option == 'C' || option == 'c') {
        optionPrice = blackScholesCall(S, K, r, T, sigma);
    }
    else if (option == 'P' || option == 'p') {
        optionPrice = blackScholesPut(S, K, r, T, sigma);
    }
    else {
        optionPrice = 0.0;
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "The theoretical option price is: $" << optionPrice << std::endl;

    return 0;
}

