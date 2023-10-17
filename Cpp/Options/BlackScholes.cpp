
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include "Definitions.h"
#include <windows.h>

// Define M_PI if not already defined. Represents the value of Pi.
#ifndef M_PI
#define M_PI 3.14159
#endif


// Function to calculate the payoff for a call option.
double callInstristicValue(double stockPrice, double strikePrice) {
    return std::max<double>(0.0, stockPrice - strikePrice);
}

// Function to Calculate the payoff for a put option.
double putInstristicValue(double stockPrice, double strikePrice) {
    return std::max<double>(0.0, strikePrice - stockPrice);
}

/**
 * @class BlackScholes
 *
 * @brief Class to compute Black-Scholes option pricing and Greeks.
 *
 * This class provides functions to compute the prices of European call
 * and put options, as well as the Greeks (Delta, Gamma, Vega, Theta, and Rho).
 */

class BlackScholes {
private:
    // Private member variables
    double S;       // Current stock price
    double K;       // Strike price
    double r;       // Risk-free interest rate
    double T;       // Time to maturity in years
    double sigma;   // Annualized volatility

    // Intermediate calculations for option pricing
    double d1, d2;

    /**
     * @brief Compute the d1 and d2 values used in Black-Scholes formulae.
     */
    void computeD1D2() {
        d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
        d2 = d1 - sigma * sqrt(T);
    }

    /**
     * @brief Cumulative distribution function of the standard normal distribution.
     * @param x Value to compute the CDF for.
     * @return Cumulative distribution from negative infinity to x.
     */
    double N(double x) const {
        return 0.5 * erfc(-x * std::sqrt(0.5));
    }

    /**
     * @brief Probability density function of the standard normal distribution.
     * @param x Value to compute the PDF for.
     * @return Value of the PDF at x.
     */
    double n(double x) const {
        return (1.0 / std::sqrt(2.0 * M_PI)) * exp(-0.5 * x * x);
    }

public:
    // Constructor
    BlackScholes(double S, double K, double r, double T, double sigma)
        : S(S), K(K), r(r), T(T), sigma(sigma) {
        computeD1D2();
    }

    // Public member functions

    /**
     * @brief Compute the call option price.
     * @return Theoretical price of the call option.
     */
    double callPrice() const {
        return S * N(d1) - K * exp(-r * T) * N(d2);
    }

    /**
     * @brief Compute the put option price.
     * @return Theoretical price of the put option.
     */
    double putPrice() const {
        return K * exp(-r * T) * N(-d2) - S * N(-d1);
    }

    /**
     * @brief Compute the call option's Delta.
     * @return Delta of the call option.
     */
    double callDelta() const {
        return N(d1);
    }

    /**
     * @brief Compute the put option's Delta.
     * @return Delta of the put option.
     */
    double putDelta() const {
        return (N(d1) - 1);
    }

    /**
     * @brief Compute the option's Gamma (same for both call and put).
     * @return Gamma of the option.
     */
    double gamma() const {
        return (n(d1) / (S * sigma * sqrt(T)));
    }

    /**
     * @brief Compute the call option's Theta.
     * @return Theta of the call option.
     */
    double callTheta() const {
        return ((-S * sigma * n(d1) / (2 * sqrt(T))) - (r * K * exp(-r * T) * N(d2)));
    }

    /**
     * @brief Compute the put option's Theta.
     * @return Theta of the put option.
     */
    double putTheta() const {
        return ((-S * sigma * n(d1) / (2 * sqrt(T))) + (r * K * exp(-r * T) * N(-d2)));
    }

    /**
     * @brief Compute the option's Vega (same for both call and put).
     * @return Vega of the option.
     */
    double optionVega() const {
        return (S * sqrt(T) * n(d1));
    }

    /**
     * @brief Compute the call option's Rho.
     * @return Rho of the call option.
     */
    double callRho() const {
        return (K * T * exp(-r * T) * N(d2));
    }

    /**
     * @brief Compute the put option's Rho.
     * @return Rho of the put option.
     */
    double putRho() const {
        return (-K * T * exp(-r * T) * N(-d2));
    }
};

std::vector<OptionDetails> getInputFromUser() {
    int numberOfOptions;
    std::vector<OptionDetails> options;

    std::cout << " -- Global Inputs -- " << std::endl;

    std::cout << "How many different options do you have? ";
    std::cin >> numberOfOptions;
    options.resize(numberOfOptions);

    // Collect the risk-free rate once outside the loop
    double commonRiskFreeRate;
    std::cout << "What is the risk free rate? (if 1%, then enter 0.01) - ";
    std::cin >> commonRiskFreeRate;

    double currentStockPrice;
    std::cout << "What is the current stock price? - ";
    std::cin >> currentStockPrice;

    for (int i = 0; i < numberOfOptions; i++) {
        std::cout << "\nDetails for Option " << i + 1 << ":" << std::endl;
        std::cout << "What is the option? (1 for call & 0 for put) - ";
        std::cin >> options[i].type;
        std::cout << "What is the quantity? (Negative for sold) - ";
        std::cin >> options[i].quantity;
        std::cout << "What is the current price for the option? - ";
        std::cin >> options[i].impliedPremium;
        std::cout << "What is the strike price? - ";
        std::cin >> options[i].strikePrice;
        std::cout << "How many days till maturity? - ";
        std::cin >> options[i].daysTillMaturity;
        std::cout << "What is the annualized volatility? (if 10%, then enter 0.1) - ";
        std::cin >> options[i].annualizedVolatility;

        options[i].riskFreeRate = commonRiskFreeRate;
        options[i].stockPrice = currentStockPrice;
    }

    return options;
}


int main() {
    std::vector<OptionDetails> options = getInputFromUser();

    std::cout << std::fixed << std::setprecision(2);

    int x = 0;
    for (const auto& optDetail : options) {

        std::cout << "" << std::endl;
        std::cout << "" << std::endl;
        std::cout << "-- Option " << x + 1 << " -- " << std::endl;

        // Convert days to years
        double T = optDetail.daysTillMaturity / 365.0;

        // Create an instance of BlackScholes using the details from the current option
        BlackScholes option(optDetail.stockPrice, optDetail.strikePrice, optDetail.riskFreeRate, T, optDetail.annualizedVolatility);

        // Display option data based on user's choice
        if (optDetail.type == 1) {

            options[x].theoreticalPremium = option.callPrice();

            std::cout << "The current call option price is: $" << optDetail.impliedPremium << std::endl;
            std::cout << "The theoretical call option price is: $" << options[x].theoreticalPremium << std::endl;
            std::cout << "Difference: $" << option.callPrice() - optDetail.impliedPremium << std::endl;
            std::cout << "" << std::endl;
            std::cout << "Delta for the call option is: " << option.callDelta() << std::endl;
            std::cout << "Gamma for the option is: " << option.gamma() << std::endl;
            std::cout << "Theta for the call option is: $" << option.callTheta() / 365 << " per day" << std::endl;
            std::cout << "Vega for the call option is: " << option.optionVega() / 100 << " per 1% volatility change" << std::endl;
            std::cout << "Rho for the call option is: " << option.callRho() << " per 1% interest rate change" << std::endl;

        }
        else if (optDetail.type == 0) {

            options[x].theoreticalPremium = option.putPrice();

            std::cout << "The current put option price is: $" << optDetail.impliedPremium << std::endl;
            std::cout << "The theoretical put option price is: $" << options[x].theoreticalPremium << std::endl;
            std::cout << "Difference: $" << option.putPrice() - optDetail.impliedPremium << std::endl;
            std::cout << "" << std::endl;
            std::cout << "Delta for the put option is: " << option.putDelta() << std::endl;
            std::cout << "Gamma for the option is: " << option.gamma() << std::endl;
            std::cout << "Theta for the put option is: $" << option.putTheta() / 365 << " per day" << std::endl;
            std::cout << "Vega for the put option is: " << option.optionVega() / 100 << " per 1% volatility change" << std::endl;
            std::cout << "Rho for the put option is: " << option.putRho() << " per 1% interest rate change" << std::endl;
        }
        else {
            std::cout << "Invalid option type entered." << std::endl;
        }
    x++;
    }

    std::ofstream dataFile("data.txt");

    // Setting the range for the chart
    double minStrikePrice = 1e30;   // Some small value
    double maxStrikePrice = -1e30;  // Some large value

    for (const auto& optDetail : options) {
        if (optDetail.strikePrice < minStrikePrice) {
            minStrikePrice = optDetail.strikePrice;
        }
        if (optDetail.strikePrice > maxStrikePrice) {
            maxStrikePrice = optDetail.strikePrice;
        }
    }

    double startPrice = minStrikePrice * 0.95; // 5% lower than min
    double endPrice = maxStrikePrice * 1.05;   // 5% higher than max

    // X Axis
    for (double stockPrice = startPrice; stockPrice <= endPrice; stockPrice += 0.5) {
        
        double theoretical_totalPayoff = 0;
        double theoretical_delta = 0;
        double theoretical_gamma = 0;
        double theoretical_vega = 0;
        double theoretical_rho = 0;

        double implied_totalPayoff = 0;
        double implied_delta = 0;
        double implied_gamma = 0;
        double implied_vega = 0;
        double implied_rho = 0;

        for (const auto& optDetail : options) {

            // Convert days to years
            double T = optDetail.daysTillMaturity / 365.0;

            // Create an instance of BlackScholes using the details from the current option
            BlackScholes option(optDetail.stockPrice, optDetail.strikePrice, optDetail.riskFreeRate, T, optDetail.annualizedVolatility);

            if (optDetail.type == 1) {

                // Y Axis
                theoretical_totalPayoff += optDetail.quantity * (callInstristicValue(stockPrice, optDetail.strikePrice) - optDetail.theoreticalPremium);
                implied_totalPayoff += optDetail.quantity * (callInstristicValue(stockPrice, optDetail.strikePrice) - optDetail.impliedPremium);
            }
            else if (optDetail.type == 0) {

                // Y Axis
                theoretical_totalPayoff += optDetail.quantity * (putInstristicValue(stockPrice, optDetail.strikePrice) - optDetail.theoreticalPremium);
                implied_totalPayoff += optDetail.quantity * (putInstristicValue(stockPrice, optDetail.strikePrice) - optDetail.impliedPremium);
            }
        }

        dataFile << stockPrice << " " << theoretical_totalPayoff << " " << implied_totalPayoff << std::endl;
    }

    dataFile.close();

    char currentDir[MAX_PATH];
    GetCurrentDirectoryA(MAX_PATH, currentDir);

    // Plot the graph using gnuplot.
    std::string command =
        "cd \"C:\\Program Files\\gnuplot\\bin\" && "
        "gnuplot -persist -e \"set arrow from graph 0,first 0 to graph 1,first 0 nohead; "
        "plot '" + std::string(currentDir) + "\\data.txt' using 1:2 with lines title 'Theoretical', '"
        + std::string(currentDir) + "\\data.txt' using 1:3 with lines title 'Implied'\"";


    system(command.c_str());

    return 0;
}