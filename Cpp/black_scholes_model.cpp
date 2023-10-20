#include <iostream>
#include <cmath>
#include <iomanip>

// Define M_PI if not already defined. Represents the value of Pi.
#ifndef M_PI
#define M_PI 3.14159
#endif

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

int main() {
    // Inputs from the user
    double S, K, r, t, sigma;
    char optionType;

    // Prompt user for required input
    std::cout << "What is the option? (Enter C for call & P for put) - ";
    std::cin >> optionType;
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
    std::cout << "-----------------------------------------------------------" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "Option data: " << std::endl;

    // Convert days to years
    double T = t / 365;

    // Create an instance of BlackScholes
    BlackScholes option(S, K, r, T, sigma);

    // Display option data based on user's choice
    std::cout << std::fixed << std::setprecision(2);

    if (optionType == 'C' || optionType == 'c') {
        std::cout << "The theoretical call option price is: $" << option.callPrice() << std::endl;
        std::cout << "Delta for the call option is: " << option.callDelta() << std::endl;
        std::cout << "Gamma for the option is: " << option.gamma() << std::endl;
        std::cout << "Theta for the call option is: $" << option.callTheta() / 365 << " per day" << std::endl;
        std::cout << "Vega for the call option is: " << option.optionVega() / 100 << " per 1% volatility change" << std::endl;
        std::cout << "Rho for the call option is: " << option.callRho() << " per 1% interest rate change" << std::endl;
    }
    else if (optionType == 'P' || optionType == 'p') {
        std::cout << "The theoretical put option price is: $" << option.putPrice() << std::endl;
        std::cout << "Delta for the put option is: " << option.putDelta() << std::endl;
        std::cout << "Gamma for the option is: " << option.gamma() << std::endl;
        std::cout << "Theta for the put option is: $" << option.putTheta() / 365 << " per day" << std::endl;
        std::cout << "Vega for the put option is: " << option.optionVega() / 100 << " per 1% volatility change" << std::endl;
        std::cout << "Rho for the put option is: " << option.putRho() << " per 1% interest rate change" << std::endl;
    }
    else {
        std::cout << "Invalid option type entered." << std::endl;
    }

    return 0;
}
