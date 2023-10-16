#include <iostream>
#include <cmath>
#include <iomanip>

class BlackScholesOption {
private:
    double S, K, r, T, sigma;
    double d1, d2;

    void computeD1D2() {
        d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
        d2 = d1 - sigma * sqrt(T);
    }

    double N(double x) const {
        return 0.5 * erfc(-x * std::sqrt(0.5));
    }

public:
    BlackScholesOption(double S, double K, double r, double T, double sigma)
        : S(S), K(K), r(r), T(T), sigma(sigma) {
        computeD1D2();
    }

    double callPrice() const {
        return S * N(d1) - K * exp(-r * T) * N(d2);
    }

    double putPrice() const {
        return K * exp(-r * T) * N(-d2) - S * N(-d1);
    }

    double callDelta() const {
        return N(d1);
    }

    double putDelta() const {
        return N(d1) - 1;
    }
};

int main() {

    /*
    Mathematical notations:

    C = Call option price
    S = Current stock price
    K = Strike price
    r = Risk-free interest rate
    t = Time to maturity
    N = Normal Distribution

    */

    double S, K, r, t, sigma;

    char optionType;
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

    double T = t / 365;  // Convert days to years

    BlackScholesOption option(S, K, r, T, sigma);

    std::cout << std::fixed << std::setprecision(2);
    if (optionType == 'C' || optionType == 'c') {
        std::cout << "The theoretical call option price is: $" << option.callPrice() << std::endl;
        std::cout << "Delta for the call option is: " << option.callDelta() << std::endl;
    }
    else if (optionType == 'P' || optionType == 'p') {
        std::cout << "The theoretical put option price is: $" << option.putPrice() << std::endl;
        std::cout << "Delta for the put option is: " << option.putDelta() << std::endl;
    }
    else {
        std::cout << "Invalid option type entered." << std::endl;
    }

    return 0;
}



