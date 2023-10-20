#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include "Definitions.h"
#include <windows.h>
#include <string>
#include <list>

using namespace std;

// Define M_PI if not already defined. Represents the value of Pi.
#ifndef M_PI
#define M_PI 3.14159
#endif


// Function to calculate the payoff for a call option.
double callInstristicValue(double stockPrice, double strikePrice) {
    return max<double>(0.0, stockPrice - strikePrice);
}

// Function to Calculate the payoff for a put option.
double putInstristicValue(double stockPrice, double strikePrice) {
    return max<double>(0.0, strikePrice - stockPrice);
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
        return 0.5 * erfc(-x * sqrt(0.5));
    }

    /**
     * @brief Probability density function of the standard normal distribution.
     * @param x Value to compute the PDF for.
     * @return Value of the PDF at x.
     */
    double n(double x) const {
        return (1.0 / sqrt(2.0 * M_PI)) * exp(-0.5 * x * x);
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

/**
* @brief Gets data from the user about the option position
*  
* Puts all data in options vector
* 
* @return options vector
*/
vector<OptionDetails> getInputFromUser() {
    
    int numberOfOptions;
    vector<OptionDetails> options;

    cout << " -- Global Inputs -- " << endl;
    cout << "How many different options do you have? ";
    cin >> numberOfOptions;
    options.resize(numberOfOptions);

    // Collecting some data outside the loop to not
    // ask the same thing over and over again
    double commonRiskFreeRate;
    cout << "What is the risk free rate? (if 1%, then enter 0.01) - ";
    cin >> commonRiskFreeRate;

    double currentStockPrice;
    cout << "What is the current underlying price? - ";
    cin >> currentStockPrice;

    double underlying;
    cout << "How much of underlying do you own? - ";
    cin >> underlying;

    // Loops through quantity of different options specified
    for (int i = 0; i < numberOfOptions; i++) {
        cout << "\nDetails for Option " << i + 1 << ":" << endl;
        cout << "What is the option? (1 for call & 0 for put) - ";
        cin >> options[i].type;
        cout << "What is the quantity? (Negative for sold) - ";
        cin >> options[i].quantity;
        cout << "What is the current price for the option? - ";
        cin >> options[i].impliedPremium;
        cout << "What is the strike price? - ";
        cin >> options[i].strikePrice;
        cout << "How many days till maturity? - ";
        cin >> options[i].daysTillMaturity;
        cout << "What is the annualized volatility? (if 10%, then enter 0.1) - ";
        cin >> options[i].annualizedVolatility;

        // Adds the outside-loop inputs for each option
        options[i].riskFreeRate = commonRiskFreeRate;
        options[i].stockPrice = currentStockPrice;
        options[i].underlying = underlying;
    }

    return options;
}

/**
* @brief Random data consisting of butterfly spread
* 
* This is to simplify testing of the program, so there
* is no need to enter the same data over and over
* again when developing or testing
* 
* @return options vector
*/
vector<OptionDetails> getButterflySpreadData() {
    vector<OptionDetails> options(3);

    double commonRiskFreeRate = 0.01;
    double currentStockPrice = 100.0;

    // Buy one lower strike price call
    options[0].type = 1; // Call
    options[0].quantity = 1;
    options[0].impliedPremium = 1.0;
    options[0].strikePrice = 95.0;
    options[0].daysTillMaturity = 30;
    options[0].annualizedVolatility = 0.2;
    options[0].riskFreeRate = commonRiskFreeRate;
    options[0].stockPrice = currentStockPrice;

    // Sell two middle strike price calls
    options[1].type = 1;
    options[1].quantity = -2;
    options[1].impliedPremium = 2.0;
    options[1].strikePrice = 100.0;
    options[1].daysTillMaturity = 30;
    options[1].annualizedVolatility = 0.2;
    options[1].riskFreeRate = commonRiskFreeRate;
    options[1].stockPrice = currentStockPrice;

    // Buy one higher strike price call
    options[2].type = 1;
    options[2].quantity = 1;
    options[2].impliedPremium = 0.5;
    options[2].strikePrice = 105.0;
    options[2].daysTillMaturity = 30;
    options[2].annualizedVolatility = 0.2;
    options[2].riskFreeRate = commonRiskFreeRate;
    options[2].stockPrice = currentStockPrice;

    return options;
}

/**
* @brief main function of the code overseeing all
* 
* @structure:
*   - Gets inputs
*   
*   ( Goal: Price each option, display greeks )
*   - Loops through all options:
*       > Prices each option
*       > Displays greeks; price for each option
* 
*   ( Goal: Draw position & Calculate greeks )
*   - Loops through all options:
*       > Prices each option
*       > Calculates payoffs for each price
*       > Charts payoffs
* 
*   ( Goal: See position PnL given volatility )
*   - Loops through all options:
*       > Prices each option, uses different volatility
*         but the same price
*       > Calculates payoffs for the expiration date
*         for each strike price
*       > Charts payoffs
* 
*/
int main() {

    cout << "Option Pricing & Position Evaluation Model. Compatible for Equity.";
    cout << "" << endl;
    int useFakeData = 1;
    cout << "Use fake data for showcase purposes? (1=yes; 0=no): ";
    cin >> useFakeData;

    vector<OptionDetails> options;
    if (useFakeData == 1) {
        options = getButterflySpreadData();
    }
    else if (useFakeData == 0) {
        options = getInputFromUser();
    }

    // 1st loop. Goal: Price each option, display greeks //

    int x = 0;
    cout << fixed << setprecision(2);
    for (const auto& optDetail : options) {
        cout << "" << endl;
        cout << "" << endl;
        cout << "-- Option " << x + 1 << " -- " << endl;
        cout << "" << endl;

        // Create an instance of BlackScholes using the details from the current option
        double T = optDetail.daysTillMaturity / 365.0;
        BlackScholes option(optDetail.stockPrice, optDetail.strikePrice, optDetail.riskFreeRate, T, optDetail.annualizedVolatility);

        if (optDetail.type == 1) {

            options[x].theoreticalPremium = option.callPrice();

            cout << "The current call option price is: $" << optDetail.impliedPremium << endl;
            cout << "The theoretical call option price is: $" << options[x].theoreticalPremium << endl;
            cout << "Difference: $" << option.callPrice() - optDetail.impliedPremium << endl;
            cout << "" << endl;
            cout << "Delta for the call option is: " << option.callDelta() << endl;
            cout << "Gamma for the option is: " << option.gamma() << endl;
            cout << "Theta for the call option is: $" << option.callTheta() / 365 << " per day" << endl;
            cout << "Vega for the call option is: " << option.optionVega() / 100 << " per 1% volatility change" << endl;
            cout << "Rho for the call option is: " << option.callRho() << " per 1% interest rate change" << endl;

        }
        else if (optDetail.type == 0) {

            options[x].theoreticalPremium = option.putPrice();

            cout << "The current put option price is: $" << optDetail.impliedPremium << endl;
            cout << "The theoretical put option price is: $" << options[x].theoreticalPremium << endl;
            cout << "Difference: $" << option.putPrice() - optDetail.impliedPremium << endl;
            cout << "" << endl;
            cout << "Delta for the put option is: " << option.putDelta() << endl;
            cout << "Gamma for the option is: " << option.gamma() << endl;
            cout << "Theta for the put option is: $" << option.putTheta() / 365 << " per day" << endl;
            cout << "Vega for the put option is: " << option.optionVega() / 100 << " per 1% volatility change" << endl;
            cout << "Rho for the put option is: " << option.putRho() << " per 1% interest rate change" << endl;
        }
        else {
            cout << "Invalid option type entered." << endl;
        }
        x++;
    }

    // 2nd loop. Goal: Draw position & Calculate greeks //

    ofstream dataFile("data.txt");

    // Setting the range for the chart
    double minStrikePrice = 1e30;
    double maxStrikePrice = -1e30;

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

    double theoretical_totalPayoff = 0;
    double theoretical_delta = 0;
    double theoretical_gamma = 0;
    double theoretical_theta = 0;
    double theoretical_vega = 0;
    double theoretical_rho = 0;

    double implied_totalPayoff = 0;
    double implied_delta = 0;
    double implied_gamma = 0;
    double implied_vega = 0;
    double implied_rho = 0;

    // X Axis
    for (double stockPrice = startPrice; stockPrice <= endPrice; stockPrice += 0.5) {

        theoretical_totalPayoff = 0;
        theoretical_delta = 0;
        theoretical_gamma = 0;
        theoretical_theta = 0;
        theoretical_vega = 0;
        theoretical_rho = 0;

        implied_totalPayoff = 0;

        for (const auto& optDetail : options) {

            // Create an instance of BlackScholes using the details from the current option
            double T = optDetail.daysTillMaturity / 365.0;
            BlackScholes option(optDetail.stockPrice, optDetail.strikePrice, optDetail.riskFreeRate, T, optDetail.annualizedVolatility);

            if (optDetail.type == 1) {

                // Y Axis
                theoretical_totalPayoff += (optDetail.quantity * (callInstristicValue(stockPrice, optDetail.strikePrice) - optDetail.theoreticalPremium)) + ((1 / 100) * optDetail.underlying * stockPrice);
                implied_totalPayoff += (optDetail.quantity * (callInstristicValue(stockPrice, optDetail.strikePrice) - optDetail.impliedPremium)) + ((1 / 100) * optDetail.underlying * stockPrice);
                theoretical_delta += option.callDelta() + optDetail.underlying;
                theoretical_gamma += option.gamma();
                theoretical_theta += option.callTheta();
                theoretical_vega += option.optionVega();
                theoretical_rho += option.callRho();
            }
            else if (optDetail.type == 0) {

                // Y Axis
                theoretical_totalPayoff += (optDetail.quantity * (putInstristicValue(stockPrice, optDetail.strikePrice) - optDetail.theoreticalPremium)) + ((1 / 100) * optDetail.underlying * stockPrice);
                implied_totalPayoff += (optDetail.quantity * (putInstristicValue(stockPrice, optDetail.strikePrice) - optDetail.impliedPremium)) + ((1 / 100) * optDetail.underlying * stockPrice);
                theoretical_delta += option.putDelta() + optDetail.underlying;
                theoretical_gamma += option.gamma();
                theoretical_theta += option.putTheta();
                theoretical_vega += option.optionVega();
                theoretical_rho += option.putRho();
            }
        }
        dataFile << stockPrice << " " << theoretical_totalPayoff << " " << implied_totalPayoff << endl;
    }

    cout << "" << endl;
    cout << "" << endl;
    cout << "-- Position Greeks -- " << endl;
    cout << "" << endl;
    cout << "Position Delta: " << theoretical_delta << endl;
    cout << "Position Gamma: " << theoretical_gamma << endl;
    cout << "Position Theta " << theoretical_theta / 100 << endl;
    cout << "Position Vega: " << theoretical_vega / 100 << endl;
    cout << "Position Rho: " << theoretical_rho / 100 << endl;

    dataFile.close();

    char currentDir[MAX_PATH];
    GetCurrentDirectoryA(MAX_PATH, currentDir);

    // Plot formatting
    string gnu_path = "C:\\Program Files\\gnuplot\\bin";
    string data_path = string(currentDir) + "\\data.txt";

    string chart_settings =
        "set title 'Position Payoff Graph';"
        "set title font 'Arial,20';"
        "set key title 'Payoffs';"
        "set grid;"
        "set xrange[" + to_string(startPrice) + ":" + to_string(endPrice) + "];"
        "set arrow from graph 0, first 0 to graph 1, first 0 nohead;";

    string theoretical = "'" + data_path + "' using 1:2 with lines title 'Theoretical'";
    string implied = ", '" + data_path + "' using 1:3 with lines title 'Implied'";

    string plot_command = chart_settings + "plot " + theoretical + implied;
    string cmd_command_1 = "cd \"" + gnu_path + "\" && gnuplot -persist -e \"" + plot_command + "\"";
    system(cmd_command_1.c_str());

    // 3rd loop. Goal: See position PnL given volatility //

    ofstream dataFileVol("data_vol.txt");

    list<double> strike_prices;
    for (const auto& optDetail : options) {
        strike_prices.push_back(optDetail.strikePrice);
    }

    for (double volatility = 0.05; volatility < 0.3; volatility += 0.01) {
        dataFileVol << volatility;

        for (const auto& price : strike_prices) {
            double theoretical_totalPayoff2 = 0;

            for (const auto& optDetail : options) {
                double T = optDetail.daysTillMaturity / 365.0;
                BlackScholes option2(price, optDetail.strikePrice, optDetail.riskFreeRate, T, volatility);

                if (optDetail.type == 1) {
                    theoretical_totalPayoff2 += (optDetail.quantity * (callInstristicValue(price, optDetail.strikePrice) - option2.callPrice())) + ((1 / 100) * optDetail.underlying * price);
                }
                else if (optDetail.type == 0) {
                    theoretical_totalPayoff2 += (optDetail.quantity * (putInstristicValue(price, optDetail.strikePrice) - option2.putPrice())) + ((1 / 100) * optDetail.underlying * price);
                }
            }
            dataFileVol << " " << theoretical_totalPayoff2;
        }
        dataFileVol << endl;
    }
    dataFileVol.close();

    // Plot formatting
    string plotCommand = "plot ";
    string data_vol_path = string(currentDir) + "\\data_vol.txt";
    size_t totalColumns = strike_prices.size() + 1;
    string chart_settings_ =
        "set title 'Position PnL vs Volatility, Given Price';"
        "set title font 'Arial,20';"
        "set key title 'Price at Expiration';"
        "set grid;"
        "set xrange[" + to_string(0.05) + ":" + to_string(0.3) + "];"
        "set arrow from graph 0, first 0 to graph 1, first 0 nohead;";

    for (size_t column = 2; column <= totalColumns; ++column) {
        auto it = strike_prices.begin();
        advance(it, column - 2);
        plotCommand += "'" + data_vol_path + "' using 1:" + to_string(column) + " with lines title 'Strike " + to_string(static_cast<int>(round(*it))) + "'";

        if (column != totalColumns) {
            plotCommand += ", ";
        }
    }

    string cmd_command_2 =
        "cd \"" + gnu_path + "\" && "
        "gnuplot -persist -e \""
        + chart_settings_
        + plotCommand + "\"";

    system(cmd_command_2.c_str());

    // 4th loop. Goal: See Position PnL given Time //

    ofstream dataFileDates("data_dates.txt");

    int expiration_time = -1e20;
    for (const auto& optDetail : options) {
        if (optDetail.daysTillMaturity > expiration_time) {
            expiration_time = optDetail.daysTillMaturity;
        }
    }

    for (const auto& optDetail : options) {
        for (double time = 0; time < expiration_time; time++) {
            dataFileDates << time;

            for (const auto& price_ : strike_prices) {
                double theoretical_totalPayoff_dates = 0;

                for (const auto& optDetail : options) {

                    double T = time / 365.0;
                    BlackScholes option3(price_, optDetail.strikePrice, optDetail.riskFreeRate, T, optDetail.annualizedVolatility);

                    if (optDetail.type == 1) {
                        theoretical_totalPayoff_dates += (optDetail.quantity * (callInstristicValue(price_, optDetail.strikePrice) - option3.callPrice())) + ((1 / 100) * optDetail.underlying * price_);
                    }
                    else if (optDetail.type == 0) {
                        theoretical_totalPayoff_dates += (optDetail.quantity * (putInstristicValue(price_, optDetail.strikePrice) - option3.putPrice())) + ((1 / 100) * optDetail.underlying * price_);
                    }
                }
                dataFileDates << " " << theoretical_totalPayoff_dates;
            }
            dataFileDates << endl;
        }
        dataFileDates.close();
    }

    // Plot formatting
    string plotCommandDates = "plot ";
    string data_dates_path = string(currentDir) + "\\data_dates.txt";
    string chart_settings_dates =
        "set title 'Position Premium vs Time, Given Price';"
        "set title font 'Arial,20';"
        "set key title 'Price at Expiration';"
        "set grid;"
        "set xrange[" + to_string(0) + ":" + to_string(expiration_time) + "];"
        "set arrow from graph 0, first 0 to graph 1, first 0 nohead;";

    for (size_t column = 2; column <= totalColumns; ++column) {
        auto it = strike_prices.begin();
        advance(it, column - 2);
        plotCommandDates += "'" + data_dates_path + "' using 1:" + to_string(column) + " with lines title 'Strike " + to_string(static_cast<int>(round(*it))) + "'";

        if (column != totalColumns) {
            plotCommandDates += ", ";
        }
    }
    string cmd_command_dates =
        "cd \"" + gnu_path + "\" && "
        "gnuplot -persist -e \""
        + chart_settings_dates
        + plotCommandDates + "\"";

    system(cmd_command_dates.c_str());
}
