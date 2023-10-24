
#include <functional>
#include <algorithm>
#include <windows.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <list>
#include <map>

#include "Definitions.h"
#include "OptionDetails.h"

using namespace std;



// IMPORTANT * IMPORTANT * IMPORTANT * IMPORTANT * IMPORTANT * IMPORTANT * IMPORTANT *
// ----------------------------------------------------------------------------------|
// --- SCRIPT REQUIRES GNUPLOT TO BE INSTALLED                                       | 
// --- GNUPLOT PATH SHOULD BE DEFINED BELOW                                          | 
string gnu_path = "C:\\Program Files\\gnuplot\\bin"; //                              |
// ----------------------------------------------------------------------------------|
// IMPORTANT * IMPORTANT * IMPORTANT * IMPORTANT * IMPORTANT * IMPORTANT * IMPORTANT * 



// Define M_PI if not already defined. Represents the value of Pi.
#ifndef M_PI
#define M_PI 3.14159
#endif

// Function to calculate the intristic value for a call option.
double callInstristicValue(double stockPrice, double strikePrice) {
    return max<double>(0.0, stockPrice - strikePrice);
}

// Function to Calculate the intristic value for a put option.
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

    // Intermediate calculations for option pricing of Black Scholes formula
    double d1, d2;

    /**
     * @brief Compute the d1 and d2 values used in Black-Scholes formula.
     */
    void computeD1D2() {
        d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
        d2 = d1 - sigma * sqrt(T);
    }

    /**
     * @brief Cumulative distribution function of the standard normal distribution.
     * @param x Value to compute the CDF for.
     * @return Cumulative distribution from negative infinity to x.
     * 
     * N(x) gives the probability that a random variable following a 
     * standard normal distribution is less than or equal to x
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
*   ( Goal: See position payoff for different dates )
*   - Loops through all options:
*       > Prices each option, uses different expiration
*         and price price
*       > Calculates payoffs
*       > Charts payoffs
*
*/
int main() {

    char currentDir[MAX_PATH];
    GetCurrentDirectoryA(MAX_PATH, currentDir);

    cout << "|--------------------------------------------------------------------|" << endl;
    cout << "|                                                                    |" << endl;
    cout << "|                    { Made by Matas Urbonavicius }                  |" << endl;
    cout << "| ================================================================== |" << endl;
    cout << "| Option Pricing & Position Evaluation Model. Compatible for Equity. |" << endl;
    cout << "| ================================================================== |" << endl;
    cout << "|                                                                    |" << endl;
    cout << "|--------------------------------------------------------------------|" << endl;
    cout << "" << endl;
    cout << "" << endl;

    cout << "|--------------------------------------------------------------------|" << endl;
    cout << "| Use fake data for showcase purposes? (1=yes; 0=no): " << endl;
    cout << "| Input: ";
    int useFakeData = 1;
    cin >> useFakeData;
    cout << "|--------------------------------------------------------------------|" << endl;

    vector<OptionDetails> options;

    if (useFakeData == 1) {

        // Mapping all strategies for demo purposes
        map<string, function<vector<OptionDetails>()>> strategyFunctions = {
            {"butterfly", getButterflySpreadData},
            {"long_call", LongCallOption},
            {"short_call", ShortCallOption},
            {"long_put", LongPutOption},
            {"short_put", ShortPutOption},
            {"call_backspread", CallBackspread},
            {"put_backspread", PutBackspread},
            {"ratio_vertical_spread", RatioVerticalSpread},
            {"straddle", Straddle},
            {"strangle", Strangle},
            {"call_time_spread", CallTimeSpread}
        };

        cout << "|--------------------------------------------------------------------|" << endl;
        cout << "| Which strategy to demonstrate?" << endl;
        cout << "| Selection:" << endl;
        for (const auto& strategyPair : strategyFunctions) {
            cout << "| - " << strategyPair.first << endl;
        }
        cout << "|--------------------------------------------------------------------|" << endl;

        cout << "| Input: ";
        string chosenStrategy;
        cin >> chosenStrategy;

        // Select strategy
        if (strategyFunctions.find(chosenStrategy) != strategyFunctions.end()) {
            options = strategyFunctions[chosenStrategy]();
        }
        else {
            cout << "| Check Spelling. Invalid." << endl;
            return 1;
        }
    }
    else if (useFakeData == 0) {
        options = getInputFromUser();
    }

    // --- 
    // 1st loop. Goal: Price each option, calculate greeks //
    // ---

    int x = 0;
    cout << fixed << setprecision(2);
    for (const auto& optDetail : options) {
        cout << "" << endl;
        cout << "|--------------------------------------------------------------------|" << endl;
        cout << "| -- Option " << x + 1 << " -- " << endl;
        cout << "|" << endl;

        double T = optDetail.daysTillMaturity / 365.0;
        BlackScholes option(optDetail.stockPrice, optDetail.strikePrice, optDetail.riskFreeRate, T, optDetail.annualizedVolatility);

        if (optDetail.type == 1) {
            cout << "| The current call option price is: $" << optDetail.impliedPremium << endl;
            cout << "| The theoretical Black Scholes call option price is: $" << option.callPrice() << endl;
            cout << "| Delta for the call option is: " << option.callDelta() << endl;
            cout << "| Gamma for the option is: " << option.gamma() << endl;
            cout << "| Theta for the call option is: $" << option.callTheta() / 365 << " per day" << endl;
            cout << "| Vega for the call option is: " << option.optionVega() / 100 << " per 1% volatility change" << endl;
            cout << "| Rho for the call option is: " << option.callRho() << " per 1% interest rate change" << endl;
            cout << "|--------------------------------------------------------------------|" << endl;

        }
        else if (optDetail.type == 0) {
            cout << "| The current put option price is: $" << optDetail.impliedPremium << endl;
            cout << "| The theoretical Black Scholes put option price is: $" << option.putPrice() << endl;
            cout << "| Delta for the put option is: " << option.putDelta() << endl;
            cout << "| Gamma for the option is: " << option.gamma() << endl;
            cout << "| Theta for the put option is: $" << option.putTheta() / 365 << " per day" << endl;
            cout << "| Vega for the put option is: " << option.optionVega() / 100 << " per 1% volatility change" << endl;
            cout << "| Rho for the put option is: " << option.putRho() << " per 1% interest rate change" << endl;
            cout << "|--------------------------------------------------------------------|" << endl;
        }
        else {
            cout << "Invalid option type entered." << endl;
        }
        x++;
    }

    // ---
    // 2nd loop. Goal: Draw position at expiration & Calculate greeks //
    // ---

    ofstream Main_Payoff_Graph_Data_File("Main_Payoff_Graph_Data.txt");

    if (!Main_Payoff_Graph_Data_File.is_open()) {
        cerr << "Error opening file." << endl; // Check if file opened correctly
        return 1;
    }

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

    double startPrice = minStrikePrice * 0.90; // 10% lower than min
    double endPrice = maxStrikePrice * 1.10;   // 10% higher than max

    double theoretical_totalPayoff = 0;
    double theoretical_delta = 0;
    double theoretical_gamma = 0;
    double theoretical_theta = 0;
    double theoretical_vega = 0;
    double theoretical_rho = 0;
    double implied_totalPayoff = 0;

    // For some reason GnuPlot shows really crappy Y range, so i will set it myself
    double minYValue = 1e30;
    double maxYValue = -1e30;

    for (double stockPrice = startPrice; stockPrice <= endPrice; stockPrice += 0.1) {
        theoretical_totalPayoff = 0;
        theoretical_delta = 0;
        theoretical_gamma = 0;
        theoretical_theta = 0;
        theoretical_vega = 0;
        theoretical_rho = 0;
        implied_totalPayoff = 0;

        for (const auto& optDetail : options) {

            double T = optDetail.daysTillMaturity / 365.0;
            BlackScholes option(optDetail.stockPrice, optDetail.strikePrice, optDetail.riskFreeRate, T, optDetail.annualizedVolatility);

            if (optDetail.type == 1) {
                theoretical_totalPayoff += (optDetail.quantity * (callInstristicValue(stockPrice, optDetail.strikePrice) - option.callPrice())) + (optDetail.underlying * (stockPrice - optDetail.stockPrice));
                implied_totalPayoff += (optDetail.quantity * (callInstristicValue(stockPrice, optDetail.strikePrice) - optDetail.impliedPremium)) + (optDetail.underlying * (stockPrice - optDetail.stockPrice));
                theoretical_delta += option.callDelta() + optDetail.underlying;
                theoretical_gamma += option.gamma();
                theoretical_theta += option.callTheta();
                theoretical_vega += option.optionVega();
                theoretical_rho += option.callRho();
            }
            else if (optDetail.type == 0) {
                theoretical_totalPayoff += (optDetail.quantity * (putInstristicValue(stockPrice, optDetail.strikePrice) - option.putPrice())) + (optDetail.underlying * (stockPrice - optDetail.stockPrice));
                implied_totalPayoff += (optDetail.quantity * (putInstristicValue(stockPrice, optDetail.strikePrice) - optDetail.impliedPremium)) + (optDetail.underlying * (stockPrice - optDetail.stockPrice));
                theoretical_delta += option.putDelta() + optDetail.underlying;
                theoretical_gamma += option.gamma();
                theoretical_theta += option.putTheta();
                theoretical_vega += option.optionVega();
                theoretical_rho += option.putRho();
            }

            if (theoretical_totalPayoff < minYValue) {
                minYValue = theoretical_totalPayoff;
            }
            if (theoretical_totalPayoff > maxYValue) {
                maxYValue = theoretical_totalPayoff;
            }
            if (implied_totalPayoff < minYValue) {
                minYValue = implied_totalPayoff;
            }
            if (implied_totalPayoff > maxYValue) {
                maxYValue = implied_totalPayoff;
            }
        }
        Main_Payoff_Graph_Data_File << stockPrice << " " << theoretical_totalPayoff << " " << implied_totalPayoff << endl;
    }
    Main_Payoff_Graph_Data_File.close();
   

    minYValue *= 1.10;
    maxYValue *= 1.10;

    cout << "" << endl;
    cout << "|--------------------------------------------------------------------|" << endl;
    cout << "| -- Position Greeks -- " << endl;
    cout << "|" << endl;
    cout << "| Position Delta: " << theoretical_delta << endl;
    cout << "| Position Gamma: " << theoretical_gamma << endl;
    cout << "| Position Theta " << theoretical_theta / 100 << endl;
    cout << "| Position Vega: " << theoretical_vega / 100 << endl;
    cout << "| Position Rho: " << theoretical_rho / 100 << endl;
    cout << "|...Generating Charts" << endl;
    cout << "|--------------------------------------------------------------------|" << endl;

    string Main_Payoff_Graph_Data_Path = string(currentDir) + "\\Main_Payoff_Graph_Data.txt";
    string chart_settings =
    "set object 1 rectangle from screen 0,0 to screen 1,1 fillcolor rgb 'white' behind;"
    "set title 'Position Payoff Graph At Expiration';"
    "set title font 'Arial,15';"
    "set title textcolor rgb 'black';"
    "set key title 'Payoffs';"
    "set key textcolor rgb 'black';"
    "set grid lc rgb 'gray';"
    "set xlabel 'Price' textcolor rgb 'black';"
    "set ylabel 'Profit / Loss' textcolor rgb 'black';"
    "set xtics textcolor rgb 'black';"
    "set ytics textcolor rgb 'black';"
    "set yrange [" + to_string(minYValue) + ":" + to_string(maxYValue) + "];"
    "set xrange[" + to_string(startPrice) + ":" + to_string(endPrice) + "];"
    "set arrow from graph 0, first 0 to graph 1, first 0 nohead lc rgb 'black';"
    "set border lc rgb 'black';";

    string theoretical_price_at_expiry = "'" + Main_Payoff_Graph_Data_Path + "' using 1:2 with lines title 'Theoretical Black-Scholes'";
    string implied_price_at_expiry = ", '" + Main_Payoff_Graph_Data_Path + "' using 1:3 with lines title 'Implied'";

    string plot_command_firstloop = chart_settings + "plot " + theoretical_price_at_expiry + implied_price_at_expiry;
    string cmd_command_firstloop = "cd \"" + gnu_path + "\" && gnuplot -persist -e \"" + plot_command_firstloop + "\"";
    system(cmd_command_firstloop.c_str());

    // ---
    // 3rd loop. Goal: See position PnL given realized volatility at given prices at expiration
    // ---

    ofstream Payoff_vs_Volatility_Graph_Data_File("Payoff_vs_Volatility_Graph_Data.txt");

    list<double> strike_prices;
    for (const auto& optDetail : options) {
        strike_prices.push_back(optDetail.strikePrice);
    }

    double Payoff_Given_Volatility_Market = 0;
    double Payoff_Given_Volatility_Theory = 0;

    for (double volatility = 0.05; volatility < 0.3; volatility += 0.001) {
        Payoff_vs_Volatility_Graph_Data_File << volatility;

        Payoff_Given_Volatility_Market = 0;
        Payoff_Given_Volatility_Theory = 0;

        for (const auto& optDetail : options) {
            double T = optDetail.daysTillMaturity / 365.0;
            BlackScholes option(optDetail.stockPrice, optDetail.strikePrice, optDetail.riskFreeRate, T, volatility);
            BlackScholes option_at_buy(optDetail.stockPrice, optDetail.strikePrice, optDetail.riskFreeRate, T, optDetail.annualizedVolatility);

            if (optDetail.type == 1) {
                Payoff_Given_Volatility_Market += optDetail.quantity * (option.callPrice() - optDetail.impliedPremium) + (optDetail.underlying * (optDetail.stockPrice - optDetail.stockPrice));
                Payoff_Given_Volatility_Theory += optDetail.quantity * (option.callPrice() - option_at_buy.callPrice()) + (optDetail.underlying * (optDetail.stockPrice - optDetail.stockPrice));
            }
            else if (optDetail.type == 0) {
                Payoff_Given_Volatility_Market += optDetail.quantity * (option.putPrice() - optDetail.impliedPremium) + (optDetail.underlying * (optDetail.stockPrice - optDetail.stockPrice));
                Payoff_Given_Volatility_Theory += optDetail.quantity * (option.putPrice() - option_at_buy.putPrice()) + (optDetail.underlying * (optDetail.stockPrice - optDetail.stockPrice));
            }
        }
        Payoff_vs_Volatility_Graph_Data_File << " " << Payoff_Given_Volatility_Market << " " << Payoff_Given_Volatility_Theory << endl;
    }
    Payoff_vs_Volatility_Graph_Data_File.close();

    string Payoff_vs_Volatility_Graph_Path = string(currentDir) + "\\Payoff_vs_Volatility_Graph_Data.txt";
    string Payoff_vs_Volatility_Graph_Settings =
        "set object 1 rectangle from screen 0,0 to screen 1,1 fillcolor rgb 'white' behind;"
        "set title 'Position PnL vs Volatility, At Current Price At Expiration';"
        "set title font 'Arial,15';"
        "set title textcolor rgb 'black';"
        "set key title 'Payoffs';"
        "set key textcolor rgb 'black';"
        "set grid lc rgb 'gray';"
        "set xlabel 'Volatility' textcolor rgb 'black';"
        "set ylabel 'Profit / Loss' textcolor rgb 'black';"
        "set xtics textcolor rgb 'black';"
        "set ytics textcolor rgb 'black';"
        "set xrange[" + to_string(0.05) + ":" + to_string(0.3) + "];"
        "set arrow from graph 0, first 0 to graph 1, first 0 nohead lc rgb 'black';"
        "set border lc rgb 'black';";

    string Payoff_Market = "'" + Payoff_vs_Volatility_Graph_Path + "' using 1:2 with lines title 'Market'";
    string Payoff_Theory = ", '" + Payoff_vs_Volatility_Graph_Path + "' using 1:3 with lines title 'Theory'";

    string plot_command_secondloop = Payoff_vs_Volatility_Graph_Settings + "plot " + Payoff_Market + Payoff_Theory;
    string cmd_command_secondloop = "cd \"" + gnu_path + "\" && gnuplot -persist -e \"" + plot_command_secondloop + "\"";
    system(cmd_command_secondloop.c_str());

    // ---
    // 4th loop. Goal: Visualize position payoff at different time stamps
    // ---

    ofstream Payoff_vs_Time_Graph_File("Payoff_vs_Time_Graph_Data.txt");
    double Payoff_At_Time_x = 0;
    double Payoff_At_Time_y = 0;
    double Payoff_At_Expiry_ = 0;

    for (double stockPrice = startPrice; stockPrice <= endPrice; stockPrice += 0.5) {
        Payoff_At_Time_x = 0;
        Payoff_At_Time_y = 0;
        Payoff_At_Expiry_ = 0;
        for (const auto& optDetail : options) {
            double T = optDetail.daysTillMaturity / 365.0;
            BlackScholes option(stockPrice, optDetail.strikePrice, optDetail.riskFreeRate, T, optDetail.annualizedVolatility);
            BlackScholes option__(stockPrice, optDetail.strikePrice, optDetail.riskFreeRate, T / 2, optDetail.annualizedVolatility);
            BlackScholes option_(optDetail.stockPrice, optDetail.strikePrice, optDetail.riskFreeRate, T, optDetail.annualizedVolatility);

            if (optDetail.type == 1) {
                Payoff_At_Time_x += optDetail.quantity * option.callPrice() + (optDetail.underlying * (stockPrice - optDetail.stockPrice));
                Payoff_At_Time_y += optDetail.quantity * option__.callPrice() + (optDetail.underlying * (stockPrice - optDetail.stockPrice));
                Payoff_At_Expiry_ += (optDetail.quantity * (callInstristicValue(stockPrice, optDetail.strikePrice) - option_.callPrice())) + (optDetail.underlying * (stockPrice - optDetail.stockPrice));

            }
            else if (optDetail.type == 0) {
                Payoff_At_Time_x += optDetail.quantity * option.putPrice() + (optDetail.underlying * (stockPrice - optDetail.stockPrice));
                Payoff_At_Time_y += optDetail.quantity * option__.putPrice() + (optDetail.underlying * (stockPrice - optDetail.stockPrice));
                Payoff_At_Expiry_ += (optDetail.quantity * (putInstristicValue(stockPrice, optDetail.strikePrice) - option_.putPrice())) + (optDetail.underlying * (stockPrice - optDetail.stockPrice));

            }
        }
        Payoff_vs_Time_Graph_File << stockPrice << " " << Payoff_At_Time_x << " " << Payoff_At_Expiry_ << " " << Payoff_At_Time_y << endl;
    }
    Payoff_vs_Time_Graph_File.close();

    string Payoff_vs_Time_Graph_Data_Path = string(currentDir) + "\\Payoff_vs_Time_Graph_Data.txt";

    string chart_settings2 =
        "set object 1 rectangle from screen 0,0 to screen 1,1 fillcolor rgb 'white' behind;"
        "set title 'Position Payoff Graph at a Given Time vs Expiration';"
        "set title font 'Arial,15';"
        "set title textcolor rgb 'black';"
        "set key title 'Payoffs';"
        "set key textcolor rgb 'black';"
        "set grid lc rgb 'gray';"
        "set xlabel 'Price' textcolor rgb 'black';"
        "set ylabel 'Profit / Loss' textcolor rgb 'black';"
        "set xtics textcolor rgb 'black';"
        "set ytics textcolor rgb 'black';"
        "set xrange[" + to_string(startPrice) + ":" + to_string(endPrice) + "];"
        "set arrow from graph 0, first 0 to graph 1, first 0 nohead lc rgb 'black';"
        "set border lc rgb 'black';";

    string theoretical2 = "'" + Payoff_vs_Time_Graph_Data_Path + "' using 1:2 with lines title 'At Current Time'";
    string theoretical3 = "'" + Payoff_vs_Time_Graph_Data_Path + "' using 1:3 with lines title 'At Expiration Date'";
    string theoretical4 = "'" + Payoff_vs_Time_Graph_Data_Path + "' using 1:4 with lines title 'In the Middle'";

    string plot_command_dates = chart_settings2 + "plot " + theoretical2 + ", " + theoretical3 + ", " + theoretical4;
    string cmd_command_12 = "cd \"" + gnu_path + "\" && gnuplot -persist -e \"" + plot_command_dates + "\"";
    system(cmd_command_12.c_str());

    // ---
    // 5th loop. Goal: visualize position PnL at different expiration dates (specifically for calendar spreads)
    // ---

    ofstream PnL_At_Expirations_File("PnL_At_Expirations_Data.txt");

    // Iterate through each option and calculate its expiration date based on the provided formula
    list<double> expiration_dates;
    for (const auto& optDetail : options) {
        double T = optDetail.daysTillMaturity / 365;

        // Check if T is not already in the expiration_dates list before adding
        if (find(expiration_dates.begin(), expiration_dates.end(), T) == expiration_dates.end()) {
            expiration_dates.push_back(T);
        }
    }

    auto smallest_expiry = min_element(expiration_dates.begin(), expiration_dates.end());
    double PnL_Position = 0;

    /*
    * Only shows this chart if we have multiple expiration dates in our position
    * So:
    * - Diagonal Spreads
    * - Calendar Spreads
    * 
    */
    if (expiration_dates.size() > 1) {

        for (double stockPrice = startPrice; stockPrice <= endPrice; stockPrice += 0.5) {
            PnL_Position = 0;

            for (const auto& optDetail : options) {

                double T = optDetail.daysTillMaturity / 365.0;
                double T_ = *smallest_expiry;

                BlackScholes option_bought(optDetail.stockPrice, optDetail.strikePrice, optDetail.riskFreeRate, T, optDetail.annualizedVolatility);
                BlackScholes option_at_expiry(stockPrice, optDetail.strikePrice, optDetail.riskFreeRate, T_, optDetail.annualizedVolatility);

                /*
                * Note that calculation of PnL differ for options
                * That are at time of maturity vs with some time left.
                *
                * At maturity:   Intristic Value - Premium Paid
                * Till maturity: Black Scholes Model Pricing - Premium Paid
                */
                // At Maturity Calculation
                if (optDetail.daysTillMaturity / 365 == T_) {
                    if (optDetail.type == 1) {
                        PnL_Position += optDetail.quantity * (callInstristicValue(stockPrice, optDetail.strikePrice) - option_bought.callPrice()) + (optDetail.underlying * (stockPrice - optDetail.stockPrice));
                    }

                    else if (optDetail.type == 0) {
                        PnL_Position += -1 * optDetail.quantity * (putInstristicValue(stockPrice, optDetail.strikePrice) - option_bought.callPrice()) + (optDetail.underlying * (stockPrice - optDetail.stockPrice));
                    }
                }
                // Till Maturity Calculation
                else {
                    if (optDetail.type == 1) {
                        PnL_Position += -1 * optDetail.quantity * (option_bought.callPrice() - option_at_expiry.callPrice());
                    }

                    else if (optDetail.type == 0) {
                        PnL_Position += (-1 * optDetail.quantity * option_bought.putPrice()) - option_at_expiry.putPrice();
                    }
                }
            }
            PnL_At_Expirations_File << stockPrice << " " << PnL_Position << endl;
        }
        PnL_At_Expirations_File.close();

        string PnL_At_Expirations_path = string(currentDir) + "\\PnL_At_Expirations_Data.txt";

        string chart_settings5 =
            "set object 1 rectangle from screen 0,0 to screen 1,1 fillcolor rgb 'black' behind;"
            "set title 'Calendar Spread Payoff Chart At Nearest Expiry';"
            "set title textcolor rgb 'white';"
            "set title font 'Arial,15';"
            "set key title 'Payoffs';"
            "set key textcolor rgb 'white';"
            "set grid lc rgb 'gray';"
            "set xlabel 'Price' textcolor rgb 'white';"
            "set ylabel 'Profit / Loss' textcolor rgb 'white';"
            "set xtics textcolor rgb 'white';"
            "set ytics textcolor rgb 'white';"
            "set xrange[" + to_string(startPrice) + ":" + to_string(endPrice) + "];"
            "set arrow from graph 0, first 0 to graph 1, first 0 nohead lc rgb 'white';"
            "set border lc rgb 'white';";

        string PnL = "'" + PnL_At_Expirations_path + "' using 1:2 with lines title 'At Nearest Expiry'";

        string plot_command_dates1 = chart_settings2 + "plot " + PnL;
        string cmd_command_122 = "cd \"" + gnu_path + "\" && gnuplot -persist -e \"" + plot_command_dates1 + "\"";
        system(cmd_command_122.c_str());
    }

    // ---
    // 6th loop. Goal: Create a 3D graph of Volatility vs Price vs Payoff at Expiration
    // ---

    ofstream Volatility_vs_Price_vs_Payoff_Data_File("Volatility_vs_Price_vs_Payoff_Data.txt");

    for (double volatility = 0.05; volatility < 0.3; volatility += 0.005) {
        for (double stockPrice = startPrice; stockPrice <= endPrice; stockPrice += 1) {

            double Payoff_Given_Volatility_Theory = 0;

            for (const auto& optDetail : options) {
                double T = optDetail.daysTillMaturity / 365.0;
                BlackScholes option(stockPrice, optDetail.strikePrice, optDetail.riskFreeRate, T, volatility);
                BlackScholes option_at_buy(optDetail.stockPrice, optDetail.strikePrice, optDetail.riskFreeRate, T, optDetail.annualizedVolatility);

                if (optDetail.type == 1) {
                    Payoff_Given_Volatility_Theory += optDetail.quantity * (option.callPrice() - option_at_buy.callPrice()) + (optDetail.underlying * stockPrice / 100);
                }
                else if (optDetail.type == 0) {
                    Payoff_Given_Volatility_Theory += optDetail.quantity * (option.putPrice() - option_at_buy.putPrice()) + (optDetail.underlying * stockPrice / 100);
                }
            }
            Volatility_vs_Price_vs_Payoff_Data_File << volatility << " " << stockPrice << " " << Payoff_Given_Volatility_Theory << endl;
        }
        Volatility_vs_Price_vs_Payoff_Data_File << endl;
    }
    Volatility_vs_Price_vs_Payoff_Data_File.close();

    // Gnuplot 3D graph settings
    string Volatility_vs_Price_vs_Payoff_Graph_Path = string(currentDir) + "\\Volatility_vs_Price_vs_Payoff_Data.txt";
    string Volatility_vs_Price_vs_Payoff_Graph_Settings =
        "set object 1 rectangle from screen 0,0 to screen 1,1 fillcolor rgb 'white' behind;"
        "set title '3D View: Volatility vs Price vs Payoff at Expiration';"
        "set title font 'Arial,15';"
        "set title textcolor rgb 'black';"
        "set key title 'Payoffs';"
        "set key textcolor rgb 'black';"
        "set grid lc rgb 'gray';"
        "set xlabel 'Volatility' textcolor rgb 'black';"
        "set ylabel 'Stock Price' textcolor rgb 'black';"
        "set zlabel 'Payoff' textcolor rgb 'black';"
        "set xtics textcolor rgb 'black';"
        "set ytics textcolor rgb 'black';"
        "set ztics textcolor rgb 'black';"
        "set border lc rgb 'black';"
        "set style fill transparent solid 0.5 noborder;"
        "set palette defined (0 'red', 1 'green');"
        "set cbrange;";

    string plot3D_theory = "'" + Volatility_vs_Price_vs_Payoff_Graph_Path + "' using 1:2:3:3 with lines lc palette title 'Theoretical Payoff'";

    string plot_command_thirdloop = Volatility_vs_Price_vs_Payoff_Graph_Settings + "splot " + plot3D_theory;
    string cmd_command_thirdloop = "cd \"" + gnu_path + "\" && gnuplot -persist -e \"" + plot_command_thirdloop + "\"";
    system(cmd_command_thirdloop.c_str());

}
