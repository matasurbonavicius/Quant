#pragma once

// Define an Option struct to encapsulate details about an option.
struct OptionDetails {
    int type;                      // 1 for call, 0 for put
    int quantity;                  // negative for selling
    double impliedPremium;
    double dividend;
    double underlying;             // Market mid price
    double stockPrice;
    double strikePrice;
    double riskFreeRate;
    double daysTillMaturity;
    double theoreticalPremium;     // Model calculated price
    double annualizedVolatility;
};

double callInstristicValue(double stockPrice, double strikePrice);
double putInstristicValue(double stockPrice, double strikePrice);
