#pragma once
#include <vector>
#include "Definitions.h"
#include <iostream>

using namespace std;

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

    cout << "|--------------------------------------------------------------------|" << endl;
    cout << "|                        -- Global Inputs --                         |" << endl;
    cout << "| How many different options do you have? ";
    cin >> numberOfOptions;
    options.resize(numberOfOptions);

    // Collecting some data outside the loop to not
    // ask the same thing over and over again
    double commonRiskFreeRate;
    cout << "| What is the risk free rate? (if 1%, then enter 0.01) - ";
    cin >> commonRiskFreeRate;

    double currentStockPrice;
    cout << "| What is the current underlying price? - ";
    cin >> currentStockPrice;

    double underlying;
    cout << "| How much of underlying do you own? - ";
    cin >> underlying;

    // Loops through quantity of different options specified
    for (int i = 0; i < numberOfOptions; i++) {
        cout << "\n| Details for Option " << i + 1 << ":" << endl;
        cout << "| What is the option? (1 for call & 0 for put) - ";
        cin >> options[i].type;
        cout << "| What is the quantity? (Negative for sold) - ";
        cin >> options[i].quantity;
        cout << "| What is the current price for the option? - ";
        cin >> options[i].impliedPremium;
        cout << "| What is the strike price? - ";
        cin >> options[i].strikePrice;
        cout << "| How many days till maturity? - ";
        cin >> options[i].daysTillMaturity;
        cout << "| What is the annualized volatility? (if 10%, then enter 0.1) - ";
        cin >> options[i].annualizedVolatility;
        cout << "|--------------------------------------------------------------------|" << endl;

        // Adds the outside-loop inputs for each option
        options[i].riskFreeRate = commonRiskFreeRate;
        options[i].stockPrice = currentStockPrice;
        options[i].underlying = underlying;
    }

    return options;
}

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

// --- Simple one option positions. Can combine with underlying to produce
// --- protected positions

// Use when having an extreme bullish view
vector<OptionDetails> LongCallOption() {
    vector<OptionDetails> options(1);

    // Buy one call
    options[0].type = 1;
    options[0].quantity = 1;
    options[0].impliedPremium = 2.0;
    options[0].strikePrice = 105.0;
    options[0].daysTillMaturity = 20;
    options[0].annualizedVolatility = 0.2;
    options[0].riskFreeRate = 0.05;
    options[0].stockPrice = 100;

    return options;
}

// Use when having an extreme bearish view
vector<OptionDetails> LongPutOption() {
    vector<OptionDetails> options(1);

    // Buy one put
    options[0].type = 0;
    options[0].quantity = 1;
    options[0].impliedPremium = 2.0;
    options[0].strikePrice = 95.0;
    options[0].daysTillMaturity = 20;
    options[0].annualizedVolatility = 0.2;
    options[0].riskFreeRate = 0.05;
    options[0].stockPrice = 100;

    return options;
}

// Use when having a bearish view
vector<OptionDetails> ShortCallOption() {
    vector<OptionDetails> options(1);

    // short one call
    options[0].type = 1;
    options[0].quantity = -1;
    options[0].impliedPremium = 2.0;
    options[0].strikePrice = 105.0;
    options[0].daysTillMaturity = 20;
    options[0].annualizedVolatility = 0.2;
    options[0].riskFreeRate = 0.05;
    options[0].stockPrice = 100;

    return options;
}

// Use when having a bullish view
vector<OptionDetails> ShortPutOption() {
    vector<OptionDetails> options(1);

    // Short one put
    options[0].type = 0;
    options[0].quantity = -1;
    options[0].impliedPremium = 2.0;
    options[0].strikePrice = 95.0;
    options[0].daysTillMaturity = 20;
    options[0].annualizedVolatility = 0.2;
    options[0].riskFreeRate = 0.05;
    options[0].stockPrice = 100;

    return options;
}

// --- Volatility Spreads

// Backspreads are delta neutral positions which consists of more long options than short options
// where options expire at the same time.
// To achieve it, options with smaller deltas must be purchased and options with bigger
// deltas must be sold

vector<OptionDetails> CallBackspread() {
    vector<OptionDetails> options(2);

    // Sell one call
    options[0].type = 1;
    options[0].quantity = -1;
    options[0].impliedPremium = 1.5;
    options[0].strikePrice = 100;
    options[0].daysTillMaturity = 30;
    options[0].annualizedVolatility = 0.2;
    options[0].riskFreeRate = 0.05;
    options[0].stockPrice = 100;

    // Buy two higher strike calls
    options[1].type = 1;
    options[1].quantity = 2;
    options[1].impliedPremium = 1.0;
    options[1].strikePrice = 105;
    options[1].daysTillMaturity = 30;
    options[1].annualizedVolatility = 0.2;
    options[1].riskFreeRate = 0.05;
    options[1].stockPrice = 100;

    return options;
}

vector<OptionDetails> PutBackspread() {
    vector<OptionDetails> options(2);

    // Sell one put
    options[0].type = 0;
    options[0].quantity = -1;
    options[0].impliedPremium = 1.5;
    options[0].strikePrice = 100;
    options[0].daysTillMaturity = 30;
    options[0].annualizedVolatility = 0.2;
    options[0].riskFreeRate = 0.05;
    options[0].stockPrice = 100;

    // Buy two lower strike puts
    options[1].type = 0;
    options[1].quantity = 2;
    options[1].impliedPremium = 1.0;
    options[1].strikePrice = 95;
    options[1].daysTillMaturity = 30;
    options[1].annualizedVolatility = 0.2;
    options[1].riskFreeRate = 0.05;
    options[1].stockPrice = 100;

    return options;
}

vector<OptionDetails> RatioVerticalSpread() {
    vector<OptionDetails> options(2);

    // Buy one call
    options[0].type = 1;
    options[0].quantity = 1;
    options[0].impliedPremium = 1.5;
    options[0].strikePrice = 100;
    options[0].daysTillMaturity = 30;
    options[0].annualizedVolatility = 0.2;
    options[0].riskFreeRate = 0.05;
    options[0].stockPrice = 100;

    // Sell two higher strike calls
    options[1].type = 1;
    options[1].quantity = -2;
    options[1].impliedPremium = 1.0;
    options[1].strikePrice = 105;
    options[1].daysTillMaturity = 30;
    options[1].annualizedVolatility = 0.2;
    options[1].riskFreeRate = 0.05;
    options[1].stockPrice = 100;

    return options;
}

vector<OptionDetails> Straddle() {
    vector<OptionDetails> options(2);

    // Buy one call
    options[0].type = 1;
    options[0].quantity = 1;
    options[0].impliedPremium = 1.5;
    options[0].strikePrice = 100;
    options[0].daysTillMaturity = 30;
    options[0].annualizedVolatility = 0.2;
    options[0].riskFreeRate = 0.05;
    options[0].stockPrice = 100;

    // Buy one put at same strike
    options[1].type = 0;
    options[1].quantity = 1;
    options[1].impliedPremium = 1.5;
    options[1].strikePrice = 100;
    options[1].daysTillMaturity = 30;
    options[1].annualizedVolatility = 0.2;
    options[1].riskFreeRate = 0.05;
    options[1].stockPrice = 100;

    return options;
}

vector<OptionDetails> Strangle() {
    vector<OptionDetails> options(2);

    // Buy one call
    options[0].type = 1;
    options[0].quantity = 1;
    options[0].impliedPremium = 1.0;
    options[0].strikePrice = 105;
    options[0].daysTillMaturity = 30;
    options[0].annualizedVolatility = 0.2;
    options[0].riskFreeRate = 0.05;
    options[0].stockPrice = 100;

    // Buy one put
    options[1].type = 0;
    options[1].quantity = 1;
    options[1].impliedPremium = 1.0;
    options[1].strikePrice = 95;
    options[1].daysTillMaturity = 30;
    options[1].annualizedVolatility = 0.2;
    options[1].riskFreeRate = 0.05;
    options[1].stockPrice = 100;

    return options;
}

vector<OptionDetails> CallTimeSpread() {
    vector<OptionDetails> options(2);

    // Buy one call with longer maturity
    options[0].type = 1;
    options[0].quantity = 1;
    options[0].impliedPremium = 4.75;
    options[0].strikePrice = 100;
    options[0].daysTillMaturity = 56;
    options[0].annualizedVolatility = 0.3;
    options[0].riskFreeRate = 0.01;
    options[0].stockPrice = 100;

    // Sell one call with shorter maturity at same strike
    options[1].type = 1;
    options[1].quantity = -1;
    options[1].impliedPremium = 3.35;
    options[1].strikePrice = 100;
    options[1].daysTillMaturity = 28;
    options[1].annualizedVolatility = 0.3;
    options[1].riskFreeRate = 0.01;
    options[1].stockPrice = 100;

    return options;
}
