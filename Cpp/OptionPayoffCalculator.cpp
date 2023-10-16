#include <iostream>
#include <limits>
#include <cmath>
#include <fstream>
#include <windows.h>
#include <algorithm> // Required for std::max
#include <vector>



// Define an Option struct to encapsulate details about an option.
struct Option {
    char type;           // 'C' for call, 'P' for put
    double strikePrice;
    double premium;
    int quantity;
};

// Function to calculate the payoff for a call option.
double callInstristicValue(double stockPrice, double strikePrice) {
    return std::max<double>(0.0, stockPrice - strikePrice);
}

// Function to Calculate the payoff for a put option.
double putInstristicValue(double stockPrice, double strikePrice) {
    return std::max<double>(0.0, strikePrice - stockPrice);
}

// Gather option details from the user.
std::vector<Option> getInputFromUser() {
    int numberOfOptions;
    std::vector<Option> options;

    std::cout << "How many different options do you have? ";
    std::cin >> numberOfOptions;
    options.resize(numberOfOptions);

    for (int i = 0; i < numberOfOptions; i++) {
        std::cout << "\nOption " << i + 1 << ": (C for Call, P for Put): ";
        std::cin >> options[i].type;
        std::cout << "Enter the strike price: ";
        std::cin >> options[i].strikePrice;
        std::cout << "Enter the premium: ";
        std::cin >> options[i].premium;
        std::cout << "Enter the quantity (positive for long, negative for short): ";
        std::cin >> options[i].quantity;
    }

    return options;
}

int main() {
    // Get the options details from the user.
    std::vector<Option> options = getInputFromUser();

    std::ofstream dataFile("data.txt");

    // Find the minimum and maximum strike prices.
    double minStrikePrice = 1e30;   // Some small value
    double maxStrikePrice = -1e30;  // Some large value

    for (const Option& opt : options) {
        if (opt.strikePrice < minStrikePrice) {
            minStrikePrice = opt.strikePrice;
        }
        if (opt.strikePrice > maxStrikePrice) {
            maxStrikePrice = opt.strikePrice;
        }
    }

    double startPrice = minStrikePrice * 0.8; // 20% lower
    double endPrice = maxStrikePrice * 1.2;   // 20% higher

    // Chart range is +-20% of the min/max strike valus
    for (double stockPrice = startPrice; stockPrice <= endPrice; stockPrice += 0.5) {
        double totalPayoff = 0;

        for (const Option& opt : options) {
            if (opt.type == 'C' || opt.type == 'c') {
                totalPayoff += opt.quantity * (callInstristicValue(stockPrice, opt.strikePrice) - opt.premium);
            }
            else if (opt.type == 'P' || opt.type == 'p') {
                totalPayoff += opt.quantity * (putInstristicValue(stockPrice, opt.strikePrice) - opt.premium);
            }
        }

        dataFile << stockPrice << " " << totalPayoff << std::endl;
    }

    dataFile.close();

    char currentDir[MAX_PATH];
    GetCurrentDirectoryA(MAX_PATH, currentDir);

    // Plot the graph using gnuplot.
    std::string command = "cd \"C:\\Program Files\\gnuplot\\bin\" && gnuplot -persist -e \"set arrow from graph 0,first 0 to graph 1,first 0 nohead; plot '" + std::string(currentDir) + "\\data.txt' with lines\"";
    system(command.c_str());

    return 0;
}




