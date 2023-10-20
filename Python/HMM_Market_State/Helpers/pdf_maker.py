import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF

class PdfReport:
    def __init__(self):
        self.df = pd.read_csv("Data/training_prediction_data.csv")

    def plot_combined_prices(self, filename):
        plt.figure(figsize=(10,6))
        self.df.plot(x='Dates', y='Combined Prices')
        plt.title('Combined Prices Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.tight_layout()
        plt.savefig(filename)

    def plot_training_and_prediction(self, filename):
        fig, ax = plt.subplots(2, figsize=(10,10))
        self.df.plot(x='Dates', y='Prices_training', ax=ax[0], label='Prices Training')
        self.df.plot(x='Dates', y='trained_values', ax=ax[0], secondary_y=True, label='Trained Values')
        self.df.plot(x='Dates', y='Prices_prediction', ax=ax[1], label='Prices Prediction')
        self.df.plot(x='Dates', y='Predicted_values', ax=ax[1], secondary_y=True, label='Predicted Values')
        ax[0].set_title('Training Data')
        ax[1].set_title('Prediction Data')
        plt.tight_layout()
        plt.savefig(filename)

    def generate_pdf(self):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Add title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "Report on Data Analysis", ln=True, align='C')

        # Add combined prices plot
        combined_prices_image = "combined_prices_tmp.png"
        self.plot_combined_prices(combined_prices_image)
        pdf.image(combined_prices_image, x=10, y=100, w=190)
        os.remove(combined_prices_image)

        # # Add training and prediction plots
        # training_image = "training_tmp.png"
        # self.plot_training_and_prediction(training_image)
        # pdf.image(training_image, x=10, y=200, w=190)
        # os.remove(training_image)

        # Description
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, (
            "Hidden Markov Models (HMM) are statistical models that can be used for pattern recognition."
            " An HMM models a system as a series of states, with each state producing observations."
            " Transitions between states are governed by probabilities, and each state has a probability"
            " distribution over the possible observations. HMMs are particularly known for their application"
            " in temporal pattern recognition such as speech, handwriting, and gesture recognition."
        ))

        # Save the pdf
        pdf.output("Helpers/report.pdf")

if __name__ == '__main__':
    # Sample data
    dates = pd.date_range('20230101', periods=10)
    df = pd.DataFrame({
        'Dates': dates,
        'Combined Prices': np.random.rand(10),
        'Prices_training': np.random.rand(10),
        'trained_values': np.random.rand(10),
        'Prices_prediction': np.random.rand(10),
        'Predicted_values': np.random.rand(10)
    })

    report = PdfReport()
    report.generate_pdf()
