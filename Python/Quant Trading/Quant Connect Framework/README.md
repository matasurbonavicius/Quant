Quant Connect Algorithm Framework

Code developed by Matas Urbonavicius

The idea behind all this is to create a uniform quant trading algorithm framework that will:

Be highly modular; highly interchangeable
Have safety features, separation of concerns
Be a great foundation on which to develop new strategies
Be as simple as it can, given its task
The framework is described in my blog: https://www.mercatuscapitis.com/2023/09/black-box-framework.html

The general thought is that we want an algorithm that is controled and monitored by independent modules: in this case- the Risk Model and the Portfolio Model; Main

How is this achieved and why is it not linear? A perfect example:

Universe Selection Model -> Alpha Model -> Portfolio Model -> -> Risk Model -> Costs Model -> Execution Model -> Trade

However, Quant Connect does not allow the Alpha Model information to travel more than the porftolio model allows. That is not ideal as we might have more than one alpha model in our algorithm. Why?

Cost reasons: having 5 separate algorithms will cost us 5x in server costs
Simplicity: if our algoright works on 5 different timeframes, it is easier to split all timeframes into separate alphas, than to combine everything into one file; alpha model. Code loses readability.
Hence this model seeks to be able to support as many alpha models as possible. That imposes a challenge in Quant Connect platform.

How do we mark the order to belong to a specific alpha model? We tag it. We can only tag it when executing. And if portfolio model cannot pass the tag further, that means all our signals from alpha models should arrive at execution model where portfolio and risk models will take the signals and create the quantities needed.

Therefore the main idea is to have quantities calculated and confirmed by 3 algorithm.

Position Sizing algorithm - Determines the base quantity
Portfolio model - Adjust quantity if needed according to portfolio
Risk model - Either allows the trade to go through or not. If not, possibly adjusts the quantity
When risk (in quantity) is determined, we can use that to make orders

Risk model also determines the SL and TP, based on instructions from alpha models
