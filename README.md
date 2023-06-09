# LOB-analysis

Knowing the effects of an order is crucial for better trading and asset management to reduce the difference in between
the decision price and the actual price. With accurate analysis of the effects, traders and asset managers can time the
orders so, that the difference in prices stays minimal.

To solve this issue, the project that you are viewing was built. The project uses data on stock order flow from NYSE
sourced by Tampere University at the courtesy of Juho Kanniainen. I use two different models in the project: first, the
Autoformer to consume all available information from the time series on order book states. This includes the midpirce as
well as the "supply and demand"- average value of limit orders in both bid and ask sides respectively. The intent is
two- fold: Firstly to check whether the limit order books' structure affects the impact the order has. Secondly, given
the past time series, we use the Autoformer model to generate a forecast that does not include any information of the
inbound order. This will be used as a baseline forecast to evaluate the trades effects by comparing the results from the
second model to the previous forecast. Essentially, the forecast is adjusted to include the effects of the order.

The second model is an adaptation of the Transformer architecture to evaluate the orders effects. It uses encoder-
decoder architecture, taking in both the baseline forecast and the inbound order. The baseline is passed throught the
decoder layer, using the same architecture as the Autoformer uses as it is well suited for time series analyses. The
encoder however is an adaptation of the classical Transformer encoder, and aims to map the order into the dimensions
specified in the model. The aim is to "transalte" the order into higher dimensional space, after which passing it to the
encoder- decoder attention would help in making the correct adjustments to the forecast.
