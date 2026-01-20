
Week plan batteries sprint 1 - NL Data.csv: this is the file with hourly rates for all nations
Week plan batteries sprint 1 - model_input.csv: this has the modelling inputs for each nation 
Week plan batteries sprint 1 - use_profiles.csv: this has, for each nation, how much elec is used for each hour in the day. 

If no user profile is available for your country, use the UK one, and make cler in output

Use the latest full year for all data calculations


You will be used to estimate the costs and revenue, optimising the trading across all the different ways of making money. If cells in model_input.csv are blank for a nation, assume the revenue available from that stream is zero.
You should output the income, for a full optimal system, which can be made while servicing each customer (and the income from each customer).

Then costs.



### to improve
might be able to artificially limit the power on large battery




### how algo works 20th jan 2026
we assume CM stress events are rare enough that the battery is usually available (current approach - optimistic)   

it solves 365 separate MILP problems (one per day), each with 24 hourly decision variables, then aggregates the results. Each day gets its own optimal strategy based on that day's actual price curve.
