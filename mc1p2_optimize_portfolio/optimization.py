"""MC1-P2: Optimize a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
import scipy.optimize as spo

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case
    #allocs = np.asarray([0.2, 0.2, 0.3, 0.3, 0.0]) # add code here to find the allocations
    #cr, adr, sddr, sr = [0.25, 0.001, 0.0005, 2.1] # add code here to compute stats

    # TODO: figure out a better guess
    all_in_one = [1.0 if i is len(syms) - 1 else 0.0 for i in range(0, len(syms))]
    split = 1.0/len(syms)
    even_split = [split for i in range(0, len(syms))]
    #example_1_answer = [  .000000000000000538105153,  .396661695,   .603338305, -.0000000000000000542000166]
    random_split = np.random.dirichlet(np.ones(len(syms)),size=1)[0]
    
    x_guess = np.poly1d(even_split)
    # Put data into a single variale. Index of objects matters! Do not generate plot during minimizing!
    data = np.asarray([sd, ed, syms, False])
    # Create a 0 to 1 bounds for each stock.
    bounds = np.asarray([(0, 1) for i in range(0, len(syms))])
    # Make sure the sum of allocations is very close to 1.
    #constraints = {'type':'eq','fun':sum_to_one}
    # Got this lambda expression from: http://stackoverflow.com/a/18769783/2498729
    constraints = ({'type':'eq', 'fun':lambda x:  1.0 - sum(x)})
    # Find the minimum negative sharpe ratio.
    result = spo.minimize(negative_sharpe_ratio, x_guess, args=(data, ), method='SLSQP', options={'disp':True}, bounds=bounds, constraints=constraints)
    # Make sure this allocation's sharpe ratio is higher than the current best allocation.
    allocs = result.x
    cr, adr, sddr, sr, ev = assess_portfolio(sd=sd, ed=ed, syms=syms, allocs=allocs, gen_plot=gen_plot)

##    # Keep track of best allocations and best sharpe ratio.
##    best_allocs = None
##    best_sr = None
##    all_allocs = []
##
##    # Do random restarts.
##    for r in range(0, 100):
##        print("r = " + str(r))
##        # Randomly generate splits.
##        random_split = np.random.dirichlet(np.ones(len(syms)),size=1)
##        while(contains_list(random_split, all_allocs)):
##            random_split = np.random.dirichlet(np.ones(len(syms)),size=1)
##        all_allocs.append(random_split)
##        print(random_split)
##        x_guess = np.asarray(random_split)
##        # Put data into a single variale. Index of objects matters!
##        data = np.asarray([sd, ed, syms, gen_plot])
##        # Create a 0 to 1 bounds for each stock.
##        bounds = np.asarray([(0, 1) for i in range(0, len(syms))])
##        # Make sure the sum of allocations is very close to 1.
##        constraints = {'type':'eq','fun':sum_to_one}
##        # Find the minimum negative sharpe ratio.
##        result = spo.minimize(negative_sharpe_ratio, x_guess, args=(data, ), method='SLSQP', options={'disp':True}, bounds=bounds, constraints=constraints)
##        # Make sure this allocation's sharpe ratio is higher than the current best allocation.
##        allocs = result.x
##        cr, adr, sddr, sr, ev = assess_portfolio(sd=sd, ed=ed, syms=syms, allocs=allocs, gen_plot=gen_plot)
##        if best_sr is None or best_sr < sr:
##            best_allocs = allocs
##            best_sr = sr
##
##    # Get best allocation's stats.
##    cr, adr, sddr, sr, ev = assess_portfolio(sd=sd, ed=ed, syms=syms, allocs=best_allocs, gen_plot=gen_plot)

    # Get daily portfolio value
    port_val = prices_SPY # add code here to compute daily portfolio values

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        pass

    return allocs, cr, adr, sddr, sr

def contains_list(small_list, big_list):
    sl = np.ndarray.tolist(small_list)
    for blnp in big_list:
        bl = np.ndarray.tolist(blnp)
        if len(sl) == len(bl):
            found = True
            for i in range(0, len(sl)):
                if sl[i] != bl[i]:
                    found = False
                    break
            if found:
                return True
        else:
            # This shouldn't happen, but just in case.
            print("Error. Uneven list sizes. Small list = " + str(small_list) + ". Sub item of big list = " + str(bl))

    # Nothing found.
    return False

def sum_to_one(allocations):
    # TODO what about constraining to exactly 1.0? or between 0 and 1?
    error = 0.02
    threshold = 1.0
    total = float(allocations.sum())
    in_bounds = total <= threshold + error and total >= threshold - error
    #in_bounds = total <= threshold + error
    #print(allocations)
    #print(total)
    #print(type(total))
    #print(in_bounds)
    return 0 if in_bounds else 1

def negative_sharpe_ratio(allocations, data):
    #print("allocations: " + str(allocations))
    cr, adr, sddr, sr, ev = assess_portfolio(sd=data[0], ed=data[1], syms=data[2], allocs=allocations, gen_plot=data[3])
    return -sr

def assess_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['GOOG','AAPL','GLD','XOM'], \
    allocs=[0.1,0.2,0.3,0.4], \
    sv=1000000, rfr=0.0, sf=252.0, \
    gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get daily portfolio value
    #port_val = prices_SPY # add code here to compute daily portfolio values
    # Normalize prices. This is also the cumulative return starting from start date.
    normed = prices/prices.ix[0,:]
    # Allocate normed values correctly.
    alloced = normed * allocs
    # Apply our starting value on allocated values.
    pos_vals = alloced * sv
    # Get total value of portfolio by summing accross each day.
    port_val = pos_vals.sum(axis=1)
    # Calculate daily return. TODO Should we remove the 0th value?
    daily_ret = port_val.copy()
    daily_ret[1:] = (daily_ret[1:]/daily_ret[:-1].values) - 1
    daily_ret.ix[0] = 0

    # Get portfolio statistics (note: std_daily_ret = volatility)
    #cr, adr, sddr, sr = [0.25, 0.001, 0.0005, 2.1] # add code here to compute stats
    cr = (port_val[-1]/port_val[0]) - 1
    adr = (daily_ret[1:]).mean()
    sddr = daily_ret[1:].std()
    sr = np.sqrt(sf) * (adr - rfr) / sddr

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        #df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        normed_SPY = prices_SPY/prices_SPY.ix[0,:]
        df_temp = pd.concat([alloced.sum(axis=1), normed_SPY], keys=['Portfolio', 'SPY'], axis=1)
        df_temp.plot()
        #normed_SPY.plot()
        #normed.plot()
        plt.title('Daily Portfolio Value & SPY', fontsize=20)
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Price', fontsize=16)
        plt.show()
        pass

    # Add code here to properly compute end value
    #ev = sv
    ev = (1 + cr) * sv
    
    return cr, adr, sddr, sr, ev

def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2009,1,1)
    end_date = dt.datetime(2010,1,1)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM', 'IBM']

    #Start Date: 2010-01-01
    #End Date: 2010-12-31
    #Symbols: ['GOOG', 'AAPL', 'GLD', 'XOM']
    #Optimal allocations: [  5.38105153e-16   3.96661695e-01   6.03338305e-01  -5.42000166e-17]
    #Sharpe Ratio: 2.00401501102
    #Volatility (stdev of daily returns): 0.0101163831312
    #Average Daily Return: 0.00127710312803
    #Cumulative Return: 0.360090826885
    start_date = dt.datetime(2010,1,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']

    #Start Date: 2004-01-01
    #End Date: 2006-01-01
    #Symbols: ['AXP', 'HPQ', 'IBM', 'HNZ']
    #Optimal allocations: [  7.75113042e-01   2.24886958e-01  -1.18394877e-16  -7.75204553e-17]
    #Sharpe Ratio: 0.842697383626
    #Volatility (stdev of daily returns): 0.0093236393828
    #Average Daily Return: 0.000494944887734
    #Cumulative Return: 0.255021425162
##    start_date = dt.datetime(2004,1,1)
##    end_date = dt.datetime(2006,1,1)
##    symbols = ['AXP', 'HPQ', 'IBM', 'HNZ']

    #A plot comparing the optimal portfolio with SPY as comparison_optimal.png using the following parameters:
    #Start Date: 2008-06-01, End Date: 2009-06-01, Symbols: ['IBM', 'X', 'GLD']
    start_date = dt.datetime(2008,6,1)
    end_date = dt.datetime(2009,6,1)
    symbols = ['IBM', 'X', 'GLD']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
