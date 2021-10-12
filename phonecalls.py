#!/usr/bin/env python3

import numpy as np
import pandas as pd
import networkx as nx
import datetime as dt
import scipy.stats as stats
import copy
import os
import math

def allcalls(infile, filt, ego, alter, timestamp, tstampformat, header=True, min_activity=1):
    '''
    This method takes a file (argument infile), usually in .csv format,
    and process it only to obtain a dataframe with columns [ego, alter,
    timestamp, universal clock, alter clock]. In order to produce that,
    it uses the following arguments
    filt            : in case you need to filter phone calls (incoming, etc.)
                    it can be an empty tuple, in case there are no filters. The first element of the
                    tuple is the label/number of the column to filter.
    ego             : which column (label or number) contains the id for ego
    alter           : same as above, for alter's identifier
    timestamp       : a list with the label(s) for the timestamp
    tsstampformat   : Python's format specification to parse dates. Look at the
                    documentation for the "datetime" module for further reference
                    https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
    header          : does the original file contains headers? It is a boolean value
                    that defualts to "True"
    min_activity    : the minimum number of phone calls between ego and alter to keep
                    the phone calls for this pair in the resulting dataframe. It
                    defaults to 1, or keep all phone calls.
    '''
    if header:
        df = pd.read_csv(infile)
        tmp = pd.DataFrame()
        if len(filt) > 2:
            for i in range(1, len(filt)):
                df2 = df.loc[df[filt[0]] == filt[i]]
                tmp = tmp.append(df2)
        elif len(filt) == 2:
            df2 = df.loc[df[filt[0]] == filt[1]]
        else:
            df2 = copy.deepcopy(df)
    else:
        df = pd.read_csv(infile, header=None)
        tmp = pd.DataFrame()
        if len(filt) > 2:
            for i in range(1, len(filt)):
                df2 = df.loc[df[filt[0]] == filt[i]]
                tmp = tmp.append(df2)
        elif len(filt) == 2:
            df2 = df.loc[df[filt[0]] == filt[1]]
        else:
            copy.deepcopy(df)

    egocol = df2[ego]
    altercol = df2[alter]
    newindex = list(df2.index)
    newdf = pd.DataFrame({'ego': egocol, 'alter': altercol}, index=newindex)
    timecol = df2[timestamp]
    if len(timecol.columns) > 1:
        for i in list(timecol.columns):
            if i == list(timecol.columns)[0]:
                newdf['time'] = timecol[i] + " "
            elif i != list(timecol.columns)[-1]:
                newdf['time'] += timecol[i] + " "
            else:
                newdf['time'] += timecol[i]
    else:
        newdf['time'] = timecol

    newdf['time'] = pd.to_datetime(newdf['time'], format=tstampformat)
    newdf.sort_values(by=['alter', 'ego', 'time'], ascending=[True, True, True], inplace=True)
    newdf['no'] = 1
    newdf['no'] = newdf.groupby(['alter', 'ego'])[['no']].cumsum()
    newdf['no'] -= 1

    tmp = newdf.groupby(['ego', 'alter'])[['no']].max().reset_index()
    tmp.columns = ['ego', 'alter', 'ma']
    tmp['ma'] += 1
    newdf = newdf.merge(tmp, on=['ego', 'alter'])
    newdf = newdf.loc[newdf['ma'] >= min_activity]

    mindate = min(newdf['time'])
    newdf['uclock'] = (newdf.loc[:, 'time'] - mindate).dt.days
    newdf.loc[:, 'firstcall'] = False
    newdf.loc[newdf['no'] == 0, 'firstcall'] = True
    tmp = newdf.loc[newdf['firstcall'], ['ego', 'alter', 'uclock']]
    tmp.columns = ['ego', 'alter', 'fcall']
    newdf = newdf.merge(tmp, left_on=['ego', 'alter'], right_on=['ego', 'alter'])
    newdf['aclock'] = newdf['uclock'] - newdf['fcall']
    newdf.drop(columns=['fcall', 'firstcall', 'no', 'ma'], inplace=True)
    newdf = newdf.astype({'uclock': int, 'aclock': int})
    newdf.reset_index(drop=True, inplace=True)

    return newdf

def limit_calls(unf_calls, T):
    '''
    This method takes a calls dataframe produced with the "allcalls" method
    and returns a new dataframe with some calls removed. The arguments are:
    
    unf_calls          : unfiltered calls dataframe produced with the "allcalls"
                       method
    T                  : all lifetime values will be truncated to this parameter.
    
    Note that all alters with lifetime > T will be set to ell = T
    '''
    maxT = max(unf_calls['uclock'])
    df = unf_calls.loc[unf_calls['aclock'] <= T].copy()
    df['ea'] = list(zip(df['ego'], df['alter']))
    #maxT = max(df['uclock'])
    mint = df.groupby('ea')[['uclock']].min()
    mint = mint.loc[mint['uclock'] < (maxT - T)]
    tokeep = list(mint.index)
    df2 = df[df['ea'].isin(tokeep)]
    if 'rm' in df2.columns:
        df3 = df2.drop(columns = ['aclock', 'rm', 'ea'])
    else:
        df3 = df2.drop(columns = ['aclock', 'ea'])
    df3['aclock'] = df2.index.map(lambda i: df2.at[i, 'uclock'] - mint['uclock'][(df2.at[i, 'ego'], df2.at[i, 'alter'])])
    df4 = df3.sort_values(by=['ego', 'alter', 'time']).reset_index(drop=True)
    return df4

def pairs(df):
    '''
    This method creates a dataframe in which every row is an ego-alter pair. It
    contains the number of calls between ego and alter "nij"; the total
    number of calls made by ego "n"; and the amount of contacts in ego's
    network "k"

    The arguments for the function are:
    df              : a phone call dataframe produced with the "allcalls" method
    '''
    df1 = df.groupby(['ego', 'alter']).count().reset_index()[['ego', 'alter', 'time']]
    df1.columns = ['ego', 'alter', 'nij']
    df2 = df1.groupby('ego')[['alter']].count().reset_index()
    df2.columns = ['ego', 'k']
    df1 = df1.merge(df2, on='ego')
    df2 = df1.groupby('ego')[['nij']].sum().reset_index()
    df2.columns = ['ego', 'n']
    df1 = df1.merge(df2, on='ego')
    df1.sort_values(by='n', ascending=False, inplace=True)
    df1.reset_index(drop=True, inplace=True)

    return df1

def lives_dictionary(callsdf):
    '''
    Use this method to create a "lives dictionary", in which the first two keys are ego and
    alter identifiers. The third key can be:

    - "t0" to get the day (universal clock) of the first appeareance of alter in ego's network
    - "tf" for the final appeareance
    - "ell" for the lifetime of alter in ego's network (in days)
    - "nij" to get the activity (number of calls) that ego makes to alter
    '''
    a = callsdf.groupby(['ego', 'alter'])[['uclock']].min().rename(columns={'uclock': 't0'})
    b = callsdf.groupby(['ego', 'alter'])[['uclock']].max().rename(columns={'uclock': 'tf'})
    c = callsdf.groupby(['ego', 'alter'])[['uclock']].count().rename(columns={'uclock': 'nij'})
    d = a.merge(b, left_index=True, right_index=True)
    d = d.merge(c, left_index=True, right_index=True)
    d['ell'] = d['tf'] - d['t0']
    result = {}
    for ego in callsdf['ego'].unique():
        df = d.loc[ego]
        result[ego] = df.to_dict('index')
    return result

def remove_alters(callsdf, lives, num_days):
    '''
    If, for some reason, we have to remove alters from the data, this method allows to remove all
    alters (and phone calls) that appeared before a certain day in the universal clock. It takes
    the following arguments

    callsdf         : a dataframe produces with the "allcalls" method
    lives           : a "lives dictionary" produced with the method of the same name
    num_days        : alters appeared before this day will be removed

    For all analysis purposes, it is recommended to use the methods "lives_dictionary" and
    "pairs" with the output of this method.
    '''
    rmalter = {}
    for ego in lives.keys():
        rmalter[ego] = []
        for alter in lives[ego].keys():
            if lives[ego][alter]['t0'] < num_days:
                rmalter[ego].append(alter)

    tmp = callsdf.loc[callsdf['uclock'] >= num_days]
    tmp['uclock'] -= num_days
    tmp.to_csv("tmp.csv")
    newdf = allcalls("tmp.csv", (), 'ego', 'alter', ['time'], '%Y-%m-%d %H:%M:%S')
    os.remove("tmp.csv")

    for i in newdf.index:
        ego = newdf.at[i, 'ego']
        alter = newdf.at[i, 'alter']
        if alter in rmalter[ego]:
            newdf.at[i, 'rm'] = 1
        else:
            newdf.at[i, 'rm'] = 0

    newdf = newdf.loc[newdf['rm'] == 0]
    newdf.drop('rm', axis=1, inplace=True)

    return newdf


def get_f(callsdf, theego, bina, binell, external_lives=False):
    '''
    This method outputs a dataframe with one row per (a, ell) combination, and the number
    of phone calls ego made to alters with that combination of parameters. The arguments are

    callsdf             : a dataframe produces with the "allcalls" or "remove_alters" methods
    theego              : specify an ego for results only using it. If the 'all' argument is
                        passed, it will calculate a dataframe per ego
    lives_dict          : dictionary produces with the "lives_dictionary" method
    bina                : the value for \Delta a
    binell              : \Delta ell
    '''
    callsdf = callsdf.sort_values(by='time')
    if theego != 'all':
        df1 = callsdf.loc[callsdf['ego'] == theego]
    else:
        df1 = callsdf.copy()

    f = {}
    for ego in df1['ego'].unique():
        f[ego] = {}
        df2 = df1.loc[df1['ego'] == ego]
        for alter in df2['alter'].unique():
            df3 = df2.loc[df2['alter'] == alter]
            df3 = df3.sort_values(by='time')
            if not external_lives:
                lamb = (max(df3['uclock']) - min(df3['uclock'])) // binell
            else:
                lamb = external_lives[ego][alter]['ell'] // binell
            df3['alpha'] = df3['aclock'] // bina
            tmp = df3.groupby('alpha').size()
            f[ego][alter] = pd.DataFrame({'lambda': lamb, 'alpha': tmp.index, 'f': tmp})
            f[ego][alter].reset_index(drop=True, inplace=True)
    return f

def plot_f_all(fresult, lives_dict, xaxis, binell):
    '''
    This method creates a dataframe to plot the results obtained with the get_f method. It takes as
    arguments

    fresult             : dataframe created using the get_f method
    lives_dict          : dictionary obtained with ther method of the same name
    xaxis               : select "alpha" or "lambda" to choose which quantity to put on the
                        horizontal axis of the plot.
    binell              : a value for Delta ell

    The resulting dataframe contains the values chosen for the horizontal axis as indices; and the values
    for the horizontal axis as the column "h". Note that it produces the sum of all calls for all egos to
    alters with particular values of lambda and alpha
    '''
    result = {}
    for ego in fresult.keys():
        for alter in fresult[ego].keys():
            lamb = lives_dict[ego][alter]['ell'] // binell
            for i in fresult[ego][alter].index:
                alpha = fresult[ego][alter].at[i, 'alpha']
                if xaxis == 'alpha':
                    result[lamb] = result.get(lamb, {})
                    result[lamb][alpha] = result[lamb].get(alpha, 0) + fresult[ego][alter].at[i, 'f']
                elif xaxis == 'lambda':
                    result[alpha] = result.get(alpha, {})
                    result[alpha][lamb] = result[alpha].get(lamb, 0) + fresult[ego][alter].at[i, 'f']

    for a in result.keys():
        result[a] = pd.DataFrame.from_dict(result[a], orient='index', columns=['h'])
        result[a].sort_index(inplace = True)

    return result

def get_g(fresult, xaxis='alpha'):
    '''
    With this method, the number of phone calls PER ALTER can be obtained. It uses the sum of f_i(lambda, alpha)
    and divides by H(alpha). It only takes two arguments
    fresult             : dataframe produced with the get_f method
    xaxis               : this is for future plotting purposes. Which quantity should be
                        in the horizontal axis? Defaults to "alpha"

    The output is a dictionary with a dataframe per ego.
    '''
    g = {}
    r = {}
    for ego in fresult.keys():
        g[ego] = {}
        alta = {}
        for alter in fresult[ego].keys():
            alla = list(fresult[ego][alter]['alpha'])
            for a in alla:
                alta[a] = alta.get(a, 0) + 1
        for alter in fresult[ego].keys():
            for i in fresult[ego][alter].index:
                a = fresult[ego][alter].at[i, 'alpha']
                l = fresult[ego][alter].at[i, 'lambda']
                f = fresult[ego][alter].at[i, 'f']
                if xaxis == 'lambda':
                    g[ego][a] = g[ego].get(a, {})
                    g[ego][a][l] = g[ego][a].get(l, 0) + f
                elif xaxis == 'alpha':
                    g[ego][l] = g[ego].get(l, {})
                    g[ego][l][a] = g[ego][l].get(a, 0) + f

        idx = 0
        df = pd.DataFrame()
        for k in g[ego].keys():
            for kk in g[ego][k].keys():
                if xaxis == 'lambda':
                    df.at[idx, 'lambda'] = kk
                    df.at[idx, 'alpha'] = k
                    df.at[idx, 'g'] = g[ego][k][kk] / alta[k]
                    idx += 1
                elif xaxis == 'alpha':
                    df.at[idx, 'lambda'] = k
                    df.at[idx, 'alpha'] = kk
                    df.at[idx, 'g'] = g[ego][k][kk] / alta[kk]
                    idx += 1
        if xaxis == 'lambda':
            df.sort_values(by=['lambda', 'alpha'], inplace=True)
        elif xaxis == 'alpha':
            df.sort_values(by=['alpha', 'lambda'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        r[ego] = df
    return r


def plot_g(gresult, xaxis):
    '''
    This method is used to produce one dataframe per ego, where the results from
    the get_g method can be easily plotted. The argumentsd are:
    gresult             : output of the get_g method
    xaxis               : which quantity (alpha or lambda) should be plotted in
                        the horizontal axis
    '''
    result = {}
    for ego in gresult.keys():
        for i in gresult[ego].index:
            l = gresult[ego].at[i, 'lambda']
            a = gresult[ego].at[i, 'alpha']
            g = gresult[ego].at[i, 'g']
            if xaxis == 'lambda':
                result[a] = result.get(a, {})
                result[a][l] = result[a].get(l, [])
                result[a][l].append(g)
            elif xaxis == 'alpha':
                result[l] = result.get(l, {})
                result[l][a] = result[l].get(a, [])
                result[l][a].append(g)
                
    r2 = {}
    for k in sorted(list(result.keys())):
        r2[k] = pd.DataFrame()
        for kk in result[k].keys():
            if xaxis == 'lambda':
                r2[k].at[kk, 'lambda'] = np.mean(result[k][kk])
            elif xaxis == 'alpha':
                r2[k].at[kk, 'alpha'] = np.mean(result[k][kk])
        r2[k].sort_index(inplace=True)

    return r2

def get_b(fresult, xaxis='alpha'):
    '''
    This is an alternative measurement to g. Instead of dividing the sum of f_i(lambda, alpha) by
    H(alpha), it does it by H(lambda). The arguments are:
    fresult             : dataframe produced with the get_f method
    xaxis               : this is for future plotting purposes. Which quantity should be
                        in the horizontal axis? Defaults to "alpha"

    The output is a dictionary with a dataframe per ego. Note that this is a variant of the g quantity,
    and therefore, the same method used to plot g can be applied here.
    '''
    g = {}
    r = {}
    for ego in fresult.keys():
        if len(fresult[ego].keys()) > 0:
            g[ego] = {}
            altl = {}
            for alter in fresult[ego].keys():
                l = list(fresult[ego][alter]['lambda'].unique())[0]
                altl[l] = altl.get(l, 0) + 1
            for alter in fresult[ego].keys():
                for i in fresult[ego][alter].index:
                    a = fresult[ego][alter].at[i, 'alpha']
                    l = fresult[ego][alter].at[i, 'lambda']
                    f = fresult[ego][alter].at[i, 'f']
                    if xaxis == 'lambda':
                        g[ego][a] = g[ego].get(a, {})
                        g[ego][a][l] = g[ego][a].get(l, 0) + f
                    elif xaxis == 'alpha':
                        g[ego][l] = g[ego].get(l, {})
                        g[ego][l][a] = g[ego][l].get(a, 0) + f

            idx = 0
            df = pd.DataFrame()
            for k in g[ego].keys():
                for kk in g[ego][k].keys():
                    if xaxis == 'lambda':
                        df.at[idx, 'lambda'] = kk
                        df.at[idx, 'alpha'] = k
                        df.at[idx, 'g'] = g[ego][k][kk] / altl[kk]
                        idx += 1
                    elif xaxis == 'alpha':
                        df.at[idx, 'lambda'] = k
                        df.at[idx, 'alpha'] = kk
                        df.at[idx, 'g'] = g[ego][k][kk] / altl[k]
                        idx += 1
            if xaxis == 'lambda':
                df.sort_values(by=['lambda', 'alpha'], inplace=True)
            elif xaxis == 'alpha':
                df.sort_values(by=['alpha', 'lambda'], inplace=True)
            df.reset_index(drop=True, inplace=True)
            r[ego] = df
    return r


def f_histell(fresult, alpha_fixed, cut_points, uptoapoint=False, binned=False, deltaF=5):
    test = {}
    if uptoapoint:
        for ego in fresult.keys():
            for alter in fresult[ego].keys():
                df = fresult[ego][alter].loc[fresult[ego][alter]['alpha'] == alpha_fixed]
                if len(df) > 0:
                    lamb = df.iloc[0]['lambda']
                    for f in df['f']:
                        for cut in cut_points:
                            if f < cut:
                                df2 = df.loc[df['f'] < cut]
                                test[cut_points.index(cut)] = test.get(cut_points.index(cut), {})
                                test[cut_points.index(cut)][lamb] = test[cut_points.index(cut)].get(lamb, 0) + 1
                                
    elif binned:
        for ego in fresult.keys():
            for alter in fresult[ego].keys():
                fresult[ego][alter]['phi'] = fresult[ego][alter]['f'] // deltaF
                df = fresult[ego][alter].loc[fresult[ego][alter]['alpha'] == alpha_fixed]
                if len(df) > 0:
                    lamb = df.iloc[0]['lambda']
                    for phi in df['phi'].unique():
                        df2 = df.loc[df['phi'] == phi]
                        test[phi] = test.get(phi, {})
                        #test[phi][lamb] = test[phi].get(lamb, 0) + sum(df2['f'])
                        test[phi][lamb] = test[phi].get(lamb, 0) + 1
                
                            
    else:
        cutp = [0] + cut_points
        for ego in fresult.keys():
            for alter in fresult[ego].keys():
                df = fresult[ego][alter].loc[fresult[ego][alter]['alpha'] == alpha_fixed]
                if len(df) > 0:
                    lamb = df.iloc[0]['lambda']
                    for i in range(len(cutp) - 1):
                        df2 = df.loc[(df['f'] >= cutp[i]) & (df['f'] < cutp[i + 1])]
                        test[i] = test.get(i, {})
                        test[i][lamb] = test[i].get(lamb, 0) + 1
                        

    for i in test.keys():
        test[i] = pd.DataFrame.from_dict(test[i], orient='index')
        test[i].sort_index(inplace=True)
        
    return test

def hatf(callsdf, lives_dict, deltat = 2000):
    result = {}
    result['all'] = {}
    df2all = {}
    numtall = max(callsdf['uclock']) // deltat
    for ego in callsdf['ego'].unique():
        result[ego] = {}
        df = callsdf.loc[callsdf['ego'] == ego]
        df2 = {}
        numt = max(df['uclock']) // deltat
        for tau in range(numt + 1):
            df2[tau] = df.loc[(df['uclock'] >= tau * deltat) & (df['uclock'] < (tau + 1) * deltat)]
            altersell = [lives_dict[ego][alter]['ell'] for alter in df2[tau]['alter'].unique()]
            altersell.sort()
            for alter in df2[tau]['alter'].unique():
                ell = lives_dict[ego][alter]['ell']
                per = int(stats.percentileofscore(altersell, ell))
                df3 = df2[tau].loc[df2[tau]['alter'] == alter]
                result[ego][tau] = result[ego].get(tau, {})
                result['all'][tau] = result['all'].get(tau, {})
                result[ego][tau][per] = result[ego][tau].get(per, 0) + len(df3)
                result['all'][tau][per] = result['all'][tau].get(per, 0) + len(df3)
    res2 = {}
    for ego in result.keys():
        res2[ego] = {}
        for tau in result[ego].keys():
            res2[ego][tau] = pd.DataFrame.from_dict(result[ego][tau], orient='index', columns=['f'])
            res2[ego][tau].sort_index(inplace=True)
            res2[ego][tau]['tmp'] = res2[ego][tau]['f'].div(sum(res2[ego][tau]['f']))
            res2[ego][tau]['ftil'] = res2[ego][tau]['tmp'].cumsum()
            res2[ego][tau].drop(columns=['tmp'], inplace=True)
            
    return res2

def get_avg(df, perclist, tau=0):
    '''
    df must be produced with the hatf function
    '''
    tmp = {}
    for ego in df.keys():
        if ego != 'all':
            df1 = df[ego][tau]
            thisp = list(df1.index)
            for p in perclist:
                if p in thisp:
                    tmp[p] = tmp.get(p, [])
                    tmp[p].append(df1.at[p, 'ftil'])
                else:
                    for pp in range(len(thisp) - 1):
                        if ((p > thisp[pp]) and (p < thisp[pp + 1])) or (p < thisp[0]):
                            x0 = thisp[pp]
                            y0 = df1.at[x0, 'ftil']
                            x1 = thisp[pp + 1]
                            y1 = df1.at[x1, 'ftil']
                            break
                    m = (y1 - y0) / (x1 - x0)
                    F = (m * p) + y0 - (m * x0)
                    tmp[p] = tmp.get(p, [])
                    tmp[p].append(F)
    for p in tmp.keys():
        tmp[p] = np.mean(tmp[p])
        
    return tmp

def get_survival(fresult, alphafixed=1, base=2, unbinned=False):
    '''
    This function takes as an input an "f dataframe"; and returns a dictionary that uses
    the gamma bins of activity during month "alphafixed" as keys, and the survival probabilities
    of alters as the values, in a dataframe. For each value of ell*, there are survival
    probabilities. The arguments are:
    fresult             : dataframe created using the get_f method
    alphafixed          : which bin of a I'm interested in using
    base                : the base for the "exponential binning"
    unbinned            : do not create bins of activity, use only the actual value. By
                          default, this is set to False
    '''
    tmp = {}
    for ego in fresult.keys():
        for alter in fresult[ego].keys():
            if type(alphafixed) == int:
                df = fresult[ego][alter].loc[fresult[ego][alter]['alpha'] == alphafixed]
            else:
                df = fresult[ego][alter].loc[(fresult[ego][alter]['alpha'] >= alphafixed[0]) & (fresult[ego][alter]['alpha'] <= alphafixed[1])]
            if len(df) > 0:
                if unbinned:
                    F = sum(df['f'])
                else:
                    F = int(math.log(sum(df['f']), base))
                tmp[F] = tmp.get(F, {})
                lamb = df.iloc[0]['lambda']
                tmp[F][lamb] = tmp[F].get(lamb, 0) + 1
    tmp2 = {}
    for F in sorted(tmp.keys()):
        df = pd.DataFrame.from_dict(tmp[F], orient='index').sort_index()
        tmp2[F] = {}
        df['p'] = df[0].div(sum(df[0]))
        for lc in range(max(df.index) + 1):
            df2 = df.loc[df.index >= lc]
            tmp2[F][lc] = round(sum(df2['p']), 6)
        tmp2[F] = pd.DataFrame.from_dict(tmp2[F], orient='index').sort_index()
    return tmp2

def get_plateau(series, allowed=0.5):
    '''
    This function obtains the height of the plateau found in the plots of b.
    The arguments are:
    series           : This is a particular series produced with the plot_g function
    allowed          : how much variation do I allow for the vertical axis, before stopping
                       the plateau serching loop
    '''
    x = list(series.index)
    mid = int(len(x) / 2)
    xlow, xhigh = x[mid - 1], x[mid + 1]
    newdf = series.loc[(series.index >= xlow) & (series.index <= xhigh)]
    grow = 1
    while (xlow != x[0]) and (xhigh != x[-1]):
        grow += 1
        newxmin, newxmax = x[mid - grow], x[mid + grow]
        tmp = series.loc[(series.index >= newxmin) & (series.index <= newxmax)]
        epsilon = max(tmp['alpha']) - min(tmp['alpha'])
        if epsilon >= allowed:
            break
        else:
            xlow, xhigh = newxmin, newxmax
            newdf = tmp
    return (xlow, xhigh, np.mean(newdf['alpha']))

def histogram(array, bins, log=True):
    xl = sorted(list(array))
    xo = xl[0]
    xf = xl[-1]
    if log:
        lmu = np.log10(xf / xo) / bins
        mu = 10**lmu
    dx = (xf - xo) / bins
    h = {}
    if log:
        for x in xl:
            if x == xf:
                h[bins - 1] = h.get(bins - 1, 0) + 1
            else:
                i = np.log10(x / xo) // lmu
                h[i] = h.get(i, 0) + 1
    else:
        for x in xl:
            if x == xf:
                h[bins - 1] = h.get(bins - 1, 0) + 1
            else:
                i = int((x - xo) // dx)
                h[i] = h.get(i, 0) + 1
    df = pd.DataFrame.from_dict(h, orient='index', columns=['h'])
    df['pmf'] = df['h'].div(sum(df['h']))
    for i in df.index:
        if log:
            df.at[i, 'label'] = xo*(mu**i)
        else:
            df.at[i, 'label'] = xo + (dx * i)
    return df

def get_avgfa(fresult, lives, ell0, ellf):
    '''
    This method produces the curves of \bar{f}(a). It takes as an input
    
    fresult         : dataframe created using the get_f method
    lives           : a "lives dictionary" produced with the method of the same name
    ell0            : starting point of the lifetime group (includes this number)
    ellf            : ending point for the lifetime group (includes this number)
    
    Note that the output of this function is a pandas dataframe with a column "f"
    that contains the Y points to be plotted. The X points are in the index of the
    dataframe. It will only produce one curve averaging over all egos whose alters
    have lifetime between ell0 and ellf (inclusive).
    '''
    fi = {}
    for ego in fresult.keys():
        nalt = 0
        fi[ego] = {}
        for alter in fresult[ego].keys():
            ell = lives[ego][alter]['ell']
            if (ell >= ell0) and (ell <= ellf):
                df = fresult[ego][alter]
                nalt += 1
                for i in df.index:
                    a = df.at[i, 'alpha']
                    f = df.at[i, 'f']
                    fi[ego][a] = fi[ego].get(a, 0) + f
        for a in fi[ego].keys():
            fi[ego][a] /= nalt
    
    tmp = {}
    for ego in fi.keys():
        for a in fi[ego].keys():
            tmp[a] = tmp.get(a, [])
            tmp[a].append(fi[ego][a])
    for a in tmp.keys():
        tmp[a] = np.mean(tmp[a])
        
    res = pd.DataFrame.from_dict(tmp, orient='index', columns=['f'])
    res = res.sort_index()
    return res


def consecutive(callsdf, ello, ellf, dayres=1):
    allie = []
    cv = []
    df = callsdf.copy()
    df['ea'] = list(zip(df['ego'], df['alter']))
    lifetime = df.groupby('ea')[['aclock']].max()
    lifetime = lifetime.loc[(lifetime['aclock'] >= ello) & (lifetime['aclock'] <= ellf)]
    use = list(lifetime.index)
    df = df[df['ea'].isin(use)]   
    for ego in df['ego'].unique():
        df1 = df.loc[df['ego'] == ego]
        for alter in df1['alter'].unique():
            df2 = df1.loc[df['alter'] == alter]
            if len(df2) > 2:
                df2 = df2.sort_values(by='time')
                ie = list(df2['aclock'].diff())[1:]
                if dayres > 1:
                    ie = [x // dayres for x in ie]
                allie += ie
                cvego = np.nanstd(ie) / np.nanmean(ie)
                if not pd.isna(cvego):
                    cv.append(np.std(ie) / np.mean(ie))
    mean = np.mean(cv)
    std = np.std(cv)
    H = histogram(allie, 30, log=False)
    Hcv = histogram(cv, 30, log=False)
    return (H, Hcv, mean, std)


def calls_by_second(callsdf):
    df = callsdf.copy(deep=True)
    minT = min(df['time'])
    df['u'] = (df['time'] - minT).dt.total_seconds()
    df['ea'] = list(zip(df['ego'], df['alter']))
    ulocmin = df.groupby('ea')[['u']].min()
    df['a'] = df.index.map(lambda i: df.at[i, 'u'] - ulocmin['u'][df.at[i, 'ea']])
    df = df.sort_values(by=['ego', 'alter', 'time'])
    g = df.groupby('ea')[['time']].count()
    g = g.loc[g['time'] > 2]
    df = df[df['ea'].isin(g.index)]
    df = df.drop(columns=['ea'])
    return df


def consecutive_bys(callsdf, ello, ellf, nolives = True):
    allie = []
    cv = []
    df = callsdf.copy()
    df['ea'] = list(zip(df['ego'], df['alter']))
    if type(nolives) != pd.core.frame.DataFrame:
        lifetime = df.groupby('ea')[['aclock']].max()
        lifetime = lifetime.loc[(lifetime['aclock'] >= ello) & (lifetime['aclock'] <= ellf)]
        use = list(lifetime.index)
    else:
        df_alt = nolives.copy(deep=True)
        df_alt['ea'] = list(zip(df_alt['ego'], df_alt['alter']))
        lifetime = df_alt.groupby('ea')[['aclock']].max()
        lifetime = lifetime.loc[(lifetime['aclock'] >= ello) & (lifetime['aclock'] <= ellf)]
        use = list(lifetime.index)
    df = df[df['ea'].isin(use)]
    for ego in df['ego'].unique():
        df1 = df.loc[df['ego'] == ego]
        for alter in df1['alter'].unique():
            df2 = df1.loc[df['alter'] == alter]
            if len(df2) > 1:
                df2 = df2.sort_values(by='time')
                ie = list(df2['a'].diff())[1:]
                allie += ie
                cvego = np.nanstd(ie) / np.nanmean(ie)
                if not pd.isna(cvego):
                    cv.append(np.std(ie) / np.mean(ie))

    mean = np.mean(cv)
    std = np.std(cv)
    H = histogram(allie, 30, log=False)
    Hcv = histogram(cv, 30, log=False)
    return (H, Hcv, mean, std)


def gaps(callsdf, ello, ellf, dayres=1, zero=False):
    allgaps = []
    allcv = []
    df = callsdf.copy()
    df['ea'] = list(zip(df['ego'], df['alter']))
    lifetime = df.groupby('ea')[['aclock']].max()
    lifetime = lifetime.loc[(lifetime['aclock'] >= ello) & (lifetime['aclock'] <= ellf)]
    df1 = df[df['ea'].isin(lifetime.index)]
    for ego in df1['ego'].unique():
        df2 = df1.loc[df1['ego'] == ego]
        for alter in df2['alter'].unique():
            df3 = df2.loc[df2['alter'] == alter]
            if len(df3) > 1:
                gaps = []
                df4 = df3.sort_values(by='time')
                maxa = max(df4['aclock'])
                d = 0
                for i in range(maxa + 1):
                    if i not in df4['aclock'].unique():
                        d += 1
                    elif (not zero) and (d != 0):
                        gaps.append(d)
                        d = 0
                    elif zero and d == 0:
                        gaps.append(d)
                    elif zero and d > 0:
                        gaps.append(d)
                        d = 0                        
                if dayres > 1:
                    gaps = [x // dayres for x in gaps]
                allgaps += gaps
                cv = np.nanstd(gaps) / np.nanmean(gaps)
                if not pd.isna(cv):
                    allcv.append(cv)
                    
    H = histogram(allgaps, 30, log=False)
    H['label'] = H['label'].replace({0:0.1})
    Hcv = histogram(allcv, 30, log=False)
    mcv = np.mean(allcv)
    scv = np.std(allcv)
    return (H, Hcv, mcv, scv)