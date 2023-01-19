#!/usr/bin/env python3

import numpy as np
import pandas as pd
import networkx as nx
import datetime as dt
import scipy.stats as stats
import statsmodels.api as sm
import pymannkendall as mk
import copy
import os
import math

def oldcalls(infile, filt, ego, alter, timestamp, tstampformat, header=True, min_activity=1):
    '''
    Old code for the method "allcalls". The updated version is in this module.
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


def allcalls(infile, filt, ego, alter, timestamp, tstampformat, header=True, min_activity=1, duration=False):
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
    duration        : Defaults to False. If not False, identifies the column with the duration of the
                    phone call, in seconds
    '''
    if header:
        df = pd.read_csv(infile)
        tmp = pd.DataFrame()
        if len(filt) > 2:
            for i in range(1, len(filt)):
                df2 = df.loc[df[filt[0]] == filt[i]]
                tmp = tmp.append(df2)
        elif len(filt) == 2:
            tmp = df.loc[df[filt[0]] == filt[1]]
        else:
            tmp = copy.deepcopy(df)
    else:
        df = pd.read_csv(infile, header=None)
        tmp = pd.DataFrame()
        if len(filt) > 2:
            for i in range(1, len(filt)):
                df2 = df.loc[df[filt[0]] == filt[i]]
                tmp = tmp.append(df2)
        elif len(filt) == 2:
            tmp = df.loc[df[filt[0]] == filt[1]]
        else:
            tmp = copy.deepcopy(df)

    egocol = tmp[ego]
    altercol = tmp[alter]
    if duration:
        durcol = tmp[duration]
        newdf = pd.DataFrame({'ego': egocol, 'alter': altercol, 'duration': durcol})
    else:
        newdf = pd.DataFrame({'ego': egocol, 'alter': altercol})
    timecol = tmp[timestamp]
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
    newdf['date'] = newdf['time'].map(lambda i: i.strftime('%Y-%m-%d'))
    newdf['date'] = pd.to_datetime(newdf['date'], format='%Y-%m-%d')
    newdf.sort_values(by=['alter', 'ego', 'time'], ascending=[True, True, True], inplace=True)
    mindate = min(newdf['date'])
    newdf['t'] = newdf['date'].map(lambda i: (i - mindate).days)
    newdf['ea'] = list(zip(newdf['ego'], newdf['alter']))
    t0i = newdf.groupby('ea')['date'].min().rename({'date': 't0i'}, axis='columns')
    newdf['a'] = newdf.index.map(lambda i: (newdf.at[i, 'date'] - t0i[newdf.at[i, 'ea']]).days)
    ncalls = newdf.groupby('ea')[['time']].count()
    ncalls = ncalls.loc[ncalls['time'] >= min_activity]
    newdf = newdf[newdf['ea'].isin(ncalls.index)]
    newdf['ego'] = newdf['ego'].map(lambda i: hex(hash(i)))
    newdf['alter'] = newdf['alter'].map(lambda i: hex(hash(i)))
    newdf['ea'] = newdf['ea'].map(lambda i: hex(hash(i)))
    if duration:
        newdf.columns = ['ego', 'alter', 'duration','time', 'date', 't', 'pair', 'a']
        newdf = newdf[['ego', 'alter', 'pair', 'time', 'date', 'duration', 't', 'a']]
    else:
        newdf.columns = ['ego', 'alter', 'time', 'date', 't', 'pair', 'a']
        newdf = newdf[['ego', 'alter', 'pair', 'time', 'date', 't', 'a']]
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

def apply_filters(unf_calls, delta):
    '''
    This function implements the following filters for the data:
    
    1. Removes dtimestamp duplicates for all ego-alter pairs
    2. Removes all ego-alter pairs with fewer than 3 calls
    3. Removes all pairs with any contact in the interval
       [T - delta, T)
    '''
    T = max(unf_calls['uclock'])
    df = unf_calls.copy(deep=True)
    df['ea'] = list(zip(df['ego'], df['alter']))
    df = df.sort_values(by=['ea', 'time'])
    df['shifted'] = df['time'].shift(-1)
    df['d'] = (df['shifted'] - df['time']).dt.total_seconds()
    torm = list(df.loc[df['d'] == 0].index)
    df = df.drop(torm)
    df = df.drop(columns = ['shifted', 'd'])
    ncalls = df.groupby('ea')[['time']].count().rename(columns={'time': 'ncalls'})
    ncalls = ncalls.loc[ncalls['ncalls'] > 2]
    df = df[df['ea'].isin(ncalls.index)]
    tmp = df.loc[df['uclock'] > (T - delta)]
    rmpairs = list(tmp['ea'].unique())
    df2 = df[~df['ea'].isin(rmpairs)]
    df3 = df2.drop(columns=['ea']).reset_index(drop=True)
    return df3

def apply_filters2(unf_calls, delta):
    '''
    This function implements the following filters for the data:
    
    1. Removes dtimestamp duplicates for all ego-alter pairs
    2. Removes all ego-alter pairs with fewer than 3 calls
    3. Removes all pairs with any contact in the interval
       [T - delta, T)
    '''
    T = max(unf_calls['t'])
    df = unf_calls.copy(deep=True)
    df = df.sort_values(by=['pair', 'time'])
    df['shifted'] = df['time'].shift(-1)
    df['d'] = (df['shifted'] - df['time']).dt.total_seconds()
    torm = list(df.loc[df['d'] == 0].index)
    df = df.drop(torm)
    df = df.drop(columns = ['shifted', 'd'])
    ncalls = df.groupby('pair')[['time']].count().rename(columns={'time': 'ncalls'})
    ncalls = ncalls.loc[ncalls['ncalls'] > 2]
    df = df[df['pair'].isin(ncalls.index)]
    tmp = df.loc[df['t'] > (T - delta)]
    rmpairs = list(tmp['pair'].unique())
    df2 = df[~df['pair'].isin(rmpairs)]
    return df2


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

def plot_b(gresult, xaxis='alpha'):
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
    Instead of dividing the sum of f_i(lambda, alpha) by
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

def get_survival(fresult, alphafixed=1, base=2, unbinned=False, lambdamax=999, countA=False, externalell=False, Dell=10):
    '''
    This function takes as an input an "f dataframe"; and returns a dictionary that uses
    the gamma bins of activity during month "alphafixed" as keys, and the survival probabilities
    of alters as the values, in a dataframe. For each value of ell*, there are survival
    probabilities. The arguments are:
    fresult             : dataframe created using the get_f method
    alphafixed          : which bin of a I'm interested in using. If a tuple is passed, it will use alters with
                          the a delimited by the first and last values of the tuple. Default value is 1
    base                : the base for the "exponential binning"
    unbinned            : do not create bins of activity, use only the actual value. By
                          default, this is set to False
    lambdamax           : Ignore any lifetime (binned) value greater than this. Default is 9999
    countA              : Count alters for each series of g. Defaults to False. If True, the function returns
                          a tuple with the first element being the usual result, and the second element 
                          the number of alters
    externalell         : If False, it will extract ell (binned) from the fresult dataframe. If a dictionary is supplied,
                          it will extract lifetimes from here. The dictionary supplied must come from the lives_dictionary
                          function. Defaults to False.
    Dell                : Delta ell to be used. Only considered when the externalell argument is not false. Defaults to 10.
    '''
    tmp = {}
    altcount = {}
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
                if type(externalell) == bool:
                    lamb = df.iloc[0]['lambda']
                else:
                    lamb = externalell[ego][alter]['ell'] // Dell
                if lamb <= lambdamax:
                    tmp[F] = tmp.get(F, {})
                    altcount[F] = altcount.get(F, 0) + 1
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
    if countA:
        return (tmp2, altcount)
    else:
        return tmp2

def get_plateau(series, pstar=0.1, arbxo=2, arbxf=2):
    '''
    This function obtains the height of the plateau found in the plots of \bar{f}(a).
    If it does not find a plateau by looking at the slopes of lines produced with an 
    OLS estimation, it draws a line with height equal to the average of the heights 
    in the interval [mina + arbxo, mina - arbxf]. The arguments are:
    
    series           : This is a particular series that contains the values for \bar{f}(a)
    pstar            : The minimum p-value accepted to asume the line has slope 0. defaults to 0.1
    arbxo            : If I did not find a plateau, arbitrarily use mina + arbxo as the starting point
    arbxf            : If I did not find a plateau, arbitrarily use maxa - arbxf as the ending point
    
    Returns a list with the coordinates of the two points that delimit the line.
    '''
    mida = max(series.index) // 2
    for i in range(1, mida + 1):
        newdf = series.loc[(series.index >= mida - i) & (series.index <= mida + i)]
        if len(newdf) > 1:
            X, Y = sm.add_constant(newdf.index), newdf['f']
            tmp = sm.OLS(Y, X).fit()
            if tmp.pvalues[1] < pstar:
                df = series.loc[(series.index >= mida - (i - 1)) & (series.index <= mida + (i - 1))]
                xo, yo = min(df.index), np.mean(df['f'])
                xf, yf = max(df.index), yo
                if xo == xf:
                    continue
                else:
                    return [(xo, yo), (xf, yf)]
    else:
        xo = min(series.index) + arbxo
        xf = max(series.index) - arbxf
        df = (series.loc[(series.index >= xo) & (series.index <= xf)])
        yo = np.mean(df['f'])
        yf = yo
        return [(xo, yo), (xf, yf)]

<<<<<<< HEAD
def histogram(array, bins, log=True, base=10, int1=False):
=======
def histogram(array, bins, log=True):
>>>>>>> fe29459643a2154374593d3273b50f5aac307273
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
    # df = df.reindex(range(bins), fill_value=0)
    df['pmf'] = df['h'].div(sum(df['h']))
    df['pdf'] = df['pmf'] / dx
    for i in df.index:
        if log:
            df.at[i, 'label'] = xo*(mu**i)
        else:
            df.at[i, 'label'] = xo + (dx * (i + 0.5))
    return df

def get_avgfa(fresult, lives, ell0, ellf, countalt=False):
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
    unialt = 0
    egolist = []
    for ego in fresult.keys():
        nalt = 0
        fi[ego] = {}
        for alter in fresult[ego].keys():
            if (ego in lives.keys()) and (alter in lives[ego].keys()):
                ell = lives[ego][alter]['ell']
                if (ell >= ell0) and (ell <= ellf):
                    df = fresult[ego][alter]
                    nalt += 1
                    unialt += 1
                    if ego not in egolist:
                        egolist.append(ego)
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
    if countalt:
        return (res, unialt, len(egolist))
    else:
        return res

def get_fal(calls, ello, ellf, bina, countalters=False):
    '''
    Similar to the get_avgfa() function on this module. The main difference
    is that in this function there is no intermidiate step. From the calls
    dataframe, I obtain the lifetime of alters and place them in the interval
    of interest, or not. The arguments are
    
    calls             : a dataframe produced with the "allcalls" or "apply_filters" methods
    ello              : lowest lifetime (in days) to keep
    ellf              : highest lifetime (in days) to keep
    bina              : value for Delta a
    
    The function returns a dictionary with two keys. The key "f" holds the values for 
    \bar{f}(alpha). The key "fi" contains a dictionary in which each entry is an ego, and 
    the values stored are \bar{f}_{i}(alpha) 
    '''
    df = calls.copy(deep=True)
    df['ea'] = list(zip(df['ego'], df['alter']))
    lf = df.groupby('ea')[['aclock']].max()
    lf = lf.loc[(lf['aclock'] >= ello) & (lf['aclock'] <= ellf)]
    df = df[df['ea'].isin(lf.index)]
    nalters = len(df['ea'].unique())
    egocount = len(df['ego'].unique())
    fi = {}
    maxt = 0
    for ego in df['ego'].unique():
        df1 = df.loc[df['ego'] == ego]
        mina = min(df1['aclock'])
        df2 = df1.copy()
        df2['a'] = df1['aclock'] // bina
        callsa = df2.groupby('a')[['time']].count().rename({'time': 'f'}, axis='columns')
        callsa['f'] /= len(df2['alter'].unique())
        callsa = callsa.sort_index()
        fi[ego] = callsa
    
    tmp = {}
    for ego in fi.keys():
        for i in fi[ego].index:
            tmp[i] = tmp.get(i, [])
            tmp[i].append(fi[ego].at[i, 'f'])
            
    tmp2 = {}
    for i in tmp.keys():
        tmp2[i] = np.nanmean(tmp[i])
        
    if len(tmp) > 1:
        f = pd.DataFrame.from_dict(tmp2, orient='index')
        f = f.sort_index()
        f.columns = ['f']
        if countalters:
            return {'f': f, 'fi': fi, 'nalters': nalters, 'egocount':egocount}
        else:
            return {'f': f, 'fi': fi}


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

def get_b_slopes(series, patternsize=3, FlagConverge=False):
    allslopes = []
    X = list(series.index)
    N = len(X)
    xo, xf = X[0], X[-1]
    yo, yf = series.at[xo, 'f'], series.at[xf, 'f']
    slope = (yf - yo) / (xf - xo)
    allslopes.append(slope)
    for i in range(1, N):
        newx = X[i // 2: N - ((i + 1) // 2)]
        if len(newx) > 1:
            xo, xf = newx[0], newx[-1]
            yo, yf = series.at[xo, 'f'], series.at[xf, 'f']
            slope = (yf - yo) / (xf - xo)
            allslopes.append(slope)
        else:
            xo, xf = X[1], X[-2]
            df = series.loc[(series.index >= xo) & (series.index <= xf)]
            yo = np.mean(df['f'])
            yf = yo
            if FlagConverge:
                return [[xo, xf], [yo, yf], False]
            else:
                return [[xo, xf], [yo, yf]]
        if (len(allslopes) >= patternsize):
            checkSlopes = list(np.sign(allslopes[-patternsize:]))
            if checkSlopes.count(checkSlopes[0]) != len(checkSlopes):
                df = series.loc[(series.index >= xo) & (series.index <= xf)]
                yo = np.mean(df['f'])
                yf = yo
                if FlagConverge:
                    return [[xo, xf], [yo, yf], True]
                else:
                    return [[xo, xf], [yo, yf]]
                
def get_survival2(callsdf, ao, af, maxell=250, binell=10, binned=True, base=2):
    cdf = callsdf.loc[callsdf['aclock'] <= maxell].copy()
    cdf['ea'] = list(zip(cdf['ego'], cdf['alter']))
    lf = cdf.groupby('ea')[['aclock']].max().rename({'aclock': 'ell'}, axis='columns')
    lf['lambda'] = lf['ell'] // binell
    tmp = cdf.loc[(cdf['aclock'] >= ao) & (cdf['aclock'] <= af)]
    vol = tmp.groupby('ea')[['time']].count().rename({'time': 'g'}, axis='columns')
    vol['gamma'] = vol['g'].map(lambda i: int(math.log(i, base)))
    vol = vol.merge(lf, left_index=True, right_index=True, how='left')
    result = {}
    if binned:
        for gamma in sorted(vol['gamma'].unique()):
            dfg = vol.loc[vol['gamma'] == gamma]
            dfg2 = dfg.groupby('lambda')[['gamma']].count().rename({'gamma': 'count'}, axis='columns').sort_index()
            dfg2['prop'] = dfg2['count'].div(sum(dfg2['count']))
            dfg3 = pd.DataFrame(index=range(max(dfg2.index) + 1))
            dfg3 = dfg3.merge(dfg2, left_index=True, right_index=True, how='outer').fillna(0)
            for i in dfg3.index:
                tmp2 = dfg3.loc[dfg3.index >= i]
                dfg3.at[i, 'p'] = sum(tmp2['prop'])
            dfg3.index.rename('a', inplace=True)
            result[gamma] = dfg3[['p']]
    else:
        for g in sorted(vol['g'].unique()):
            dfg = vol.loc[vol['g'] == g]
            dfg2 = dfg.groupby('lambda')[['gamma']].count().rename({'gamma': 'count'}, axis='columns').sort_index()
            dfg2['prop'] = dfg2['count'].div(sum(dfg2['count']))
            dfg3 = pd.DataFrame(index=range(max(dfg2.index) + 1))
            dfg3 = dfg3.merge(dfg2, left_index=True, right_index=True, how='outer').fillna(0)
            for i in dfg3.index:
                tmp2 = dfg3.loc[dfg3.index >= i]
                dfg3.at[i, 'p'] = sum(tmp2['prop'])
            dfg3.index.rename('a', inplace=True)
            result[g] = dfg3[['p']]
    return result

def get_b_mk(series, FlagConverge=False):
    '''
    This method takes a series of f(a) and gets the "steady region" height. It does it by using the Mann-Kendall
    test for trends. If it finds a trend, the algorithm keeps iterating. When it find no trend, it will output
    coordinates for the x and y axes.
    
    The arguments for this function are:
    series          : the series of f(a) to obtain b(ell)
    
    This function returns a list with the following elements: the values for the starting and ending points in
    the horizontal axis; the values for the vertical axis;
    '''
    X = sorted(list(series.index))
    N = len(series)
    q = 0
    tmp = mk.original_test(series['f'])
    if tmp[0] == 'notrend':
        if FlagConverge:
            return [[X[0], X[-1]], [np.mean(series['f']), np.mean(series['f'])], True, q]
        else:
            return [[X[0], X[-1]], [np.mean(series['f']), np.mean(series['f'])], q]
    else:
        for i in range(1, N):
            newx = X[i // 2: N - ((i + 1) // 2)]
            df = series.loc[(series.index >= newx[0]) & (series.index <= newx[-1])]
            if len(df) > 1:
                tmp = mk.original_test(df['f'])
                if tmp[0] == 'notrend':
                    if FlagConverge:
                        return [[newx[0], newx[-1]], [np.mean(df['f']), np.mean(df['f'])], True, q]
                    else:
                        return [[newx[0], newx[-1]], [np.mean(df['f']), np.mean(df['f'])], q]
                    q += 1
        else:
            df = series.loc[(series.index >= X[1]) & (series.index <= X[-2])]
            if len(df) > 0:
                if FlagConverge:
                    return [[list(df.index)[0], list(df.index)[-1]], [np.mean(df['f']), np.mean(df['f'])], False, 999]
                else:
                    return [[list(df.index)[0], list(df.index)[-1]], [np.mean(df['f']), np.mean(df['f'])], 999]
            else:
                df = series.loc[(series.index >= X[0]) & (series.index <= X[-1])]
                if FlagConverge:
                    return [[list(df.index)[0], list(df.index)[-1]], [np.mean(df['f']), np.mean(df['f'])], False, 999]
                else:
                    return [[list(df.index)[0], list(df.index)[-1]], [np.mean(df['f']), np.mean(df['f'])], 999]