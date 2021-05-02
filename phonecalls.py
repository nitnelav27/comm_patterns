#!/usr/bin/env python3

import numpy as np
import pandas as pd
import networkx as nx
import datetime as dt
import copy
import os

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
            df3.sort_values(by='time', inplace=True)
            if not external_lives:
                lamb = (df3.iloc[-1]['uclock'] - df3.iloc[0]['uclock']) // binell
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
