#! /usr/bin/env python

print 'v3'

import os
import re
import sys
import imp
import os.path as op
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# for dprime
from scipy.stats import norm
from math import exp,sqrt
Z = norm.ppf

def gather_experiment_info(exp_name=None, dirs=None, altmodel=None):
    """Import an experiment module and add some formatted information."""

    # Import the base experiment
    exp_file = op.join(dirs["analydir"], exp_name + ".py")
    exp = imp.load_source(exp_name, exp_file)
    exp_dict = dict() # could load in default here, and just replace a few things

    def keep(k):
        return not re.match("__.*__", k)

    exp_dict.update({k: v for k, v in exp.__dict__.items() if keep(k)})

    # Possibly import the alternate model details
    if altmodel is not None:

    	alt_file = op.join(dirs["analydir"], "%s-%s.py" % (exp_name, altmodel))
    	alt = imp.load_source(altmodel, alt_file)
    	alt_dict = {k: v for k, v in alt.__dict__.items() if keep(k)}
        # Update the base information with the altmodel info
        exp_dict.update(alt_dict)

    # Save the __doc__ attribute to the dict
    exp_dict["comments"] = "" if exp.__doc__ is None else exp.__doc__
    if altmodel is not None:
        exp_dict["comments"] += "" if alt.__doc__ is None else alt.__doc__


    return exp_dict


def dPrime_list(hits, fas, olds, news):
    # Floors an ceilings are replaced by half hits and half FA's
    halfHit = 0.5/olds
    halfFa = 0.5/news

    # Calculate hitrate and avoid d' infinity
    hitRate = hits/olds
    hitRate[hitRate == 1] = 1-halfHit[hitRate == 1]
    hitRate[hitRate == 0] = halfHit[hitRate == 0]

    # Calculate false alarm rate and avoid d' infinity
    faRate = fas/news
    faRate[faRate == 1] = 1-halfFa[faRate == 1]
    faRate[faRate == 0] = halfFa[faRate == 0]

    # Return d', beta, c and Ad'
    out = {}
    out['d'] = Z(hitRate) - Z(faRate)
    out['beta'] = [exp(x)/2 for x in Z(faRate)**2 - Z(hitRate)**2]
    out['c'] = -(Z(hitRate) + Z(faRate))/2
    out['Ad'] = norm.cdf(out['d']/sqrt(2))
    return out


def dPrime(hits, fas, olds, news):
    # Floors an ceilings are replaced by half hits and half FA's
    halfHit = 0.5/olds
    halfFa = 0.5/news

    # Calculate hitrate and avoid d' infinity
    hitRate = hits/olds
    if hitRate == 1: hitRate = 1-halfHit
    if hitRate == 0: hitRate = halfHit

    # Calculate false alarm rate and avoid d' infinity
    faRate = fas/news
    if faRate == 1: faRate = 1-halfFa
    if faRate == 0: faRate = halfFa

    # Return d', beta, c and Ad'
    out = {}
    out['d'] = Z(hitRate) - Z(faRate)
    out['beta'] = exp(Z(faRate)**2 - Z(hitRate)**2)/2
    out['c'] = -(Z(hitRate) + Z(faRate))/2
    out['Ad'] = norm.cdf(out['d']/sqrt(2))
    return out

def add_subinfo(data, info_dict, col_name):
# Example: add_subinfo(dt, genders, 'gender')

    for info in info_dict.keys():
            data.loc[data.subid.isin(info_dict[info]), col_name] = info

    return data

def calculate_spikes(speed, trial_type, verbose=True):
    
    '''
    Calculate spikes in speed
    speed: series (e.g., data.speed)
    trial_type: str, habit or shortcut
    '''
    z_speed = pd.Series(sp.stats.zscore(speed))
    
    # Flag pauses/downward spikes in speed
    mask = z_speed < -3 # spike = 3 SD below mean
    
    # add a spike if start out slow
    if mask[0]:
        add_spike = 1
        if verbose:
            print 'adding spike for slow start'
    else: add_spike = 0
    
    # Estimate spike #
    if trial_type == 'habit':
        remove_end = 4 # remove last few samples when lining up to cross finish line
    else: remove_end = 0
    spike_count = mask[:len(mask)-remove_end].diff().sum()/2 + add_spike
    
    return spike_count, mask

# Plotting functions
def plot_environment(env, proj, dirs, lower_lim=-2):

    fig = plt.figure(figsize=(6, 6))
    plt.ylim(lower_lim,60)
    plt.xlim(lower_lim,60)
    
    buildings = pd.read_csv(dirs['building_coords'])

    if env in buildings.env.unique():

        coords = buildings[buildings.env == env]
        plt.scatter(coords.x, coords.y,  
                    s=25, marker='.', color='gray', label='_nolegend_')

    goals = pd.read_csv(dirs['goal_coords'])
    goal_types = proj['goals'][env].keys()
    for goal_type in goal_types:
        goal = proj['goals'][env][goal_type]
        color = proj['palette'][goal_type]

        plt.scatter(goals.loc[goals.item == goal].x.astype(float),  
                    goals.loc[goals.item == goal].y.astype(float),  
                    s=400, marker='*', c=color, label=goal_type)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax = fig.get_axes()[0]
    ax.tick_params(labelbottom='off', 
                   labelleft='off')
    return fig, ax

# Plotting from parsed data
def plot_path_z(data, proj, dirs, z_dim='speed', reverse=True, scale_fac=200):
    env = data.env.unique()[0]
    fig, ax = plot_environment(env, proj, dirs)
    
    
    # norm the z var, make positive
    z_var = sp.stats.zscore(data[z_dim].astype(float))
    z_var = z_var + np.abs(z_var.min()) + 1
    
    if reverse:
        z_var = 1/z_var * scale_fac
    else:
        z_var = z_var * scale_fac
    
    plt.scatter(data.x.astype(float),  
                data.y.astype(float),
                c=z_var, s=z_var, 
                marker='o', alpha=.5, cmap=plt.cm.PiYG_r)

    # Determine starting direction, add arrow
    x_move = np.diff(np.array(data.x)[:2])
    y_move = np.diff(np.array(data.y)[:2])
    
    if np.abs(x_move) > np.abs(y_move):
        if x_move > 0:
            tri_dir = '>'
        else: tri_dir = '<'
    else:
        if y_move > 0:
            tri_dir = '^'
        else: tri_dir = 'v'
    plt.scatter(np.array(data.x)[0], np.array(data.y)[0], 
                color='black', s=200, marker=tri_dir)
    
    ax = fig.get_axes()[0]
    return fig, ax


# Plotting for coding (from dp) -- need to update!
def plot_paths(env, subj, dp, proj, dirs):
    fig, ax = plot_environment(env, proj, dirs)
    plt.scatter(dp[(dp.env == env) & (dp.subid == subj) & (dp.c3 == "PandaEPL_avatar")].x.astype(float),  
                dp[(dp.env == env) & (dp.subid == subj) & (dp.c3 == "PandaEPL_avatar")].y.astype(float),
                c=dp[(dp.env == env) & (dp.subid == subj) & (dp.c3 == "PandaEPL_avatar")].time.astype(float),
                s=5, marker='o', alpha=1)
    ax = fig.get_axes()[0]
    return fig, ax
    

def plot_paths_group(env, proj, dp):
    fig, ax = plot_environment(env, proj)
    plt.scatter(dp[(dp.env == env) & (dpt.subid.isin(subj_list)) & (dp.c3 == "PandaEPL_avatar")].x.astype(float),  
                dp[(dp.env == env) & (dpt.subid.isin(subj_list)) & (dp.c3 == "PandaEPL_avatar")].y.astype(float),
                s=.5, marker='.', alpha=.3)
    ax = fig.get_axes()[0]
    return fig, ax
    
def plot_path(env, subj, goal, dpt, proj, dp):
    fig, ax = plot_environment(env, dp, proj)
    plt.scatter(dpt[(dpt.env == env) & (dpt.subid == subj) & (dpt.c3 == "PandaEPL_avatar") & (dpt.instructions == goal)].x.astype(float),  
                dpt[(dpt.env == env) & (dpt.subid == subj) & (dpt.c3 == "PandaEPL_avatar") & (dpt.instructions == goal)].y.astype(float),
                s=.5, marker='.', alpha=.3)
    ax = fig.get_axes()[0]
    return fig, ax
    
def plot_path_group(env, subj_list, goal, dpt, proj, dp):
    fig, ax = plot_environment(env, dp, proj)
    plt.scatter(dpt[(dpt.env == env) & (dpt.subid.isin(subj_list)) & (dpt.c3 == "PandaEPL_avatar") & (dpt.instructions == goal)].x.astype(float),  
                dpt[(dpt.env == env) & (dpt.subid.isin(subj_list)) & (dpt.c3 == "PandaEPL_avatar") & (dpt.instructions == goal)].y.astype(float),
                s=.5, marker='.', alpha=.3)
    ax = fig.get_axes()[0]
    return fig, ax
    