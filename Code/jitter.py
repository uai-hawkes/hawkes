import os
import sys

from absl import logging
import collections
from collections import defaultdict
import functools
import io
import itertools
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
import random
import seaborn
from scipy.ndimage import gaussian_filter1d
import scipy.interpolate
import sklearn.model_selection
from tqdm import tqdm
import time
import warnings

import util


class JitterTool(object):
  """Toolbox related to jitter methods."""

  @classmethod
  def wait(
      cls, 
      wait_time=0.001):
    """Used to test parallel computing."""
    time.sleep(wait_time)


  @classmethod
  def bin_spike_times(
      cls,
      spike_times,
      bin_width,
      len_trial=None,
      trial_window=None):
    """Convert spike times to spike bins, spike times list to spike bins matrix.

    # spike times outside the time range will not be counted in the bin at the
    # end. A time bin is left-closed right-open [t, t+delta).
    # e.g. t = [0,1,2,3,4], y = [0, 0.1, 0.2, 1.1, 5, 6]
    # output: [3, 1, 0, 0]

    Args:
      spike_times: The format can be list, np.ndarray.
    """
    if trial_window is None:
      trial_window = [0, len_trial]
    bins = np.arange(trial_window[0], trial_window[1]+bin_width, bin_width)
    num_bins = len(bins)

    if len(spike_times) == 0:
      return np.zeros(num_bins-1), bins[:-1]

    # multiple spike_times.
    elif isinstance(spike_times[0], list) or isinstance(spike_times[0], np.ndarray):
      num_trials = len(spike_times)
      num_bins = num_bins - 1
      spike_hist = np.zeros((num_trials, num_bins))
      for r in range(num_trials):
        spike_hist[r], _ = np.histogram(spike_times[r], bins)

    # single spike_times.
    else:
      spike_hist, _ = np.histogram(spike_times, bins)
    return spike_hist, bins[:-1]


  @classmethod
  def spike_times_statistics(
      cls,
      spike_times,
      trial_length,
      hist_bin_width=0.005,
      verbose=1):
    """Check basic spike times statistics."""
    def list_depth(x):
      for item in x:
        if not isinstance(item, list) and not isinstance(item, np.ndarray):
          return 1
        # Empty spike train. Only happens at last second layer.
        elif len(item) == 0:
          return 2
        else:
          return list_depth(item)+1

    num_layers = list_depth(spike_times)
    if num_layers == 1:
      num_nodes = 1
      num_trials = 1
    if num_layers == 2:
      num_trials = len(spike_times)
      num_nodes = 1
    elif num_layers == 3:
      num_nodes = len(spike_times)
      num_trials = len(spike_times[0])
    print(f'layers {num_layers}, nodes {num_nodes}, trials {num_trials}')

    if num_layers == 2:
      # Spike counts.
      spike_cnts = np.zeros(num_trials)
      for r in range(num_trials):
        spikes = np.array(spike_times[r])
        spike_cnts[r] = len(spikes)
      mean = np.mean(spike_cnts)
      var = np.std(spike_cnts) ** 2
      mean_fr = spike_cnts.sum() / trial_length / num_trials

      # Inter-spike intervals.
      isi = []
      for r in range(num_trials):
        isi_trial = list(np.diff(np.array(spike_times[r])))
        isi.extend(isi_trial)
      scale_hat = np.mean(isi)
      spike_hist, bins = cls.bin_spike_times(
          spike_times, hist_bin_width, trial_length)

      print(f'meanFR {np.round(mean_fr,3)}\tmeanISI {np.round(1/ scale_hat,3)}')
      if verbose < 1:
        return

      gs_kw = dict(width_ratios=[1,1.5], height_ratios=[1])
      fig, axs = plt.subplots(figsize=(6, 2),
          gridspec_kw=gs_kw, nrows=1, ncols=2)
      x = np.arange(0, 6*np.max(scale_hat), np.min(scale_hat)/10)
      ax = fig.add_subplot(axs[0])
      y_exp = scipy.stats.expon.pdf(x, scale=scale_hat)
      plt.plot(x, y_exp, 'r')
      seaborn.distplot(isi, bins=100, kde=False, norm_hist=True, color='grey')
      plt.xlim(-0.05*x[-1], x[-1])

      ax = fig.add_subplot(axs[1])
      plt.plot(bins, spike_hist.mean(axis=0) / hist_bin_width, c='k')
      # index_range = bins > (trial_length * 0.6)
      # mean_fr = np.mean(spike_hist[:,index_range]) / hist_bin_width
      # plt.axhline(y=mean_fr, ls='--', color='r', label='simulation mean')
      # plt.ylim(ylim)
      plt.show()

    if num_layers == 3:
      # Spike counts.
      spike_cnts = np.zeros([num_nodes, num_trials])
      for n in range(num_nodes):
        for r in range(num_trials):
          spikes = np.array(spike_times[n][r])
          # spikes = spikes[spikes > trial_length/3]
          spike_cnts[n,r] = len(spikes)
      mean = np.mean(spike_cnts, axis=1)
      var = np.std(spike_cnts, axis=1) ** 2
      mean_fr = spike_cnts.sum() / trial_length / num_trials

      # Inter-spike intervals.
      isi = [[] for _ in range(num_nodes)]
      scale_hat = np.zeros(num_nodes)
      for n in range(num_nodes):
        for r in range(num_trials):
          isi_trial = list(np.diff(np.array(spike_times[n][r])))
          isi[n].extend(isi_trial)
        scale_hat[n] = np.mean(isi[n])

      print(f'mean {np.round(mean, 3)}\n'+
            f'meanISI {np.round(1/ scale_hat,3)}')
      if verbose < 1:
        return

      num_cols = min(8, num_nodes)
      num_rows = np.ceil(num_nodes / num_cols).astype(int)
      gs_kw = dict(width_ratios=[1]*num_cols, height_ratios=[1]*num_rows)
      fig, axs = plt.subplots(figsize=(3*num_cols, 2*num_rows),
          gridspec_kw=gs_kw, nrows=num_rows, ncols=num_cols)
      plt.subplots_adjust(hspace=0.1, wspace=0.2)
      axs = axs.reshape(-1) if num_cols > 1 else [axs]
      x = np.arange(0, 6*np.max(scale_hat), np.min(scale_hat)/10)
      for n in range(num_nodes):
        ax = fig.add_subplot(axs[n])
        y_exp = scipy.stats.expon.pdf(x, scale=scale_hat[n])
        plt.plot(x, y_exp, 'r')
        seaborn.distplot(isi[n], bins=100, kde=False, norm_hist=True, color='grey')
        plt.xlim(-0.05*x[-1], x[-1])

      plt.show()


  @classmethod
  def plot_p_value_distribution(
      cls,
      p_vals_list,
      bin_width=0.02,
      ylim=[0, 2],
      file_path=None):
    """Plot the distribution of p-values.

    Args:
      p_vals_list: A list of list. It can plot multiple plots.
    """
    num_plots = len(p_vals_list)
    bins = np.arange(0, 1+bin_width, bin_width)
    plt.figure(figsize=[4 * num_plots, 2.5])
    for i in range(num_plots):
      ax = plt.subplot(1, num_plots, i+1)
      num_spikes = len(p_vals_list[i])
      seaborn.distplot(p_vals_list[i], bins=bins,
          kde=False, norm_hist=True, color='grey')
      plt.axhline(y=1, ls='--', c='k')
      plt.ylim(ylim)
      plt.xlim(-0.03, 1.03)
      plt.yticks([0, 1], [0, 1])
      # plt.xticks([0, 1], [0, 1])
      if i == 0:
        plt.xlabel('p-value')

    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      print('Save figure to:', file_path)
    plt.show()


  @classmethod
  def plot_p_value_qq_plot(
      cls,
      p_vals,
      test_size=0.05,
      file_path=None):
    """Plot the distribution of p-values.

    Args:
      p_vals_list: A list of list. It can plot multiple plots.
    """
    num_sims = len(p_vals)
    quantiles = np.linspace(0, 1, 1000)
    ecdf = np.zeros_like(quantiles)
    mcdf = np.zeros_like(quantiles)

    for q, quantile in enumerate(quantiles):
      ecdf[q] = (p_vals <= quantile).sum() / num_sims
      mcdf[q] = quantile

    c_alpha = np.sqrt(-np.log(test_size / 2) / 2)
    CI_up = mcdf + c_alpha/np.sqrt(num_sims)
    CI_dn = mcdf - c_alpha/np.sqrt(num_sims)

    gs_kw = dict(width_ratios=[1], height_ratios=[1]*1)
    fig, axs = plt.subplots(figsize=(3.3, 3), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    ax = fig.add_subplot(axs)
    plt.plot(mcdf, CI_up, '--', c='lightgrey', lw=1, label='95% CI')
    plt.plot(mcdf, CI_dn, '--', c='lightgrey', lw=1)
    plt.plot(mcdf, ecdf, 'k', lw=1.5)
    plt.axis([-0.015, 1.01, -0.01, 1.01])
    plt.xticks([0, 1], [0, 1])
    plt.yticks([0, 1], [0, 1])
    plt.xlabel('Theoretical quantile')
    plt.ylabel('Numerical quantile')
    # plt.legend(loc='lower right')

    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      print('Save figure to:', file_path)
    plt.show()


  @classmethod
  def jitter_hist_basic(
      cls,
      spike_bins,
      jitter_window_width,):
    """Jitter spike bins (as opposed to binned spike times).

    Args:
      spike_bins: array of spike times.
      jitter_window_width: jitter window width.

    Returns:
      jittered_spike_bins: array of spike times.
    """
    pass


  @classmethod
  def jitter_bins_interval(
      cls,
      spike_bins,
      jitter_window_width,):
    """Jitter spike bins (as opposed to binned spike times).

    Args:
      spike_bins: array of spike times.
      jitter_window_width: jitter window width.

    Returns:
      jittered_spike_bins: array of spike times.
    """
    pass


  @classmethod
  def jitter_spike_times_basic_single(
      cls,
      spike_times,
      jitter_window_width=0.02,
      num_jitter=1,
      verbose=False):
    """Jitter spike (as opposed to binned spike train) centered at each spike.
    This is for only one spike train.

    Platkiewicz Stark Amarasingham 2017 - Spike-centered jitter can mistake
    temporal structure.

    Args:
      spike_times: 1-D array of spike times.
      jitter_window_width: jitter window width.

    Returns:
      jittered_spike_times: array of spike times.
    """
    num_spikes = len(spike_times)

    if num_spikes == 0:
      return spike_times

    spike_times = np.array(spike_times)
    if len(spike_times.shape) > 1:
      raise ValueError('jitter_spike_times_interval: only take 1-D array.')

    # Platkiewicz et al. use half window size.
    # jitter_window_width = jitter_window_width * 2
    jittered_spike_times = np.zeros([num_jitter, num_spikes])
    jitter_noise = jitter_window_width * np.random.rand(num_jitter, num_spikes)
    jittered_spike_times = spike_times - jitter_window_width/2 + jitter_noise;

    # This step is very important as it impacts the synchrony detection.
    # The default sort is in ASCENDNG order.
    jittered_spike_times.sort(axis=1)

    if num_jitter == 1:
      jittered_spike_times = jittered_spike_times[0]

    if verbose:
      plt.figure(figsize=[8, 1])
      plt.plot(spike_times, np.zeros(num_spikes)+2, 'ks', ms=2)

      if num_jitter == 1:
        plt.plot(jittered_spike_times, np.zeros(num_spikes)-1, 'gs', ms=2)
      else:
        for row in range(num_jitter):
          plt.plot(jittered_spike_times[row], np.zeros(num_spikes)-row, 'gs', ms=2)
      plt.xlim(left=0)

    return jittered_spike_times


  @classmethod
  def jitter_spike_times_basic(
      cls,
      spike_times,
      jitter_window_width=0.02,
      num_jitter=1,
      verbose=False):
    """Jitter spike (as opposed to binned spike train) centered at each spike.

    Platkiewicz Stark Amarasingham 2017 - Spike-centered jitter can mistake
    temporal structure.

    Args:
      spike_times: 1-D array of spike times.
      jitter_window_width: jitter window width.

    Returns:
      jittered_spike_times: array of spike times.
    """
    # Empty train.
    if len(spike_times) == 0:
      return spike_times

    # Multiple spike trains.
    if isinstance(spike_times[0], list) or isinstance(spike_times[0], np.ndarray):
      num_trials = len(spike_times)
      jittered_spike_times = [np.zeros([num_jitter, len(spike_times[r])])
          for r in range(num_trials)]

      for r in range(num_trials):
        jittered_spike_times[r] = cls.jitter_spike_times_basic_single(
            spike_times[r], jitter_window_width, num_jitter, verbose)
      return jittered_spike_times

    # Single spike train.
    else:
      return cls.jitter_spike_times_basic_single(
          spike_times, jitter_window_width, num_jitter, verbose)


  @classmethod
  def jitter_spike_times_interval_single(
      cls,
      spike_times,
      jitter_window_width,
      num_jitter=1,
      verbose=False):
    """Jitter spike times (as opposed to binned spike train) in intervals.
    This is for only one spike train.

    Platkiewicz Stark Amarasingham 2017 - Spike-centered jitter can mistake
    temporal structure.

    Examples:
    x = [0.511]
    y = jittertool.jitter_spike_times_interval(x, 0.1)
    print(x); print(y)

    Args:
      spike_times: 1-D array of spike times.
      jitter_window_width: jitter window width.

    Returns:
      jittered_spike_times: array of spike times.
    """
    num_spikes = len(spike_times)

    if num_spikes == 0:
      return spike_times

    spike_times = np.array(spike_times)
    if len(spike_times.shape) > 1:
      raise ValueError('jitter_spike_times_interval: only take 1-D array.')

    # Randomly assign the window.
    # jitter_window_width = np.random.exponential(
    #     scale=jitter_window_width, size=(num_jitter,1))
    # Platkiewicz et al. use half window size.
    # jitter_window_width = jitter_window_width * 2

    jittered_spike_times = np.zeros([num_jitter, num_spikes])
    bin_index = np.floor(spike_times / jitter_window_width)
    floored_spike_times = jitter_window_width * bin_index
    jitter_noise = jitter_window_width * np.random.rand(num_jitter, num_spikes)
    jittered_spike_times = floored_spike_times + jitter_noise;

    # This step is very important as it impacts the synchrony detection.
    # The default sort is in ASCENDNG order.
    jittered_spike_times.sort(axis=1)

    if num_jitter == 1:
      jittered_spike_times = jittered_spike_times[0]

    if verbose:
      plt.figure(figsize=[8, 1])
      plt.plot(spike_times, np.zeros(num_spikes)+2, 'ks', ms=2)

      if num_jitter == 1:
        plt.plot(jittered_spike_times, np.zeros(num_spikes)-1, 'gs', ms=2)
      else:
        for row in range(num_jitter):
          plt.plot(jittered_spike_times[row], np.zeros(num_spikes)-row, 'gs', ms=2)
      plt.xlim(left=0)

    return jittered_spike_times


  @classmethod
  def jitter_spike_times_interval(
      cls,
      spike_times,
      jitter_window_width,
      num_jitter=1,
      data_dim=None,
      verbose=False):
    """Jitter spike times (as opposed to binned spike train) in intervals.

    Platkiewicz Stark Amarasingham 2017 - Spike-centered jitter can mistake
    temporal structure.

    Examples:
    x = [0.511]
    y = jittertool.jitter_spike_times_interval(x, 0.1)
    print(x); print(y)

    Args:
      spike_times: 1-D array of spike times.
      jitter_window_width: jitter window width.

    Returns:
      jittered_spike_times: array of spike times.
    """
    if len(spike_times) == 0:
      return spike_times

    # 2-D.
    if ((data_dim is None or data_dim == 2) and
        (isinstance(spike_times[0], list) or
         isinstance(spike_times[0], np.ndarray))):
      num_trials = len(spike_times)
      jittered_spike_times = [np.zeros([num_jitter, len(spike_times[r])])
          for r in range(num_trials)]

      for r in range(num_trials):
        jittered_spike_times[r] = cls.jitter_spike_times_interval_single(
            spike_times[r], jitter_window_width, num_jitter, verbose)
      return jittered_spike_times

    # 1-D.
    elif data_dim is None or data_dim == 1:
      return cls.jitter_spike_times_interval_single(
          spike_times, jitter_window_width, num_jitter, verbose)

    # 3-D for multivariate regression.
    elif data_dim == 3 and num_jitter == 1:
      num_nodes = len(spike_times)
      jittered_spike_times = [[] for _ in range(num_nodes)]
      for n in range(num_nodes):
        node_spike_times = spike_times[n]
        num_trials = len(node_spike_times)
        jittered_node_spike_times = [np.zeros(len(node_spike_times[r]))
                                     for r in range(num_trials)]
        for r in range(num_trials):
          jittered_node_spike_times[r] = cls.jitter_spike_times_interval_single(
              node_spike_times[r], jitter_window_width, 1, verbose)

        jittered_spike_times[n] = jittered_node_spike_times

      return jittered_spike_times


  @classmethod
  def spike_times_synchrony(
      cls,
      spike_times_x,
      spike_times_y,
      synchrony_window=0.03,
      synchrony_range=[0, 9999999],
      synchrony_type='spike_centered',
      input_sorted=True):
    """Synchrony between two spike times arrays.

    Platkiewicz Stark Amarasingham 2017 - Spike-centered jitter can mistake
    temporal structure.

    Examples:
    x = [0.1, 0.4]
    y = [0.1, 0.2, 0.3, 0.399, 0.4, 0.401, 0.402]
    z = jittertool.spike_times_synchrony(x, y, 0.02)
    Output: 5

    x = [[0.1, 0.4],
         [0.102, 0.401]]
    y = [[0.2, 0.3, 0.399, 0.4, 0.401, 0.402],
         [0.2, 0.3, 0.399, 0.4, 0.401, 0.402]]
    z = jittertool.spike_times_synchrony(x, y, 0.02)
    Output: [4, 4]

    Args:
      spike_times_x: The method takes multiple trials. It can be either,
          1 vs 1, N vs 1, 1 vs N, or N vs N. Other formats are not accepted.
      spike_times_y: It's important to note that `spike_times_x` and
          `spike_times_y` both have to be sorted in ascending order.
      input_sorted: This affects the algorithm efficiency. It's better to sort
          the spike trains first.
    """
    spike_times_x = np.array(spike_times_x)
    spike_times_y = np.array(spike_times_y)

    # Trial 1 vs 1.
    if len(spike_times_x.shape) == 1 and len(spike_times_y.shape) == 1:
      sync_cnt = cls._spike_times_synchrony_single_trial(
          spike_times_x, spike_times_y,
          synchrony_window, synchrony_range, synchrony_type, input_sorted)
      return sync_cnt

    # Trial N vs N.
    if (len(spike_times_x.shape) == 2 and len(spike_times_y.shape) == 2 and
        spike_times_x.shape[0] == spike_times_y.shape[0]):
      num_trials = spike_times_x.shape[0]
      sync_cnt = np.zeros(num_trials)
      for r in range(num_trials):
        sync_cnt[r] = cls._spike_times_synchrony_single_trial(
            spike_times_x[r], spike_times_y[r],
            synchrony_window, synchrony_range, synchrony_type, input_sorted)
      return sync_cnt

    # Trial 1 vs N.
    if (len(spike_times_x.shape) == 1 and len(spike_times_y.shape) == 2):
      num_trials = spike_times_y.shape[0]
      sync_cnt = np.zeros(num_trials)
      for r in range(num_trials):
        sync_cnt[r] = cls._spike_times_synchrony_single_trial(
            spike_times_x, spike_times_y[r],
            synchrony_window, synchrony_range, synchrony_type, input_sorted)
      return sync_cnt

    # Trial N vs 1.
    if (len(spike_times_x.shape) == 2 and len(spike_times_y.shape) == 1):
      num_trials = spike_times_x.shape[0]
      sync_cnt = np.zeros(num_trials)
      for r in range(num_trials):
        sync_cnt[r] = cls._spike_times_synchrony_single_trial(
            spike_times_x[r], spike_times_y,
            synchrony_window, synchrony_range, synchrony_type, input_sorted)
      return sync_cnt

    # Otherwise, the function does not take other formats like M vs N trials.
    raise ValueError(f'Input shape:{spike_times_x.shape, spike_times_y.shape}' +
        ' Only take trials 1 vs 1, N vs 1, 1 vs N, or N vs N.')


  @classmethod
  def _spike_times_synchrony_single_trial(
      cls,
      spike_times_x,
      spike_times_y,
      synchrony_bin,
      synchrony_range=[0, 100],
      synchrony_type='spike_centered',
      input_sorted=True):
    """Detect synchrony between single pair of spikes.

    Args:
      synchrony_type: 'spike_centered', 'binned'.
    """
    if len(spike_times_x.shape) != 1 or len(spike_times_y.shape) != 1:
      raise ValueError('Only takes single trial.')

    if synchrony_type == 'spike_centered':
      sync_cnt = 0
      y_start_id = 0
      for x_spike in spike_times_x:
        if x_spike < synchrony_range[0] or x_spike > synchrony_range[1]:
          continue
        # sync_cnt += sum((x_spike - synchrony_bin <= spike_times_y) &
        #             (spike_times_y <= x_spike + synchrony_bin))

        for y_id in range(y_start_id, len(spike_times_y)):
          if (x_spike - synchrony_bin <= spike_times_y[y_id] <=
                  x_spike + synchrony_bin):
            sync_cnt += 1

          # Note that the inputs have to be sorted if we use the following.
          # If the y-spike is on the left of the x_i-interval, the next round
          # search of x_(i+1)-spike should at least start from y_start_id+1 since
          # x_(i+1) > x_(i).
          elif input_sorted and spike_times_y[y_id] < x_spike - synchrony_bin:
            y_start_id += 1

          # If y+i is on the right side of the x_i interval, then no need to
          # continue as y_i+1 > y_i.
          elif input_sorted and x_spike + synchrony_bin < spike_times_y[y_id]:
            break

      return sync_cnt

    elif synchrony_type == 'binned':
      bins = np.arange(synchrony_range[0], synchrony_range[1]+synchrony_bin,
                       synchrony_bin)
      spike_bin_x = np.histogram(spike_times_x, bins=bins)[0]
      spike_bin_y = np.histogram(spike_times_y, bins=bins)[0]
      sync_cnt = np.sum(spike_bin_x * spike_bin_y)
      return sync_cnt


  @classmethod
  def spike_times_synchrony_p_value(
      cls,
      spike_times_x,
      spike_times_y,
      jitter_type,
      jitter_window_width,
      synchrony_window,
      synchrony_range=[0, 99999],
      synchrony_type='spike_centered',
      num_jitter=500):
    """Calculate the p-value of the synchrony.

    Platkiewicz Stark Amarasingham 2017 - Spike-centered jitter can mistake
    temporal structure.

    Args:
      jitter_func: 'interval', 'basic'.
    """
    if jitter_type is None or jitter_type == 'interval':
      jitter_func = cls.jitter_spike_times_interval
    elif jitter_type == 'basic':
      jitter_func = cls.jitter_spike_times_basic
      # print('basic')

    synchrony_raw = cls.spike_times_synchrony(spike_times_x, spike_times_y,
        synchrony_window=synchrony_window, synchrony_range=synchrony_range,
        synchrony_type=synchrony_type, input_sorted=True)
    # Random correction for synchrony being discrete.
    synchrony_raw_rnd = synchrony_raw + np.random.rand() * 0.5

    spike_times_x_surrogate = jitter_func(spike_times_x,
        jitter_window_width=jitter_window_width, num_jitter=num_jitter,
        verbose=False)
    # Jitter both side.
    # spike_times_y_surrogate = jitter_func(spike_times_y,
    #     jitter_window_width, num_jitter, verbose=False)
    # synchrony_surrogate = cls.spike_times_synchrony(
    #     spike_times_x_surrogate, spike_times_y_surrogate, input_sorted=True)

    synchrony_surrogate = cls.spike_times_synchrony(
        spike_times_x_surrogate, spike_times_y,
        synchrony_window=synchrony_window, synchrony_range=synchrony_range,
        synchrony_type=synchrony_type, input_sorted=True)
    # Random correction for synchrony being discrete.
    synchrony_surrogate_rnd = synchrony_surrogate + np.random.rand(num_jitter) * 0.5

    p_val = sum(synchrony_surrogate >= synchrony_raw) + 1
    p_val = p_val / (num_jitter + 1)
    p_val_rnd = sum(synchrony_surrogate_rnd >= synchrony_raw_rnd) + 1
    p_val_rnd = p_val_rnd / (num_jitter + 1)
    return p_val, p_val_rnd


  @classmethod
  def p_val_poisson_platkiewicz(
      cls,
      n,
      lmbd):
    """Approximate the p-value using Poisson."""
    p_val = 1 - scipy.stats.poisson.cdf(n-1, lmbd)
    return p_val


  @classmethod
  def p_val_poisson_rnd_platkiewicz(
      cls,
      n,
      lmbd):
    """Approximate the p-value using Poisson."""
    p_val = 1 - scipy.stats.poisson.cdf(n-1, lmbd)
    p_val = p_val - np.random.rand() * scipy.stats.poisson.pmf(n, lmbd)
    return p_val


  @classmethod
  def p_val_binomial_platkiewicz(
      cls,
      n,
      N,
      p):
    """Approximate the p-value using Poisson."""
    p_val = 1 - scipy.stats.binom.cdf(n-1, N, p)
    return p_val


  @classmethod
  def p_val_binomial_rnd_platkiewicz(
      cls,
      n,
      N,
      p):
    """Approximate the p-value using Poisson."""
    p_val = 1 - scipy.stats.binom.cdf(n-1, N, p)
    p_val = p_val - np.random.rand() * scipy.stats.binom.pmf(n, N, p)
    return p_val


  @classmethod
  def joint_poisson_mu_CI_pval(
      cls,
      m,
      delta,
      n,
      raw_stat,
      ci_alpha):
    """Joint pmfs together using Poisson approximation.

    Args:
      m, n are spike counts in the JITTER window not the spike train window.
    """
    mu = np.dot(m, n) / delta
    CI_down = scipy.stats.poisson.ppf(ci_alpha/2, mu)
    CI_up = scipy.stats.poisson.ppf(1-ci_alpha/2, mu)
    # P(x >= xorr)
    p_val = 1 - scipy.stats.poisson.cdf(raw_stat-1, mu)
    # P(x >= xorr+1) + u * P(x = xorr)
    p_val_rnd = (1 - scipy.stats.poisson.cdf(raw_stat, mu) +
        scipy.stats.poisson.pmf(raw_stat, mu) * np.random.rand())
    return CI_down, CI_up, mu, p_val, p_val_rnd


  @classmethod
  def joint_normal_mu_CI_pval(
      cls,
      m,
      delta,
      n,
      raw_stat,
      ci_alpha):
    """Joint pmfs together using Poisson approximation.

    Args:
      m, n are spike counts in the JITTER window not the spike train window.
    """
    # The mean and variance are estimated assuming each jitter bin follows the
    # binomial distribution. The Normal mean and variance are derived from that
    # directly.
    p = m / delta
    mean = np.dot(p, n)
    var = np.dot(p * (1-p), n)
    std = np.sqrt(var)
    CI_down = scipy.stats.norm.ppf(ci_alpha/2, loc=mean, scale=std)
    CI_up = scipy.stats.norm.ppf(1-ci_alpha/2, loc=mean, scale=std)
    p_val = 1 - scipy.stats.norm.cdf(raw_stat, loc=mean, scale=std)
    return CI_down, CI_up, mean, p_val, p_val


  @classmethod
  def joint_poisson_pmf(
      cls,
      m,
      delta,
      n,
      verbose=False):
    """Joint pmfs together using Poisson approximation."""
    mu = np.dot(m, n) / delta
    pmf_x = np.arange(m.sum() + n.sum())
    joint_pmf = scipy.stats.poisson.pmf(pmf_x, mu)

    if verbose:
      plt.figure(figsize=[8,2])
      plt.plot(joint_pmf, 'k')
      plt.xlim(mu-100, mu+100)

    epsilon = 1e-12
    valid_index = np.where(joint_pmf > epsilon)[0]
    joint_pmf = joint_pmf[valid_index]
    joint_pmf = joint_pmf / joint_pmf.sum()
    pmf_x = pmf_x[valid_index]

    return joint_pmf, pmf_x


  @classmethod
  def joint_normal_pmf(
      cls,
      m,
      delta,
      n,
      verbose=False):
    """Joint pmfs together using Poisson approximation."""
    p = m / delta
    mean = np.dot(p, n)
    var = np.dot(p * (1-p), n)
    std = np.sqrt(var)
    pmf_x = np.arange(m.sum() + n.sum())
    # The pmf is the integral over the pmf_x interval, the width is 1.
    joint_pmf = scipy.stats.norm.pdf(pmf_x, loc=mean, scale=std) * 1

    if verbose:
      plt.figure(figsize=[8,2])
      plt.plot(joint_pmf, 'k')
      plt.xlim(mean-100, mean+100)

    epsilon = 1e-12
    valid_index = np.where(joint_pmf > epsilon)[0]
    joint_pmf = joint_pmf[valid_index]
    joint_pmf = joint_pmf / joint_pmf.sum()
    pmf_x = pmf_x[valid_index]

    return joint_pmf, pmf_x


  @classmethod
  def joint_binom_pmf(
      cls,
      m,
      delta,
      n,
      verbose=False):
    """Joint pmfs together using convolution."""
    joint_pmf = np.ones(1)
    for i in range(len(m)):
      if m[i] ==0 or n[i] == 0:
        continue
      pmf = cls.binom_pmf_single(m[i], delta, n[i])
      joint_pmf = np.convolve(joint_pmf, pmf)

    pmf_x = np.arange(len(joint_pmf))

    if verbose:
      peak = np.dot(m / delta, n)
      plt.figure(figsize=[8,2])
      plt.plot(joint_pmf, 'k')
      plt.xlim(peak-100, peak+100)

    epsilon = 1e-12
    # Keep the session in the beginning, remove the right tail.
    valid_index = np.where(joint_pmf > epsilon)[0][-1]
    joint_pmf = joint_pmf[:valid_index]
    joint_pmf = joint_pmf / joint_pmf.sum()
    pmf_x = pmf_x[:valid_index]

    return joint_pmf, pmf_x


  @classmethod
  def binom_pmf(
      cls,
      m,
      delta,
      n):
    """Get the pmf of a binomial distribution."""
    if len(m) == len(n) and len(m) == 1:
      return cls.binom_pmf_single(m, delta, n)
    elif len(m) == len(n) and len(m) > 1:
      pmfs = [np.ones(1) for _ in range(len(m))]
      for t in range(len(m)):
        pmfs[t] = cls.binom_pmf_single(m[t], delta, n[t])
      return pmfs


  @classmethod
  @functools.lru_cache(maxsize=None)
  def binom_pmf_single(
      cls,
      m,
      delta,
      n):
    """Get the pmf of a binomial distribution.

    ASSUMPTION: We assume that in the spike train m, the spikes are binary,
        meaning that each bin has at most 1 spike. Even this assumption is not
        true in some case when the firing rate is very high, 

    Args:
      In order to cache the data well. The function stores integers.
    """
    if m == 0 or n == 0:
      return np.ones(1)

    p = np.clip(m / delta, a_min=0, a_max=1)
    cnt = np.arange(n+1)
    pmf = scipy.stats.binom.pmf(cnt, n, p)

    return pmf


  @classmethod
  @functools.lru_cache(maxsize=None)
  def binaried_pmf_single(
      cls,
      m,
      delta,
      n):
    """Get the pmf of a binomial distribution.

    ASSUMPTION: We assume that in the spike train m, the spikes are binary,
        meaning that each bin has at most 1 spike. Even this assumption is not
        true in some case when the firing rate is very high.

    Examples:
    f(2, 4, 2): P(k=0) = 4/16, P(k=1) = (8 + 2)/16, P(k=2) = 2/16
    f(2, 5, 2): P(k=0) = 9/25, P(k=1) = (2x3x2 + 2)/25, P(k=2) = 2 / 25
    f(2, 4, 3): P(k=0) = 8/64, P(k=1) = (3x2x4+3x2x2+2)/64, P(k=2) = (12+6)/64
    f(3, 4, 2): P(k=0) = 1/16, P(k=1) = (2x3x1+3)/16, P(k=2) = 6/16
    """
    if m == 0 or n == 0:
      return np.ones(1)
    if m >= delta:
      pmf = np.zeros(n+1)
      pmf[n] = 1

    overlap_cnt_max = min(m, n) + 1
    pmf = np.zeros(overlap_cnt_max)

    for k in range(overlap_cnt_max):
      if k == 0:
        pmf[k] = (delta - m) ** n
        continue

      cnt_c_sum = 0
      for c in range(k, n+1):
        comb_n_c = scipy.special.comb(n, c)
        cnt_non_overlap = (delta - m) ** (n-c)
        cnt_overlap = (scipy.special.comb(m, k) * math.factorial(k) *
                       cls.num_set_nonempty_partitions(c, k))
        cnt_c = comb_n_c * cnt_non_overlap * cnt_overlap
        cnt_c_sum += cnt_c

      pmf[k] = cnt_c_sum

    pmf = pmf / (delta) ** n
    return pmf


  @classmethod
  @functools.lru_cache(maxsize=None)
  def num_set_nonempty_partitions(
      cls,
      n,
      k):
    """Number of partitions of a set with size n into k non-empty partitions.

    This is a recursive function based on the relation:
    f(n, k) = f(n-1, k-1) + k*f(n, k-1)
    The bases cases are:
    f(n, k) = 0, if n = 0, or k = 0, or k > n.
    f(n, k) = 1, if k = 1, or k = n.

    Test cases:
    f(2, 1) = 1
    f(3, 1) = 1
    f(3, 3) = 1
    f(3, 2) = 3
    f(4, 2) = 4 + 3 = 7 = f(3, 1) + 2*f(3, 2)
    f(4, 3) = 4C2 = 6 = f(3, 2) + 3*f(3, 3)
    f(5, 3) = #1-1-3 + #1-2-2 = 5C2 + 5*4C2 / 2 = 25
    f(5, 3) = f(4, 2) + 3 * f(4, 3) = 25
    """
    if n == 0 or k == 0 or k > n:
      return 0
    elif n == k:
      return 1
    else:
      return (cls.num_set_nonempty_partitions(n-1, k-1) +
              k * cls.num_set_nonempty_partitions(n-1, k))


  @classmethod
  def pmf_CI_mean_pval(
      cls,
      x,
      pmf,
      raw_stat,
      ci_alpha=0.05):
    """CI and mean of a pmf."""
    # epsilon = 1e-12
    # valid_index = np.where(pmf > epsilon)[0]
    # pmf = pmf[valid_index]
    # pmf = pmf / pmf.sum()
    # x = x[valid_index]

    cdf = np.cumsum(pmf)
    CI_quantile_left = ci_alpha / 2
    CI_quantile_right = 1 - CI_quantile_left
    left_index = np.where(cdf >= CI_quantile_left)[0][0]
    right_index = np.where(cdf >= CI_quantile_right)[0][0]
    CI_left = x[left_index]
    CI_right = x[right_index]
    mean = np.dot(x, pmf)

    # Platkiewicz et. al. 2017, random correction.
    if raw_stat > max(x):
      p_val, p_val_rnd = 0, 0
    else:
      raw_index = np.where(x >= raw_stat)[0][0]
      p_val = pmf[raw_index:].sum()
      p_val_rnd = np.random.rand() * pmf[raw_index] + pmf[raw_index+1:].sum()

    return CI_left, CI_right, mean, p_val, p_val_rnd


  @classmethod
  def mc_cross_correlation_mean_CI_pval(
      cls,
      lags,
      xorr_raw,
      xorr_jitter,
      ci_alpha):
    """Obtain mean, CI, p-values from MC jitter outcomes."""
    num_lags = len(lags)
    num_jitter = xorr_jitter.shape[0]

    xorr_CI_up = np.quantile(xorr_jitter, 1-ci_alpha/2, axis=0)
    xorr_CI_down = np.quantile(xorr_jitter, ci_alpha/2, axis=0)
    xorr_mean = np.mean(xorr_jitter, axis=0)

    # p-values using the distribution.
    p_vals = np.ones(num_lags)
    p_vals_rnd = np.ones(num_lags)
    for t in range(num_lags):
      p_val = (sum(xorr_jitter[:,t] > xorr_raw[t]) + 1) / (num_jitter + 1)
      p_vals[t] = p_val
      # Platkiewicz et. al. 2017. Random correction.
      xorr_raw_rnd = xorr_raw[t] + np.random.rand() - 0.5
      xorr_jitter_rnd = xorr_jitter[:,t] + np.random.rand(num_jitter) - 0.5
      p_val_rnd = (sum(xorr_jitter_rnd > xorr_raw_rnd) + 1) / (num_jitter + 1)
      p_vals_rnd[t] = p_val_rnd

    return xorr_mean, xorr_CI_down, xorr_CI_up, p_vals, p_vals_rnd


  @classmethod
  def mc_cross_correlation_mean_CI_pval_uniform(
      cls,
      lags,
      xorr_raw,
      xorr_jitter,
      ci_alpha):
    """Obtain mean, CI, p-values from MC jitter outcomes.

    Reference: Harris Amarasingham 2011.
    """
    num_lags = len(lags)
    num_jitter = xorr_jitter.shape[0]
    # Remove largest values for robustness.
    xorr_jitter_sorted = np.sort(xorr_jitter, axis=0)
    # Normalization.
    # xorr_jitter_sorted = xorr_jitter_sorted[:-1]
    xorr_jitter_mean = np.mean(xorr_jitter_sorted, axis=0)
    xorr_jitter_std = np.std(xorr_jitter_sorted, axis=0)
    xorr_jitter_normed = (xorr_jitter - xorr_jitter_mean) / xorr_jitter_std
    # Maximums and minimums across lags.
    maximums = np.max(xorr_jitter_normed, axis=1)
    minimums = np.min(xorr_jitter_normed, axis=1)
    max_quantile = np.quantile(maximums, q=1-ci_alpha/2)
    min_quantile = np.quantile(minimums, q=ci_alpha/2)
    # Un-normalization.
    xorr_CI_up = max_quantile * xorr_jitter_std + xorr_jitter_mean
    xorr_CI_down = min_quantile * xorr_jitter_std + xorr_jitter_mean
    # p-val.
    xorr_raw_normed = (xorr_raw - xorr_jitter_mean) / xorr_jitter_std
    p_val_up = np.sum(np.max(xorr_raw_normed) < maximums) / num_jitter
    p_val_down = np.sum(np.min(xorr_raw_normed) > minimums) / num_jitter

    return xorr_CI_down, xorr_CI_up, p_val_up, p_val_down


  @classmethod
  def cross_correlation_jitter_mean(
      cls,
      time_bins,
      spike_hist_x,
      spike_hist_y,
      lag_range,
      kernel_width,
      kernel_type='square',
      smooth_type='bin',
      verbose=False):
    """Cross-correlogram mean approximation.

    X = input * kernel. Then do the same as the convolution.
    Here are two equivalent definitions.
    C(tau) = sum_t X(t-tau) Y(t)
    C(tau) = sum_t X(t) Y(t+tau)

    This means, if lag=3, then the pair is the following.
    x:        1,2,3,4,5,6,7,8
              | | | | |
    y:  1,2,3,4,5,6,7,8

    Args:
      spike_hist_x: The shape should be (..., num_bins)
      spike_hist_y: The shape should be (..., num_bins)
      smooth_type: 'bin', 'convolve'
    """
    spike_hist_x = np.array(spike_hist_x)
    spike_hist_y = np.array(spike_hist_y)

    if spike_hist_x.shape[-1] != spike_hist_y.shape[-1]:
      raise ValueError('Inputs x, y shapes do not math.')

    if len(spike_hist_x.shape) == 1 and len(spike_hist_y.shape) == 1:
      num_trials = 1
      dimension = '1-1'
    elif len(spike_hist_x.shape) == 2 and len(spike_hist_y.shape) == 2:
      num_trials = spike_hist_x.shape[0]
      if num_trials != spike_hist_y.shape[0]:
        raise ValueError('Inputs x, y num_trials do not math.')
      dimension = '2-2'

    bin_width = time_bins[1] - time_bins[0]
    lag_index_left = np.ceil(lag_range[0] / bin_width).astype(int)
    lag_index_right = np.floor(lag_range[1] / bin_width).astype(int)
    lag_index_range = np.arange(lag_index_left, lag_index_right+1)
    lags = lag_index_range * bin_width
    num_lags = len(lag_index_range)
    num_bins = len(time_bins)

    if dimension == '1-1':
      if kernel_type == 'square':
        num_bins_kernel = int(kernel_width / bin_width)
        kernel = np.ones(num_bins_kernel) / num_bins_kernel
      elif kernel_type == 'triangle':
        pass
      if smooth_type == 'convolve':
      # Convolved spike trains.
        spike_hist_x_conv = np.convolve(spike_hist_x, kernel, mode='same')
      elif smooth_type == 'bin':
      # Bin-averag trains.
        num_bins_kernel = int(kernel_width / bin_width)
        left_intervals = np.arange(0, num_bins, num_bins_kernel)
        interval_cnt = np.add.reduceat(spike_hist_x, left_intervals)
        spike_hist_x_reduced = interval_cnt / num_bins_kernel
        spike_hist_x_conv = np.kron(spike_hist_x_reduced, np.ones(num_bins_kernel))
      cross_correlogram = np.zeros(num_lags)
      for i, lag in enumerate(lag_index_range):
        if lag > 0:
          cross_correlogram[i] = np.dot(spike_hist_x_conv[:-lag], spike_hist_y[lag:])
        elif lag < 0:
          cross_correlogram[i] = np.dot(spike_hist_x_conv[-lag:], spike_hist_y[:lag])
        else:
          cross_correlogram[i] = np.dot(spike_hist_x_conv, spike_hist_y)

    elif dimension == '2-2':
      if smooth_type == 'convolve':
      # Convolved spike trains.
        if kernel_type == 'square':
          num_bins_kernel = int(kernel_width / bin_width)
          kernel = np.ones([1, num_bins_kernel]) / num_bins_kernel
        elif kernel_type == 'triangle':
          pass
        spike_hist_x_conv = scipy.ndimage.convolve(spike_hist_x, kernel)
      elif smooth_type == 'bin':
      # Bin-averag trains.
        num_bins_kernel = int(kernel_width / bin_width)
        left_intervals = np.arange(0, num_bins, num_bins_kernel)
        interval_cnt = np.add.reduceat(spike_hist_x, left_intervals, axis=1)
        spike_hist_x_reduced = interval_cnt / num_bins_kernel
        spike_hist_x_conv = np.kron(spike_hist_x_reduced, np.ones([1,num_bins_kernel]))
      cross_correlogram = np.zeros((num_trials, num_lags))
      for i, lag in enumerate(lag_index_range):
        if lag > 0:
          cross_correlogram[:,i] = np.einsum('ij,ij->i',
              spike_hist_x_conv[:, :-lag], spike_hist_y[:, lag:])
        elif lag < 0:
          cross_correlogram[:,i] = np.einsum('ij,ij->i',
              spike_hist_x_conv[:, -lag:], spike_hist_y[:, :lag])
        else:
          cross_correlogram[:,i] = np.einsum('ij,ij->i',
              spike_hist_x_conv, spike_hist_y)
      cross_correlogram = np.sum(cross_correlogram, axis=0)

    if verbose:
      plt.figure(figsize=[5, 2])
      plt.plot(lags * 1000, cross_correlogram, '.-', color='grey')
      plt.xlabel('lag [ms]')
      plt.ylabel('xorr-correlogram')
      plt.axvline(x=0, linewidth=0.4, color='darkgrey')
      plt.show()

    return cross_correlogram, lags


  @classmethod
  def cross_correlation(
      cls,
      time_bins,
      spike_hist_x,
      spike_hist_y,
      lag_range,
      verbose=False):
    """Cross-correlogram.
    
    Here are two equivalent definitions.
    C(tau) = sum_t X(t-tau) Y(t)
    C(tau) = sum_t X(t) Y(t+tau)

    This means, if lag=3, then the pair is the following.
    x:        1,2,3,4,5,6,7,8
              | | | | |
    y:  1,2,3,4,5,6,7,8

    Args:
      spike_hist_x: The shape should be (..., num_bins)
      spike_hist_y: The shape should be (..., num_bins)
    """
    spike_hist_x = np.array(spike_hist_x)
    spike_hist_y = np.array(spike_hist_y)

    if spike_hist_x.shape[-1] != spike_hist_y.shape[-1]:
      raise ValueError('Inputs x, y shapes do not math.')

    if len(spike_hist_x.shape) == 1 and len(spike_hist_y.shape) == 1:
      num_trials = 1
      dimension = '1-1'
    elif len(spike_hist_x.shape) == 2 and len(spike_hist_y.shape) == 2:
      num_trials = spike_hist_x.shape[0]
      if num_trials != spike_hist_y.shape[0]:
        raise ValueError('Inputs x, y num_trials do not math.')
      dimension = '2-2'

    bin_width = time_bins[1] - time_bins[0]
    lag_index_left = np.ceil(lag_range[0] / bin_width).astype(int)
    lag_index_right = np.floor(lag_range[1] / bin_width).astype(int)
    lag_index_range = np.arange(lag_index_left, lag_index_right+1)
    lags = lag_index_range * bin_width
    num_lags = len(lag_index_range)

    if dimension == '1-1':
      cross_correlogram = np.zeros(num_lags)
      for i, lag in enumerate(lag_index_range):
        if lag > 0:
          cross_correlogram[i] = np.dot(spike_hist_x[:-lag], spike_hist_y[lag:])
        elif lag < 0:
          cross_correlogram[i] = np.dot(spike_hist_x[-lag:], spike_hist_y[:lag])
        else:
          cross_correlogram[i] = np.dot(spike_hist_x, spike_hist_y)

    elif dimension == '2-2':
      cross_correlogram = np.zeros((num_trials, num_lags))
      for i, lag in enumerate(lag_index_range):
        if lag > 0:
          cross_correlogram[:,i] = np.einsum('ij,ij->i',
              spike_hist_x[:, :-lag], spike_hist_y[:, lag:])
        elif lag < 0:
          cross_correlogram[:,i] = np.einsum('ij,ij->i',
              spike_hist_x[:, -lag:], spike_hist_y[:, :lag])
        else:
          cross_correlogram[:,i] = np.einsum('ij,ij->i',
              spike_hist_x, spike_hist_y)
      cross_correlogram = np.sum(cross_correlogram, axis=0)

    if verbose:
      plt.figure(figsize=[5, 2])
      plt.plot(lags * 1000, cross_correlogram, '.-', color='grey')
      plt.xlabel('lag [ms]')
      plt.ylabel('xorr-correlogram')
      plt.axvline(x=0, linewidth=0.4, color='darkgrey')
      plt.show()

    return cross_correlogram, lags


  @classmethod
  def cross_correlation_jitter(
      cls,
      spike_times_x,
      spike_times_y,
      spk_bin_width,
      trial_length,
      lag_range,
      jitter_window_width,
      distribution_type='poisson',
      num_jitter=500,
      ci_alpha=0.05,
      verbose=False,
      file_path=None):
    """Perform jitter based test on cross correlation.

    Args:
      spike_times_x:
      spike_times_y:
      distribution_type:
          'mc', 'mc_sim'. Monte Carlo methods.
          'binom', 'poisson', 'poisson_pmf', 'norm'. Closed-form.
      verbose: 2, two panel plot.
               1, one panel plot.
    """
    # Raw statistic.
    spike_hist_x, bins = cls.bin_spike_times(
        spike_times_x, spk_bin_width, trial_length)
    spike_hist_y, _    = cls.bin_spike_times(
        spike_times_y, spk_bin_width, trial_length)
    xorr_raw, lags = cls.cross_correlation(
        bins, spike_hist_x, spike_hist_y, lag_range, verbose=False)
    num_lags = len(lags)
    num_trials = len(spike_times_x)

    # MC.
    if distribution_type == 'mc':
      xorr_jitter = np.zeros([num_jitter, num_lags])
      trange = tqdm(range(num_jitter)) if verbose else range(num_jitter)
      for r in trange:
        spike_times_surrogate_x = cls.jitter_spike_times_interval(
            spike_times_x, jitter_window_width, num_jitter=1, verbose=False)
        spike_hist_surrogate_x, _ = cls.bin_spike_times(
            spike_times_surrogate_x, spk_bin_width, trial_length)
        xorr_jitter[r], lags = cls.cross_correlation(
            bins, spike_hist_surrogate_x, spike_hist_y, lag_range, verbose=False)
      (xorr_mean, xorr_CI_down, xorr_CI_up, p_vals, p_vals_rnd
          ) = cls.mc_cross_correlation_mean_CI_pval(
          lags, xorr_raw, xorr_jitter, ci_alpha)

    # MC with simultaneous band.
    elif distribution_type == 'mc_sim':
      xorr_jitter = np.zeros([num_jitter, num_lags])
      if verbose:
        if hasattr(tqdm, '_instances'):
          tqdm._instances.clear()
        trange = tqdm(range(num_jitter), ncols=100, file=sys.stdout)
      else:
        trange = range(num_jitter)
      for r in trange:
        spike_times_surrogate_x = cls.jitter_spike_times_interval(
            spike_times_x, jitter_window_width, num_jitter=1, verbose=False)
        spike_hist_surrogate_x, _ = cls.bin_spike_times(
            spike_times_surrogate_x, spk_bin_width, trial_length)
        xorr_jitter[r], lags = cls.cross_correlation(
            bins, spike_hist_surrogate_x, spike_hist_y, lag_range, verbose=False)
      (xorr_mean, xorr_CI_down, xorr_CI_up, p_vals, p_vals_rnd
          ) = cls.mc_cross_correlation_mean_CI_pval(
          lags, xorr_raw, xorr_jitter, ci_alpha)
      (xorr_CI_down_uniform, xorr_CI_up_uniform, p_val_max, p_val_min
          ) = cls.mc_cross_correlation_mean_CI_pval_uniform(
          lags, xorr_raw, xorr_jitter, ci_alpha)

    elif distribution_type in ['binom', 'poisson', 'poisson_pmf', 'normal']:
      delta = int(jitter_window_width / spk_bin_width)
      xorr_CI_down = np.zeros(num_lags)
      xorr_CI_up = np.zeros(num_lags)
      xorr_mean = np.zeros(num_lags)
      p_vals = np.zeros(num_lags)
      p_vals_rnd = np.zeros(num_lags)

      spike_jitter_hist_x, bins = cls.bin_spike_times(
          spike_times_x, jitter_window_width, trial_length)
      spike_jitter_hist_x = spike_jitter_hist_x.reshape(-1)
      spike_times_y_shift = [np.zeros(1) for r in range(num_trials)]
      if verbose:
        trange = enumerate(tqdm(lags, ncols=100, file=sys.stdout))
      else:
        trange = enumerate(lags)
      for t, lag in trange:
        # Shift the y-spike train.
        for r in range(num_trials):
          spike_times_y_shift[r] = np.array(spike_times_y[r]) - lag
        spike_jitter_hist_y, _    = cls.bin_spike_times(
            spike_times_y_shift, jitter_window_width, trial_length)
        spike_jitter_hist_y = spike_jitter_hist_y.reshape(-1)

        if distribution_type == 'binom':
          pmf, pmf_x = cls.joint_binom_pmf(
              spike_jitter_hist_y, delta, spike_jitter_hist_x, verbose=False)
          (xorr_CI_down[t], xorr_CI_up[t], xorr_mean[t], p_vals[t], p_vals_rnd[t]
              ) = cls.pmf_CI_mean_pval(
              pmf_x, pmf, raw_stat=xorr_raw[t], ci_alpha=ci_alpha)

        elif distribution_type == 'poisson_pmf':
          pmf, pmf_x = cls.joint_poisson_pmf(
              spike_jitter_hist_y, delta, spike_jitter_hist_x, verbose=False)
          (xorr_CI_down[t], xorr_CI_up[t], xorr_mean[t], p_vals[t], p_vals_rnd[t]
              ) = cls.pmf_CI_mean_pval(
              pmf_x, pmf, raw_stat=xorr_raw[t], ci_alpha=ci_alpha)

        # No pmf output. CI, p-val are calculated directly.
        elif distribution_type == 'poisson':
          (xorr_CI_down[t], xorr_CI_up[t], xorr_mean[t], p_vals[t], p_vals_rnd[t]
              ) = cls.joint_poisson_mu_CI_pval(
              spike_jitter_hist_y, delta, spike_jitter_hist_x,
              raw_stat=xorr_raw[t], ci_alpha=ci_alpha)

        elif distribution_type == 'normal':
          (xorr_CI_down[t], xorr_CI_up[t], xorr_mean[t], p_vals[t], p_vals_rnd[t]
              ) = cls.joint_normal_mu_CI_pval(
              spike_jitter_hist_y, delta, spike_jitter_hist_x,
              raw_stat=xorr_raw[t], ci_alpha=ci_alpha)

    if verbose == 2:
      gs_kw = dict(width_ratios=[1, 1], height_ratios=[1])
      fig, axs = plt.subplots(figsize=(12, 3), gridspec_kw=gs_kw,
          nrows=1, ncols=2)
      # plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)

      ax = fig.add_subplot(axs[0])
      if distribution_type == 'mc_sim':
        plt.fill_between(lags * 1000, xorr_CI_down_uniform, xorr_CI_up_uniform,
                         facecolor='lightgrey', alpha=0.8,
                         label=f'{(1-ci_alpha) * 100}% simultaneous CI')
        # plt.plot(lags * 1000, xorr_CI_down_uniform, 'r', lw=1)
        # plt.plot(lags * 1000, xorr_CI_up_uniform, 'r', lw=1)
      plt.fill_between(lags * 1000, xorr_CI_down, xorr_CI_up,
                       facecolor='grey', alpha=0.8,
                       label=f'{(1-ci_alpha) * 100}% CI')
      plt.plot(lags * 1000, xorr_mean, c='grey', ls='--')
      plt.plot(lags * 1000, xorr_raw, c='k')
      plt.axvline(x=0, lw=0.4, c='darkgrey')
      plt.xlabel('Lag [ms]')
      plt.ylabel('Cross-correlogram')
      plt.title(f'Jitter window={jitter_window_width * 1000} ms,  ' +
                f'spk bin={spk_bin_width  * 1000} ms')
      # plt.ylim(top=max(xorr_CI_up) + 0.4*(max(xorr_CI_up)-min(xorr_CI_down)))
      plt.legend(loc='lower left', ncol=2)

      ax = fig.add_subplot(axs[1])
      if distribution_type == 'mc_sim':
        plt.fill_between(lags * 1000, xorr_CI_down_uniform - xorr_mean,
                         xorr_CI_up_uniform - xorr_mean,
                         facecolor='lightgrey', alpha=0.8)
        # plt.plot(lags * 1000, xorr_CI_down_uniform - xorr_mean, 'r', lw=1)
        # plt.plot(lags * 1000, xorr_CI_up_uniform - xorr_mean, 'r', lw=1)
      plt.fill_between(lags * 1000, xorr_CI_down - xorr_mean,
                       xorr_CI_up - xorr_mean,
                       facecolor='grey', alpha=0.8)
      plt.plot(lags * 1000, xorr_raw - xorr_mean, c='k')
      plt.axvline(x=0, lw=0.4, c='darkgrey')
      plt.axhline(y=0, lw=0.4, c='darkgrey')
      plt.xlabel('Lag [ms]', fontsize=12)
      plt.ylabel('CCG [spike pairs]', fontsize=12)
      plt.title('Mean corrected')

      if file_path is not None:
        plt.savefig(file_path, bbox_inches='tight')
        print('Save figure to:', file_path)
      plt.show()

    if verbose == 1:
      gs_kw = dict(width_ratios=[1], height_ratios=[1])
      fig, axs = plt.subplots(figsize=(5, 3), gridspec_kw=gs_kw,
          nrows=1, ncols=1)
      ax = fig.add_subplot(axs)
      # if distribution_type == 'mc_sim':  # Simultaneous band only for MC.
      #   plt.fill_between(lags * 1000, xorr_CI_down_uniform - xorr_mean,
      #                    xorr_CI_up_uniform - xorr_mean,
      #                    facecolor='lightgrey', alpha=0.8,
      #                    label=f'{(1-ci_alpha) * 100}% simultaneous CI')
      #   # plt.plot(lags * 1000, xorr_CI_down_uniform - xorr_mean, 'r', lw=1)
      #   # plt.plot(lags * 1000, xorr_CI_up_uniform - xorr_mean, 'r', lw=1)

      plt.fill_between(lags * 1000, xorr_CI_down - xorr_mean,
                       xorr_CI_up - xorr_mean,
                       facecolor='dodgerblue', alpha=0.24,
                       label=f'{(1-ci_alpha) * 100}% CI')

      plt.plot(lags * 1000, xorr_raw - xorr_mean, c='tab:blue')
      # plt.ylim(np.min(xorr_raw - xorr_mean)*1.5, np.max(xorr_raw - xorr_mean)*1.3)
      plt.axvline(x=0, lw=0.4, c='darkgrey')
      plt.axhline(y=0, lw=0.4, c='darkgrey')
      plt.xlabel('Lag [ms]', fontsize=12)
      plt.ylabel('CCG [excess spike pairs]', fontsize=12)
      # plt.title('Mean corrected\n'+
      #           f'Jitter window={jitter_window_width * 1000} ms,  ' +
      #           f'spk bin={spk_bin_width  * 1000} ms')
      # plt.legend(loc=(1.05, 0.5), ncol=1)
      # plt.legend(loc='lower left', ncol=2)
      plt.xlim(lag_range[0]*1000, lag_range[1]*1000)

      if file_path is not None:
        plt.savefig(file_path, bbox_inches='tight')
        print('Save figure to:', file_path)
      plt.show()

    if distribution_type == 'mc_sim':
      return (lags, xorr_raw, xorr_mean,
              xorr_CI_down, xorr_CI_up, p_vals, p_vals_rnd,
              xorr_CI_down_uniform, xorr_CI_up_uniform, p_val_max, p_val_min)
    return lags, xorr_raw, xorr_mean, xorr_CI_down, xorr_CI_up, p_vals, p_vals_rnd


  @classmethod
  def cross_correlation_jitter_analytical_methods_comparison_pointwise(
      cls,
      spike_times_x,
      spike_times_y,
      spk_bin_width,
      trial_length,
      lag,
      jitter_window_width,
      num_jitter=1000,
      ci_alpha=0.05,
      verbose=False):
    """Perform jitter based test on cross correlation.

    Args:
      spike_times_x:
      spike_times_y:
    """
    # Raw statistic.
    lag_range = [lag-spk_bin_width/2, lag+spk_bin_width/2]
    spike_hist_x, bins = cls.bin_spike_times(
        spike_times_x, spk_bin_width, trial_length)
    spike_hist_y, _    = cls.bin_spike_times(
        spike_times_y, spk_bin_width, trial_length)
    xorr_raw, lags = cls.cross_correlation(
        bins, spike_hist_x, spike_hist_y, lag_range, verbose=False)

    # Monte Carlo result.
    num_trials = len(spike_times_x)
    num_lags = len(lags)
    xorr_jitter = np.zeros([num_jitter, num_lags])

    for r in tqdm(range(num_jitter)):
      spike_times_surrogate_x = cls.jitter_spike_times_interval(
          spike_times_x, jitter_window_width, num_jitter=1, verbose=False)
      spike_hist_surrogate_x, _ = cls.bin_spike_times(
          spike_times_surrogate_x, spk_bin_width, trial_length)
      xorr_jitter[r], lags = cls.cross_correlation(
          bins, spike_hist_surrogate_x, spike_hist_y, lag_range, verbose=False)

    lag_index = np.where(lags==lag)[0]
    xorr_jitter_samples = xorr_jitter[:, lag_index]
    mc_CI_left = np.quantile(xorr_jitter_samples, ci_alpha/2)
    mc_CI_right = np.quantile(xorr_jitter_samples, 1-ci_alpha/2)
    mc_mean = np.mean(xorr_jitter_samples)

    # Analytical result.
    for r in range(num_trials):
      spike_times_y[r] = np.array(spike_times_y[r]) - lag

    spike_jitter_hist_x, bins = cls.bin_spike_times(
        spike_times_x, jitter_window_width, trial_length)
    spike_jitter_hist_y, _    = cls.bin_spike_times(
        spike_times_y, jitter_window_width, trial_length)

    spike_jitter_hist_x = spike_jitter_hist_x.reshape(-1)
    spike_jitter_hist_y = spike_jitter_hist_y.reshape(-1)
    delta = int(jitter_window_width / spk_bin_width)
    binom_pmf, binom_pmf_x = cls.joint_binom_pmf(
          spike_jitter_hist_y, delta, spike_jitter_hist_x, verbose=False)
    poisson_pmf, poisson_pmf_x = cls.joint_poisson_pmf(
          spike_jitter_hist_y, delta, spike_jitter_hist_x, verbose=False)
    normal_pmf, normal_pmf_x = cls.joint_normal_pmf(
          spike_jitter_hist_y, delta, spike_jitter_hist_x, verbose=False)

    (binom_CI_left, binom_CI_right, binom_mean, binom_p_val, binom_p_val_rnd
        ) = cls.pmf_CI_mean_pval(
        binom_pmf_x, binom_pmf, raw_stat=xorr_raw, ci_alpha=0.01)
    (poisson_CI_left, poisson_CI_right, poisson_mean, poisson_p_val, poisson_p_val_rnd
        ) = cls.pmf_CI_mean_pval(
        poisson_pmf_x, poisson_pmf, raw_stat=xorr_raw, ci_alpha=0.01)
    (normal_CI_left, normal_CI_right, normal_mean, normal_p_val, normal_p_val_rnd
        ) = cls.joint_normal_mu_CI_pval(
        spike_jitter_hist_y, delta, spike_jitter_hist_x,
        raw_stat=xorr_raw, ci_alpha=0.01)

    print('binom len, poisson len: ', len(binom_pmf), len(poisson_pmf))
    peak_x = np.dot(spike_jitter_hist_y / delta, spike_jitter_hist_x)
    peak_y = np.max(binom_pmf)
    plt.figure(figsize=[8,4])
    # plot - MC.
    seaborn.distplot(xorr_jitter_samples,
                     bins=45, color='green', kde=True, label='MC')
    plt.plot([mc_CI_left, mc_mean, mc_CI_right], [peak_y*1.3]*3,
             'o-', color='g', label=f'MC {(1-ci_alpha)*100}% CI + mean')
    # plot - binom.
    plt.plot(binom_pmf_x, binom_pmf, 'b', label='binomial')
    plt.plot([binom_CI_left, binom_mean, binom_CI_right], [peak_y*1.4]*3,
             'ro-',color='b', label=f'Binomial {(1-ci_alpha)*100}% CI + mean')
    # plot - Normal.
    plt.plot(normal_pmf_x, normal_pmf, 'c', label='Normal')
    plt.plot([normal_CI_left, normal_mean, normal_CI_right], [peak_y*1.5]*3,
             'co-',color='c', label=f'Normal {(1-ci_alpha)*100}% CI + mean')
    # plot - Poisson.
    plt.plot(poisson_pmf_x, poisson_pmf, 'r', label='Poisson')
    plt.plot([poisson_CI_left, poisson_mean, poisson_CI_right], [peak_y*1.6]*3,
             'ro-',color='r', label=f'Poisson {(1-ci_alpha)*100}% CI + mean')
    # plt.axvline(x=xorr_raw, ls='--', c='k', label='test statistic')
    plt.legend()
    plt.title(f'lag={lag*1000} ms')
    plt.xlabel('Cross correlation')
    plt.ylabel('PMF/PDF')
    plt.show()


  @classmethod
  def cross_correlation_jitter_analytical_methods_comparison_band(
      cls,
      spike_times_x,
      spike_times_y,
      spk_bin_width,
      trial_length,
      lag_range,
      jitter_window_width,
      jitter_type,
      kernel_type='square',
      smooth_type='bin',
      num_jitter=1000,
      ci_alpha=0.05,
      verbose=False):
    """Perform jitter based test on cross correlation.

    Args:
      spike_times_x:
      spike_times_y:
    """
    # Raw statistic.
    spike_hist_x, spk_bins = cls.bin_spike_times(
        spike_times_x, spk_bin_width, trial_length)
    spike_hist_y, _    = cls.bin_spike_times(
        spike_times_y, spk_bin_width, trial_length)
    xorr_raw, lags = cls.cross_correlation(
        spk_bins, spike_hist_x, spike_hist_y, lag_range, verbose=False)

    # Monte Carlo result.
    if len(spike_hist_x.shape) == 1 and len(spike_hist_y.shape) == 1:
      num_trials = 1
    elif len(spike_hist_x.shape) == 2 and len(spike_hist_y.shape) == 2:
      num_trials = spike_hist_x.shape[0]
      if num_trials != spike_hist_y.shape[0]:
        raise ValueError('Inputs x, y num_trials do not math.')
    num_lags = len(lags)
    xorr_jitter = np.zeros([num_jitter, num_lags])

    for r in tqdm(range(num_jitter)):
      if jitter_type == 'interval':
        spike_times_surrogate_x = cls.jitter_spike_times_interval(
            spike_times_x, jitter_window_width, num_jitter=1, verbose=False)
      elif jitter_type == 'basic':
        spike_times_surrogate_x = cls.jitter_spike_times_basic(
            spike_times_x, jitter_window_width, num_jitter=1, verbose=False)
      spike_hist_surrogate_x, _ = cls.bin_spike_times(
          spike_times_surrogate_x, spk_bin_width, trial_length)
      xorr_jitter[r], lags = cls.cross_correlation(
          spk_bins, spike_hist_surrogate_x, spike_hist_y, lag_range, verbose=False)

    mc_CI_down = np.quantile(xorr_jitter, 1-ci_alpha/2, axis=0)
    mc_CI_up = np.quantile(xorr_jitter, ci_alpha/2, axis=0)
    mc_mean = np.mean(xorr_jitter, axis=0)

    # Analytical result.
    lag_index_left = np.ceil(lag_range[0] / spk_bin_width).astype(int)
    lag_index_right = np.floor(lag_range[1] / spk_bin_width).astype(int)
    lag_index_range = np.arange(lag_index_left, lag_index_right+1)
    lags = lag_index_range * spk_bin_width
    num_lags = len(lag_index_range)

    delta = int(jitter_window_width / spk_bin_width)
    binom_CI_down = np.zeros(num_lags)
    binom_CI_up = np.zeros(num_lags)
    binom_mean = np.zeros(num_lags)
    poisson_CI_down = np.zeros(num_lags)
    poisson_CI_up = np.zeros(num_lags)
    poisson_mean = np.zeros(num_lags)
    normal_CI_down = np.zeros(num_lags)
    normal_CI_up = np.zeros(num_lags)
    normal_mean = np.zeros(num_lags)
    spike_times_y_shift = [np.zeros(1) for r in range(num_trials)]

    # for t, lag in enumerate(tqdm(lags)):
    for t, lag in enumerate(lags):
      # Shift the y-spike train.
      if num_trials == 1:
        spike_times_y_shift = np.array(spike_times_y) - lag
      elif num_trials > 1:
        for r in range(num_trials):
          spike_times_y_shift[r] = np.array(spike_times_y[r]) - lag

      spike_jitter_hist_x, jitter_bins = cls.bin_spike_times(
          spike_times_x, jitter_window_width, trial_length)
      spike_jitter_hist_y, _    = cls.bin_spike_times(
          spike_times_y_shift, jitter_window_width, trial_length)

      spike_jitter_hist_x = spike_jitter_hist_x.reshape(-1)
      spike_jitter_hist_y = spike_jitter_hist_y.reshape(-1)
      
      binom_pmf, binom_pmf_x = cls.joint_binom_pmf(
            spike_jitter_hist_y, delta, spike_jitter_hist_x, verbose=False)
      poisson_pmf, poisson_pmf_x = cls.joint_poisson_pmf(
            spike_jitter_hist_y, delta, spike_jitter_hist_x, verbose=False)
      normal_pmf, normal_pmf_x = cls.joint_normal_pmf(
          spike_jitter_hist_y, delta, spike_jitter_hist_x, verbose=False)

      (binom_CI_down[t], binom_CI_up[t], binom_mean[t], _, _
          ) = cls.pmf_CI_mean_pval(
          binom_pmf_x, binom_pmf, raw_stat=xorr_raw[t], ci_alpha=ci_alpha)
      (poisson_CI_down[t], poisson_CI_up[t], poisson_mean[t], _, _
          ) = cls.pmf_CI_mean_pval(
          poisson_pmf_x, poisson_pmf, raw_stat=xorr_raw[t], ci_alpha=ci_alpha)
      (normal_CI_down[t], normal_CI_up[t], normal_mean[t], _, _
              ) = cls.joint_normal_mu_CI_pval(
              spike_jitter_hist_y, delta, spike_jitter_hist_x,
              raw_stat=xorr_raw, ci_alpha=0.01)

    # xorr mean convolution approxmation.
    xorr_conv_mean, _ = cls.cross_correlation_jitter_mean(
        spk_bins, spike_hist_x, spike_hist_y, lag_range,
        kernel_width=jitter_window_width, kernel_type=kernel_type,
        smooth_type=smooth_type, verbose=False)

    plt.figure(figsize=[8,4])
    lags_plot = lags * 1000
    plt.plot(lags_plot, xorr_raw, 'k:', label='raw xorr')
    plt.plot(lags_plot, xorr_conv_mean, 'r', label='xorr conv mean')

    # plot - MC.
    plt.plot(lags_plot, mc_CI_down, 'g', label='MC')
    plt.plot(lags_plot, mc_CI_up, 'g')
    plt.plot(lags_plot, mc_mean, 'g--')
    # plot - binom.
    plt.plot(lags_plot, binom_CI_down, 'b', label='Binomial')
    plt.plot(lags_plot, binom_CI_up, 'b')
    plt.plot(lags_plot, binom_mean, 'b--')
    # plot - Poisson.
    plt.plot(lags_plot, poisson_CI_down, 'r', label='Poisson')
    plt.plot(lags_plot, poisson_CI_up, 'r')
    plt.plot(lags_plot, poisson_mean, 'r--')
    # plot - Normal.
    plt.plot(lags_plot, normal_CI_down, 'c', label='Normal')
    plt.plot(lags_plot, normal_CI_up, 'c')
    plt.plot(lags_plot, normal_mean, 'c--')
    plt.xlabel('lag [ms]')
    plt.ylabel('cross correlation')
    plt.legend()
    plt.show()

    plt.figure(figsize=[8,4])
    plt.plot(lags_plot, xorr_raw, 'k:', label='raw xorr')
    plt.plot(lags_plot, mc_mean, 'c--', label='mc')
    plt.plot(lags_plot, binom_mean, 'b--', label='binomial')
    plt.plot(lags_plot, poisson_mean, 'g--', label='poisson')
    plt.plot(lags_plot, xorr_conv_mean, 'r', label='xorr conv mean')
    plt.legend()
    plt.xlabel('lag [ms]')
    plt.ylabel('cross correlation')
    plt.show()


  @classmethod
  def verify_xorr_significance(
      cls,
      lags,
      xorr_raw,
      xorr_mean,
      xorr_CI_down,
      xorr_CI_up,
      detect_threshold=0.1,
      verbose=False):
    """Check if xorr crosses the CI bands."""
    lag_bin_width = np.abs(lags[1] - lags[0])
    diff_up = xorr_raw - xorr_CI_up
    diff_down = xorr_CI_down - xorr_raw

    volume_up = diff_up[diff_up > 0].sum() * lag_bin_width
    volume_down = diff_down[diff_down > 0].sum() * lag_bin_width
    if verbose:
      print(volume_up, volume_down)
    return volume_up > detect_threshold, volume_down > detect_threshold,


  @classmethod
  def extract_simultaneous_pval(
      cls,
      model_par_list,
      verbose=False):
    """Extract saved output list of `cross_correlation_jitter`."""
    num_models = len(model_par_list)
    p_vals_simualtanous_up = np.zeros(num_models)
    p_vals_simualtanous_down = np.zeros(num_models)
    print(f'num_models {num_models}')

    for m, par in enumerate(model_par_list):
      p_vals_simualtanous_up[m] = par[-2]
      p_vals_simualtanous_down[m] = par[-1]
    return p_vals_simualtanous_up, p_vals_simualtanous_down


  @classmethod
  def spike_trains_neg_log_likelihood(
      cls,
      log_lmbd,
      spike_trains):
    """Calculates the log-likelihood of a spike train given log firing rate.

    When it calculates the negative log_likelihood funciton, it assumes that it
    is a function of lambda instead of spikes. So it drops out the terms that
    are not related to the lambda, which is the y! (spikes factorial) term.

    Args:
      log_lmbd: The format can be in two ways.
          timebins 1D array.
          trials x timebins matrix. Different trials have differnet intensity.
              In this case, `spike_trains` and `log_lmbd` have matching rows.
      spike_trains: Trials x timebins matrix.
    """
    num_trials, num_bins = spike_trains.shape
    if num_trials == 0:
      return 0

    log_lmbd = np.array(log_lmbd)
    if len(log_lmbd.shape) == 2:  # Trialwise intensity.
      x, num_bins_log_lmbd = log_lmbd.shape
      if x != num_trials:
        raise ValueError('Number of trials does not match intensity size.')
      if num_bins != num_bins_log_lmbd:
        raise ValueError('The length of log_lmbd should be equal to spikes.')

      # Equivalent to row wise dot product then take the sum.
      nll = - np.sum(spike_trains * log_lmbd)
      nll += np.exp(log_lmbd).sum()
      return nll

    elif len(log_lmbd.shape) == 1:  # Single intensity for all trials.
      num_bins_log_lmbd = len(log_lmbd)
      if num_bins != num_bins_log_lmbd:
        raise ValueError('The length of log_lmbd should be equal to spikes.')
      nll = - np.dot(spike_trains.sum(axis=0), log_lmbd)
      nll += np.exp(log_lmbd).sum() * num_trials
      return nll


  def bivariate_spike_hist_coupling_filter_regression_lrt(
      self,
      spike_times_x,
      spike_times_y,
      trial_length,
      spk_bin_width,
      jitter_window_width,
      jitter_type='interval',
      basis_type='B0B1',
      epsilon=1e-6,
      cache_parameters=True):
    """Coupling filter model.

    In this method, we model the lambda = X beta directly instead of
    log lambda = X beta.
    """
    spike_hist_x, bins = self.bin_spike_times(
        spike_times_x, spk_bin_width, trial_length)
    spike_hist_y, _    = self.bin_spike_times(
        spike_times_y, spk_bin_width, trial_length)
    num_bins = len(bins)
    if len(spike_hist_x.shape) == 1 and len(spike_hist_y.shape) == 1:
      num_trials = 1
      dimension = '1-1'
    elif len(spike_hist_x.shape) == 2 and len(spike_hist_y.shape) == 2:
      num_trials = spike_hist_x.shape[0]
      if num_trials != spike_hist_y.shape[0]:
        raise ValueError('Inputs x, y num_trials do not math.')
      dimension = '2-2'

    # Construct jitter correction basis.
    num_bins_kernel = int(jitter_window_width / spk_bin_width)
    intervals = np.arange(0, num_bins, num_bins_kernel)
    if dimension == '1-1':
      if jitter_type == 'interval':
        interval_cnt = np.add.reduceat(spike_hist_x, left_intervals)
        spike_hist_x_reduced = interval_cnt / num_bins_kernel
        spike_hist_x_conv = np.kron(spike_hist_x_reduced, np.ones(num_bins_kernel))
      elif jitter_type == 'basic':
        kernel = np.ones([1, num_bins_kernel]) / num_bins_kernel
        spike_hist_x_conv = scipy.ndimage.convolve(spike_hist_x, kernel)

    elif dimension == '2-2':
      if jitter_type == 'interval':
        interval_cnt = np.add.reduceat(spike_hist_x, intervals, axis=1)
        spike_hist_x_reduced = interval_cnt / num_bins_kernel
        spike_hist_x_conv = np.kron(spike_hist_x_reduced, np.ones([1,num_bins_kernel]))
      elif jitter_type == 'basic':
        kernel = np.ones([1, num_bins_kernel]) / num_bins_kernel
        spike_hist_x_conv = scipy.ndimage.convolve(spike_hist_x, kernel)

    # Construct data.
    y = spike_hist_y.reshape(-1, 1)
    x = spike_hist_x.reshape(-1)
    x_conv = spike_hist_x_conv.reshape(-1)

    if cache_parameters and hasattr(self, 'beta_0') and hasattr(self, 'beta_1'):
      beta_0 = self.beta_0
      beta_1 = self.beta_1
    else:
      beta_0, beta_1 = None, None

    # H0 hypothesis.
    if basis_type == 'B0B1':
      X = x_conv.reshape(-1, 1)
    elif basis_type == 'CB1':
      X = np.ones((len(x), 1))
    y_hat, beta_0, nll_0 = self.optimize_spike_hist_lambda_linear_model(
        y, X, beta=beta_0, epsilon=epsilon, verbose=0)

    # H1 hypothesis.
    if basis_type == 'B0B1':
      X = np.vstack([x_conv, x]).T
    elif basis_type == 'CB1':
      const = np.ones(len(x))
      X = np.vstack([const, x]).T
    y_hat, beta_1, nll_1 = self.optimize_spike_hist_lambda_linear_model(
        y, X, beta=beta_1, epsilon=epsilon, verbose=0)

    if cache_parameters:
      self.beta_0 = beta_0
      self.beta_1 = beta_1
    log_likelihood_ratio = 2 * (nll_0 - nll_1)
    p_val = 1 - scipy.stats.chi2.cdf(log_likelihood_ratio, df=1)
    return p_val


  @classmethod
  def bivariate_spike_hist_coupling_filter_regression(
      cls,
      spike_times_x,
      spike_times_y,
      trial_length,
      spk_bin_width,
      jitter_window_width,
      jitter_type='interval',
      basis_type='B1',
      epsilon=1e-6,
      link_func='linear',
      verbose=False):
    """Coupling filter model.

    In this method, we model the lambda = X beta directly instead of
    log lambda = X beta.
    """
    spike_hist_x, bins = cls.bin_spike_times(
        spike_times_x, spk_bin_width, trial_length)
    spike_hist_y, _    = cls.bin_spike_times(
        spike_times_y, spk_bin_width, trial_length)
    num_bins = len(bins)
    if len(spike_hist_x.shape) == 1 and len(spike_hist_y.shape) == 1:
      num_trials = 1
      dimension = '1-1'
    elif len(spike_hist_x.shape) == 2 and len(spike_hist_y.shape) == 2:
      num_trials = spike_hist_x.shape[0]
      if num_trials != spike_hist_y.shape[0]:
        raise ValueError('Inputs x, y num_trials do not math.')
      dimension = '2-2'

    # Construct jitter correction basis.
    num_bins_kernel = int(jitter_window_width / spk_bin_width)
    intervals = np.arange(0, num_bins, num_bins_kernel)
    if basis_type in ['B0B1', 'B0'] and dimension == '1-1':
      if jitter_type == 'interval':
        interval_cnt = np.add.reduceat(spike_hist_x, left_intervals)
        spike_hist_x_reduced = interval_cnt / num_bins_kernel
        spike_hist_x_conv = np.kron(spike_hist_x_reduced, np.ones(num_bins_kernel))
      if jitter_type == 'interval_binary':
        interval_cnt = np.add.reduceat(spike_hist_x, left_intervals)
        spike_hist_x_reduced = 1 - np.power((1-1/num_bins_kernel), interval_cnt)
        spike_hist_x_conv = np.kron(spike_hist_x_reduced, np.ones(num_bins_kernel))
      elif jitter_type == 'basic':
        kernel = np.ones([1, num_bins_kernel]) / num_bins_kernel
        spike_hist_x_conv = scipy.ndimage.convolve(spike_hist_x, kernel)

    elif basis_type in ['B0B1', 'B0'] and dimension == '2-2':
      if jitter_type == 'interval':
        interval_cnt = np.add.reduceat(spike_hist_x, intervals, axis=1)
        spike_hist_x_reduced = interval_cnt / num_bins_kernel
        spike_hist_x_conv = np.kron(spike_hist_x_reduced, np.ones([1,num_bins_kernel]))
      if jitter_type == 'interval_binary':
        interval_cnt = np.add.reduceat(spike_hist_x, intervals, axis=1)
        spike_hist_x_reduced = 1 - np.power((1-1/num_bins_kernel), interval_cnt)
        spike_hist_x_conv = np.kron(spike_hist_x_reduced, np.ones([1,num_bins_kernel]))
      elif jitter_type == 'basic':
        kernel = np.ones([1, num_bins_kernel]) / num_bins_kernel
        spike_hist_x_conv = scipy.ndimage.convolve(spike_hist_x, kernel)

    y = spike_hist_y.reshape(-1, 1)
    # y = spike_hist_y_conv.reshape(-1, 1)

    if link_func == 'linear':
      if basis_type == 'B0B1':
        x_conv = spike_hist_x_conv.reshape(-1)
        x = spike_hist_x.reshape(-1)
        X = np.vstack((x_conv, x)).T
      elif basis_type == 'B0':
        X = spike_hist_x_conv.reshape(-1, 1)
      elif basis_type == 'B1':
        X = spike_hist_x.reshape(-1, 1)

      y_hat, beta, nll = cls.optimize_spike_hist_lambda_linear_model(
          y, X, beta=None, offset=0, learning_rate=0.9, max_num_itrs=2000,
          epsilon=epsilon, verbose=verbose)

    elif link_func == 'exp':
      # y_log_mean_fr = np.log(np.mean(y))
      # offset = np.ones((len(y), 1)) * y_log_mean_fr
      offset = 0

      if basis_type == 'B0B1':
        spike_hist_x_conv = np.log(spike_hist_x_conv).reshape(-1)
        spike_hist_x_conv[spike_hist_x_conv==np.NINF] = 0
        x = spike_hist_x.reshape(-1)
        X = np.vstack([spike_hist_x_conv, x]).T
      elif basis_type == 'B1':
        X = spike_hist_x.reshape(-1, 1)

      y_hat, beta, nll = cls.optimize_spike_hist_log_lambda_linear_model(
          y, X, beta=None, offset=offset, learning_rate=0.9, max_num_itrs=2000,
          epsilon=epsilon, verbose=True)

    elif link_func == 'mse':
      y_hat, beta = cls.optimize_spike_hist_gaussian_linear_model(
          y, X, beta=None)

    if verbose and basis_type == 'B0':
      x = spike_hist_x.reshape(-1)
      x_conv = spike_hist_x_conv.reshape(-1)
      plt.figure(figsize=[16, 2.5])
      plt.plot(x)
      plt.plot(x_conv)
      plt.xlim(0, 500)
      plt.show()

    if verbose:
      print(f'overall mean FR: {np.mean(y) / spk_bin_width:.2f}')
      print(f'estimat mean FR: {np.mean(y_hat) / spk_bin_width:.2f}')
      y_hat = y_hat.reshape(num_trials, num_bins)
      plt.figure(figsize=(8, 3))
      plt.plot(spike_hist_y.mean(axis=0) / spk_bin_width)
      plt.plot(y_hat.mean(axis=0) / spk_bin_width)
      plt.show()

    return beta


  @classmethod
  def bivariate_spike_hist_coupling_filter_regression_closed(
      cls,
      spike_times_x,
      spike_times_y,
      trial_length,
      spk_bin_width,
      link_func='linear',
      basis_type='B1_binary',
      jitter_window_width=None):
    """Coupling filter model.

    In this method, we model the lambda = spike_x beta.
    Let y, x be binary sequences.
    nll = sum {-y log(x beta) + x beta}
    = - S log beta + N beta
    S is the number of sync between x and y. N is the number of spikes in x.
    """
    spike_hist_x, bins = cls.bin_spike_times(
        spike_times_x, spk_bin_width, trial_length)
    spike_hist_y, _    = cls.bin_spike_times(
        spike_times_y, spk_bin_width, trial_length)
    num_bins = len(bins)
    num_x_spikes = np.sum(spike_hist_x)

    if link_func == 'linear' and basis_type == 'B1_binary':
      y = spike_hist_y.reshape(-1)
      x = spike_hist_x.reshape(-1)
      x = (x != 0) + 0
      num_sync = np.dot(y, x)
      # Solve min_beta -S log beta + N beta. Take derivative we get.
      beta = num_sync / num_x_spikes

    elif link_func == 'linear' and basis_type == 'B1_cnt':
      y = spike_hist_y.reshape(-1)
      x = spike_hist_x.reshape(-1)
      num_sync = np.dot(y, x)
      # Solve min_beta -S log beta + N beta. Take derivative we get.
      beta = num_sync / num_x_spikes

    elif link_func == 'linear' and basis_type == 'B0':
      num_bins_kernel = int(jitter_window_width / spk_bin_width)
      intervals = np.arange(0, num_bins, num_bins_kernel)
      interval_cnt = np.add.reduceat(spike_hist_x, intervals, axis=1)
      spike_hist_x_reduced = (interval_cnt != 0) + 0
      if len(spike_hist_x.shape) == 1 and len(spike_hist_y.shape) == 1:
        kron_kernel = np.ones(num_bins_kernel)
      elif len(spike_hist_x.shape) == 2 and len(spike_hist_y.shape) == 2:
        kron_kernel = np.ones([1,num_bins_kernel])
      spike_hist_x_conv = np.kron(spike_hist_x_reduced, kron_kernel)
      x_conv = spike_hist_x_conv.reshape(-1)
      y = spike_hist_y.reshape(-1)
      num_sync = np.dot(y, x_conv)
      beta = num_sync / num_x_spikes

    elif link_func == 'exp':
      # Solve min_beta -S beta + N exp(beta). Take derivative we get.
      beta = np.log(num_sync / num_x_spikes)

    return beta


  @classmethod
  def bivariate_spike_hist_coupling_filter_regression_closed_mean(
      cls,
      spike_times_x,
      spike_times_y,
      trial_length,
      jitter_window_width,
      spk_bin_width,
      link_func='linear',
      mean_type='binary',
      verbose=False):
    """Coupling filter model.

    In this method, we model the lambda = spike_x beta.
    Let y, x be binary sequences.
    nll = sum {-y log(x beta) + x beta}
    = - S log beta + N beta
    S is the number of sync between x and y. N is the number of spikes in x.
    """
    if link_func != 'linear':
      raise TypeError('Only supports linear link now.')

    spike_hist_x, bins = cls.bin_spike_times(
        spike_times_x, spk_bin_width, trial_length)
    spike_hist_y, _ = cls.bin_spike_times(
        spike_times_y, spk_bin_width, trial_length)
    num_bins = len(bins)
    num_x_spikes = np.sum(spike_hist_x)

    num_bins_kernel = int(jitter_window_width / spk_bin_width)
    intervals = np.arange(0, num_bins, num_bins_kernel)
    interval_cnt = np.add.reduceat(spike_hist_x, intervals, axis=1)

    if mean_type == 'binary':
      spike_hist_x_reduced = 1 - np.power((1-1/num_bins_kernel), interval_cnt)
    elif mean_type == 'cnt':
      spike_hist_x_reduced = interval_cnt / num_bins_kernel

    spike_hist_x_mean = np.kron(spike_hist_x_reduced, np.ones([1,num_bins_kernel]))

    y = spike_hist_y.reshape(-1)
    x_mean = spike_hist_x_mean.reshape(-1)
    beta_mean = np.dot(y, x_mean) / num_x_spikes

    if verbose:
      print('beta mean', beta_mean)
    return beta_mean


  @classmethod
  def bivariate_spike_hist_coupling_filter_regression_jitter(
      cls,
      spike_times_x,
      spike_times_y,
      spk_bin_width,
      trial_length,
      jitter_window_width,
      link_func='linear',
      distribution_type='binom',
      calculation_type='closed',
      mean_correct_type=None,
      num_jitter=500,
      ci_alpha=0.01,
      binarize=True,
      verbose=False):
    """Perform jitter based test on cross correlation.

    Args:
      spike_times_x:
      spike_times_y:
      calculation_type: 'closed', 'optimize'
      calculation_type 'closed' + mean_correct_type 'binary' or'cnt'
      calculation_type 'optimize' + mean_correct_type True or False
    """
    if mean_correct_type is not None:
      beta_mean_hat = cls.bivariate_spike_hist_coupling_filter_regression_closed_mean(
          spike_times_x, spike_times_y, trial_length=trial_length,
          jitter_window_width=jitter_window_width, spk_bin_width=spk_bin_width,
          mean_type=mean_correct_type, link_func=link_func, verbose=False)
    else:
      beta_mean_hat = 0

    if calculation_type == 'closed':
      # Raw statistic.
      basis_type = 'B1_binary' if binarize else 'B1_cnt'
      beta_raw = cls.bivariate_spike_hist_coupling_filter_regression_closed(
          spike_times_x, spike_times_y, trial_length, spk_bin_width,
          link_func=link_func, basis_type=basis_type)
      beta_raw = beta_raw - beta_mean_hat

    elif calculation_type == 'optimize':
      # Note there's no `binary` option in the regression as it is the raw
      # calculation, and it is the binary by its nature.
      basis_type = 'B1' if mean_correct_type is None else 'B0B1'
      beta = cls.bivariate_spike_hist_coupling_filter_regression(
          spike_times_x, spike_times_y, trial_length, spk_bin_width,
          jitter_window_width, jitter_type='interval', link_func=link_func,
          basis_type=basis_type, epsilon=1e-9)
      if basis_type == 'B1':
        beta_raw = beta[0,0]
      elif basis_type == 'B0B1':
        delta = int(jitter_window_width / spk_bin_width)
        beta_raw = beta[1,0] * (delta-1) / delta

    # MC.
    if distribution_type == 'mc':
      beta_jitter = np.zeros([num_jitter])
      trange = tqdm(range(num_jitter), ncols=100) if verbose else range(num_jitter)
      for r in trange:
        spike_times_surrogate_x = cls.jitter_spike_times_interval(
            spike_times_x, jitter_window_width, num_jitter=1, verbose=False)
        if calculation_type == 'closed':
          (beta_jitter[r]
              ) = cls.bivariate_spike_hist_coupling_filter_regression_closed(
              spike_times_surrogate_x, spike_times_y, trial_length, spk_bin_width,
              link_func=link_func, basis_type=basis_type)
          beta_jitter[r] = beta_jitter[r] - beta_mean_hat
        elif calculation_type == 'optimize':
          # No need to remove the mean explicitly because it will be removed
          # by adding the jitter basis through `B0B1` basis design.
          beta = cls.bivariate_spike_hist_coupling_filter_regression(
              spike_times_surrogate_x, spike_times_y, trial_length, spk_bin_width,
              jitter_window_width, jitter_type='interval', link_func=link_func,
              basis_type=basis_type, epsilon=1e-9)
          if basis_type == 'B1':
            beta_jitter[r] = beta[0,0]
          elif basis_type == 'B0B1':
            beta_jitter[r] = beta[1,0] * (delta-1) / delta

      (beta_mean, beta_CI_down, beta_CI_up, p_val
          ) = cls.mc_regression_sync_mean_CI_pval(
          beta_raw, beta_jitter, ci_alpha)

    elif distribution_type == 'binom':
      spike_hist_x, _ = cls.bin_spike_times(
          spike_times_x, jitter_window_width, trial_length)
      spike_hist_x = spike_hist_x.reshape(-1)
      spike_hist_y, _ = cls.bin_spike_times(
          spike_times_y, jitter_window_width, trial_length)
      spike_hist_y = spike_hist_y.reshape(-1)
      delta = int(jitter_window_width / spk_bin_width)

      pmf, pmf_x = cls.regression_sync_binom_pmf(
          spike_hist_y, delta, spike_hist_x, binarize=binarize, verbose=False)
      pmf_x = pmf_x - beta_mean_hat
      (beta_CI_down, beta_CI_up, beta_mean, p_val, p_val_rnd
          ) = cls.pmf_CI_mean_pval(
          pmf_x, pmf, raw_stat=beta_raw, ci_alpha=ci_alpha)

    elif distribution_type == 'poisson':
      spike_hist_x, _ = cls.bin_spike_times(
          spike_times_x, jitter_window_width, trial_length)
      spike_hist_x = spike_hist_x.reshape(-1)
      spike_hist_y, _ = cls.bin_spike_times(
          spike_times_y, jitter_window_width, trial_length)
      spike_hist_y = spike_hist_y.reshape(-1)
      delta = int(jitter_window_width / spk_bin_width)

      (beta_CI_down, beta_CI_up, beta_mean, p_val, p_val_rnd
          ) = cls.regression_sync_poisson_mu_CI_pval(
          spike_hist_y, delta, spike_hist_x,
          beta_raw=beta_raw, ci_alpha=ci_alpha, verbose=False)
      beta_CI_down -= beta_mean_hat
      beta_CI_up -= beta_mean_hat
      beta_mean -= beta_mean_hat

    if verbose:
      print(f'p-val: {p_val:.3e}')
      print('beta_raw', beta_raw)
      print('beta_mean', beta_mean)
      print('beta_CI_down', beta_CI_down)
      print('beta_CI_up', beta_CI_up)

      plt.figure(figsize=[6, 3])
      if distribution_type == 'mc':
        # bins = np.arange(0.01, 0.08, 0.001)
        seaborn.distplot(beta_jitter, bins=None, color='grey', kde=False,
            norm_hist=True)
      elif distribution_type == 'binom':
        plt.plot(pmf_x, pmf/(pmf_x[1]-pmf_x[0]), color='k')
      elif distribution_type == 'normal':
        pass
      elif distribution_type == 'poisson':
        pass

      # plt.axvline(x=beta_raw, ls='--', c='k')
      plt.axvline(x=beta_mean, c='b')
      plt.axvline(x=beta_CI_down, c='b', ls=':')
      plt.axvline(x=beta_CI_up, c='b', ls=':')
      # plt.xlim(left=0)
      plt.xlim([-0.02, 0.02])
      # plt.ylim([0, 80])
      plt.xlabel(r'$\beta$')
      plt.show()

    return p_val, beta_raw


  @classmethod
  def create_sync_prob_mat(
      cls,
      num_nodes,
      prob_range):
    """Create a symmetric synchrony probability matrix.

    prob_range: [p_min, p_max]
    """
    p_min, p_max = prob_range
    sync_prob_mat = np.zeros([num_nodes, num_nodes])
    for row in range(num_nodes):
      for col in range(num_nodes):
        if row >= col:
          continue
        sync_prob_mat[row, col] = np.random.rand() * (p_max - p_min) + p_min
        sync_prob_mat[col, row] = sync_prob_mat[row, col]

    return sync_prob_mat


  @classmethod
  def multivariate_spike_hist_coupling_filter_regression(
      cls,
      spike_times_x,
      spike_times_y,
      trial_length,
      spk_bin_width,
      jitter_window_width,
      jitter_type='interval',
      basis_type='B1',
      epsilon=1e-6,
      link_func='linear',
      beta_init=None,
      verbose=False):
    """Coupling filter model.

    In this method, we model the lambda = X beta directly instead of
    log lambda = X beta.
    """
    spike_hist_y, bins = cls.bin_spike_times(
        spike_times_y, spk_bin_width, trial_length)
    num_bins = len(bins)
    num_nodes = len(spike_times_x)
    spike_hist_x = [np.empty(0) for _ in range(num_nodes)]

    for n in range(num_nodes):
      spike_hist_x[n], _ = cls.bin_spike_times(
          spike_times_x[n], spk_bin_width, trial_length)
    # spike_hist_x: neuron x trials x bins.
    spike_hist_x = np.stack(spike_hist_x, axis=0)

    if len(spike_hist_x[0].shape) == 1 and len(spike_hist_y.shape) == 1:
      num_trials = 1
      dimension = '1-1'
    elif len(spike_hist_x[0].shape) == 2 and len(spike_hist_y.shape) == 2:
      num_trials = spike_hist_y.shape[0]
      if num_trials != spike_hist_y.shape[0]:
        raise ValueError('Inputs x, y num_trials do not math.')
      dimension = '2-2'

    # Construct jitter correction basis.
    num_bins_kernel = int(jitter_window_width / spk_bin_width)
    intervals = np.arange(0, num_bins, num_bins_kernel)
    # TODO
    if basis_type in ['B0B1', 'B0'] and dimension == '1-1':
      raise ValueError('Only supports multiple trials.')

    elif basis_type in ['B0B1', 'B0'] and dimension == '2-2':
      if jitter_type == 'interval':
        interval_cnt = np.add.reduceat(spike_hist_x, intervals, axis=-1)
        spike_hist_x_reduced = interval_cnt / num_bins_kernel
        spike_hist_x_conv = np.kron(spike_hist_x_reduced,
                                    np.ones([1,1,num_bins_kernel]))
      if jitter_type == 'interval_binary':
        interval_cnt = np.add.reduceat(spike_hist_x, intervals, axis=-1)
        spike_hist_x_reduced = 1 - np.power((1-1/num_bins_kernel), interval_cnt)
        spike_hist_x_conv = np.kron(spike_hist_x_reduced,
                                    np.ones([1,1,num_bins_kernel]))
      elif jitter_type == 'basic':
        # Unit test code.
        # a = np.array([[[1, 2, 0, 0],
        #               [5, 3, 0, 4],
        #               [0, 0, 0, 7],
        #               [9, 3, 0, 0]]])
        # k = np.array([[[0,0],[0,1]]])
        # print(a.shape, k.shape)
        # scipy.ndimage.convolve(a, k)
        kernel = np.ones([1,1,num_bins_kernel]) / num_bins_kernel
        spike_hist_x_conv = scipy.ndimage.convolve(spike_hist_x, kernel)

    y = spike_hist_y.reshape(-1, 1)

    if link_func == 'linear':
      if basis_type == 'B0B1':
        x_conv = spike_hist_x_conv.reshape(num_nodes, -1)
        x = spike_hist_x.reshape(num_nodes, -1)
        X = np.vstack((x, x_conv)).T
      elif basis_type == 'B0':
        X = spike_hist_x_conv.reshape(num_nodes, -1).T
      elif basis_type == 'B1':
        X = spike_hist_x.reshape(num_nodes, -1).T

      y_hat, beta, nll = cls.optimize_spike_hist_lambda_linear_model(
          y, X, beta=beta_init, offset=0, learning_rate=0.9, max_num_itrs=2000,
          epsilon=epsilon, verbose=verbose)

    # TODO
    elif link_func == 'exp':
      raise ValueError('Only supports linear now.')

    if verbose and basis_type == 'B0':
      x = spike_hist_x.reshape(-1)
      x_conv = spike_hist_x_conv.reshape(-1)
      plt.figure(figsize=[16, 2.5])
      plt.plot(x)
      plt.plot(x_conv)
      plt.xlim(0, 500)
      plt.show()

    if verbose:
      print(f'overall mean FR: {np.mean(y) / spk_bin_width:.2f}')
      print(f'estimat mean FR: {np.mean(y_hat) / spk_bin_width:.2f}')
      y_hat = y_hat.reshape(num_trials, num_bins)
      plt.figure(figsize=(8, 3))
      plt.plot(spike_hist_y.mean(axis=0) / spk_bin_width)
      plt.plot(y_hat.mean(axis=0) / spk_bin_width)
      plt.show()

    return beta


  @classmethod
  def multivariate_spike_hist_coupling_filter_regression_jitter(
      cls,
      spike_times_x,
      spike_times_y,
      spk_bin_width,
      trial_length,
      jitter_window_width,
      link_func='linear',
      distribution_type='mc',
      basis_type=None,
      num_jitter=500,
      mean_correct=False,
      ci_alpha=0.01,
      verbose=False):
    """Perform jitter based test on cross correlation.

    Args:
      spike_times_x:
      spike_times_y:
    """
    num_nodes = len(spike_times_x)
    beta = cls.multivariate_spike_hist_coupling_filter_regression(
        spike_times_x, spike_times_y, trial_length, spk_bin_width,
        jitter_window_width, jitter_type='interval', link_func=link_func,
        basis_type=basis_type, epsilon=1e-9, verbose=False)
    if basis_type == 'B1':
      beta_raw = beta[:,0]
    elif basis_type == 'B0B1':
      delta = int(jitter_window_width / spk_bin_width)
      beta_raw = beta[:num_nodes,0] * (delta-1) / delta

    # MC.
    if distribution_type == 'mc':
      beta_jitter = np.zeros([num_jitter, num_nodes])
      # trange = tqdm(range(num_jitter), ncols=100) if verbose else range(num_jitter)
      trange = range(num_jitter)
      for r in trange:
        spike_times_surrogate_x = cls.jitter_spike_times_interval(
            spike_times_x, jitter_window_width, num_jitter=1, data_dim=3,
            verbose=False)
        # No need to remove the mean explicitly because it will be removed
        # by adding the jitter basis through `B0B1` basis design.
        beta = cls.multivariate_spike_hist_coupling_filter_regression(
            spike_times_surrogate_x, spike_times_y, trial_length, spk_bin_width,
            jitter_window_width, jitter_type='interval', link_func=link_func,
            basis_type=basis_type, beta_init=beta.copy(), epsilon=1e-9,
            verbose=False)

        if basis_type == 'B1':
          beta_jitter[r] = beta.reshape(-1)
        elif basis_type == 'B0B1':
          beta_jitter[r] = beta[:num_nodes,0] * (delta-1) / delta

      if mean_correct:
        null_beta_mean = beta_jitter.mean(axis=0)
        beta_jitter = beta_jitter - null_beta_mean
        beta_raw = beta_raw - null_beta_mean

      p_val = np.zeros(num_nodes)
      for n in range(num_nodes):
        (beta_mean, beta_CI_down, beta_CI_up, p_val[n]
            ) = cls.mc_regression_sync_mean_CI_pval(
            beta_raw[n], beta_jitter[:,n], ci_alpha)

    if verbose == 1 and num_nodes > 1:
      # print(f'p-val: {p_val:.3e}')
      # print('beta_raw', beta_raw)
      # print('beta_mean', beta_mean)
      # print('beta_CI_down', beta_CI_down)
      # print('beta_CI_up', beta_CI_up)

      fig = plt.figure(figsize=[3.5, 3])
      ax = fig.add_subplot(1, 1, 1)
      plt.plot([-1,1], [-1,1], ls='--', c='lightgrey')
      if distribution_type == 'mc':
        plt.plot(beta_jitter[:,0], beta_jitter[:,1], 'k.', ms=3)
      plt.plot(beta_raw[0], beta_raw[1], 'r+', ms=8)
      val_min = min(0, np.min(beta_jitter), min(beta_raw))
      val_max = max(np.max(beta_jitter), max(beta_raw))
      margin = 0.01
      ax.set_xticks(np.arange(-0.1, 0.2, 0.02))
      ax.set_yticks(np.arange(-0.1, 0.2, 0.02))
      ax.set_aspect('equal', 'box')
      ax.grid()
      ax.axis([val_min-margin, val_max+margin, val_min-margin, val_max+margin])
      plt.show()

    if verbose == 1 and num_nodes == 1:
      fig = plt.figure(figsize=[3, 2])
      ax = fig.add_subplot(1, 1, 1)
      if distribution_type == 'mc':
        seaborn.distplot(beta_jitter, norm_hist=True, kde=False, color='grey')
      plt.axvline(x=beta_raw, color='b')
      val_min = min(0, np.min(beta_jitter), min(beta_raw))
      val_max = max(np.max(beta_jitter), max(beta_raw))
      margin = 0.01
      ax.set_xticks(np.arange(-0.1, 0.2, 0.02))
      plt.xlim(val_min-margin, val_max+margin)
      plt.show()

    if verbose == 2:
      num_cols = min(8, num_nodes)
      num_rows = np.ceil(num_nodes / num_cols).astype(int)
      gs_kw = dict(width_ratios=[1]*num_cols, height_ratios=[1]*num_rows)
      fig, axs = plt.subplots(figsize=(2.5*num_cols, 1.5*num_rows),
          gridspec_kw=gs_kw, nrows=num_rows, ncols=num_cols)
      plt.subplots_adjust(hspace=0.1, wspace=0.1)
      axs = axs.reshape(-1) if num_cols > 1 else [axs]
      val_min = min(0, np.min(beta_jitter), min(beta_raw))
      val_max = max(np.max(beta_jitter), max(beta_raw))
      margin = 0.01
      for n in range(num_nodes):
        ax = fig.add_subplot(axs[n])
        ax.tick_params(left=False, labelleft=False, labelbottom=True)
        # if n == 0:
        #   ax.tick_params(left=False, labelleft=False, labelbottom=True)
        seaborn.distplot(beta_jitter[:,n], bins=None, norm_hist=True, kde=False,
                         color='grey')
        plt.axvline(x=beta_raw[n], color='b')
        # ax.xaxis.set_major_locator(MultipleLocator(0.02))
        # plt.xlim(val_min-margin, val_max+margin)
        # plt.xlim(left=val_min-margin)
      plt.show()

    return p_val, beta_raw


  @classmethod
  def plot_beta_comparison(
      cls,
      beta,
      plot_type=None):
    """Compare betas from multiple models.

    Args:
      beta: num_itrs x num_models x num_nodes
    """
    if len(beta.shape) == 2:
      separate = False
      num_itrs, num_models = beta.shape
    elif len(beta.shape) == 3:
      num_itrs, num_models, num_nodes = beta.shape

    # Beta comparison all together.
    if plot_type == 1:
      plt.figure(figsize=(4, 4))
      plt.axvline(x=0, c='lightgrey')
      plt.axhline(y=0, c='lightgrey')
      plt.plot([-1, 1], [-1, 1], ls='--', c='lightgrey')

      plt.plot(beta[:,0], beta[:,1], 'k.', ms=2)
      val_min = np.min(beta)
      val_max = np.max(beta)
      margin = 0.01
      plt.axis([val_min-margin, val_max+margin, val_min-margin, val_max+margin])
      plt.xlabel(r'$\beta$ case 1')
      plt.ylabel(r'$\beta$ case 2')
      plt.show()
    # Separate beta entry comparison.
    elif plot_type == 2:
      num_cols = min(8, num_nodes)
      num_rows = np.ceil(num_nodes / num_cols).astype(int)
      gs_kw = dict(width_ratios=[1]*num_cols, height_ratios=[1]*num_rows)
      fig, axs = plt.subplots(figsize=(2.5*num_cols, 2.5*num_rows),
          gridspec_kw=gs_kw, nrows=num_rows, ncols=num_cols)
      plt.subplots_adjust(hspace=0.1, wspace=0.1)
      axs = axs.reshape(-1) if num_nodes > 1 else [axs]
      val_min = min(0, np.min(beta))
      val_max = np.max(beta)
      margin = 0.01
      for n in range(num_nodes):
        ax = fig.add_subplot(axs[n])
        ax.tick_params(left=True, labelleft=False, labelbottom=False)
        if n == 0:
          ax.tick_params(left=True, labelleft=True, labelbottom=True)
          plt.xlabel(r'$\beta$ case 1')
          plt.ylabel(r'$\beta$ case 2')
        plt.axvline(x=0, c='lightgrey')
        plt.axhline(y=0, c='lightgrey')
        plt.plot([-1,1], [-1,1], ls='--', c='lightgrey')
        plt.plot(beta[:,0,n], beta[:,1,n], 'k.', ms=2)
        # ax.xaxis.set_major_locator(MultipleLocator(0.02))
        # ax.yaxis.set_major_locator(MultipleLocator(0.02))
        ax.set_aspect('equal', 'box')
        plt.axis([val_min-margin, val_max+margin, val_min-margin, val_max+margin])
      plt.show()
    elif plot_type == 3:
      for m in range(2):
        if np.sum(np.abs(beta[:,m,:])) == 0:
          # Skip the empty model.
          continue
        num_cols = min(8, num_nodes)
        num_rows = np.ceil(num_nodes / num_cols).astype(int)
        gs_kw = dict(width_ratios=[1]*num_cols, height_ratios=[1]*num_rows)
        fig, axs = plt.subplots(figsize=(3*num_cols, 2*num_rows),
            gridspec_kw=gs_kw, nrows=num_rows, ncols=num_cols)
        plt.subplots_adjust(hspace=0.1, wspace=0.2)
        axs = axs.reshape(-1) if num_nodes > 1 else [axs]
        val_min = min(0, np.min(beta))
        val_max = np.max(beta)
        margin = 0.01
        for n in range(num_nodes):
          ax = fig.add_subplot(axs[n])
          # ax.tick_params(left=True, labelleft=False, labelbottom=False)
          # if n == 0:
          #   ax.tick_params(left=True, labelleft=True, labelbottom=True)
          #   plt.xlabel(r'$\beta$ case 1')
          #   plt.ylabel(r'$\beta$ case 2')
          # plt.axvline(x=0, c='lightgrey')
          # plt.axhline(y=0, c='lightgrey')
          # plt.plot([-1,1], [-1,1], ls='--', c='lightgrey')
          # plt.plot(beta[:,0,n], beta[:,1,n], 'k.', ms=2)
          seaborn.distplot(beta[:,m,n], kde=False, norm_hist=False, color='grey')

          # ax.xaxis.set_major_locator(MultipleLocator(0.02))
          # ax.yaxis.set_major_locator(MultipleLocator(0.02))
          # ax.set_aspect('equal', 'box')
          # plt.axis([val_min-margin, val_max+margin, val_min-margin, val_max+margin])
          plt.xlim(left=val_min-margin)
        plt.show()


  @classmethod
  def regression_sync_poisson_mu_CI_pval(
      cls,
      m,
      delta,
      n,
      beta_raw,
      ci_alpha,
      verbose=False):
    """Joint pmfs together using Poisson approximation."""
    stat_raw = beta_raw * np.sum(n)
    mu = np.dot(m, n) / delta
    mu_beta = mu / np.sum(n)
    CI_down = scipy.stats.poisson.ppf(ci_alpha/2, mu) / np.sum(n)
    CI_up = scipy.stats.poisson.ppf(1-ci_alpha/2, mu) / np.sum(n)
    # P(x >= xorr)
    p_val = 1 - scipy.stats.poisson.cdf(stat_raw-1, mu)
    # P(x >= xorr+1) + u * P(x = xorr)
    p_val_rnd = (1 - scipy.stats.poisson.cdf(stat_raw, mu) +
        scipy.stats.poisson.pmf(stat_raw, mu) * np.random.rand())

    if verbose:
      pmf_x = np.arange(np.sum(m) + np.sum(n)) / np.sum(n)
      plt.figure(figsize=[8,2])
      plt.plot(pmf_x, joint_pmf, 'k')
      # plt.xlim(mu-100, mu+100)

    return CI_down, CI_up, mu_beta, p_val, p_val_rnd


  @classmethod
  def regression_sync_binom_pmf(
      cls,
      m,
      delta,
      n,
      verbose=False,
      binarize=True):
    """Joint pmfs together using convolution."""
    joint_pmf = np.ones(1)
    num_spikes = np.sum(n)
    m = m.astype(int)
    n = n.astype(int)

    for i in range(len(m)):
      if m[i] == 0 or n[i] == 0:
        continue
      if binarize:
        pmf = cls.binaried_pmf_single(m[i], delta, n[i])
      else:
        pmf = cls.binom_pmf_single(m[i], delta, n[i])
      joint_pmf = np.convolve(joint_pmf, pmf)

    pmf_x = np.arange(len(joint_pmf)) / num_spikes

    if verbose:
      peak_x = np.dot(m / delta, n) / num_spikes
      plt.figure(figsize=[8,2])
      plt.plot(pmf_x, joint_pmf, 'k')
      # plt.xlim(peak_x, peak_x+100)

    epsilon = 1e-12
    # Keep the session in the beginning, remove the right tail.
    valid_index = np.where(joint_pmf > epsilon)[0][-1]
    joint_pmf = joint_pmf[:valid_index]
    joint_pmf = joint_pmf / joint_pmf.sum()
    pmf_x = pmf_x[:valid_index]

    return joint_pmf, pmf_x


  @classmethod
  def mc_regression_sync_mean_CI_pval(
      cls,
      beta_raw,
      beta_jitter,
      ci_alpha):
    """Obtain mean, CI, p-values from regression."""
    num_jitter = len(beta_jitter)
    beta_CI_up = np.quantile(beta_jitter, 1-ci_alpha/2)
    beta_CI_down = np.quantile(beta_jitter, ci_alpha/2)
    beta_mean = np.mean(beta_jitter)

    # p-values using the distribution.
    p_val = (sum(beta_jitter > beta_raw) + 1) / (num_jitter + 1)

    return beta_mean, beta_CI_down, beta_CI_up, p_val


  @classmethod
  def optimize_spike_hist_lambda_linear_model(
      cls,
      y,
      X,
      beta=None,
      offset=0,
      max_num_itrs=2000,
      learning_rate=0.6,
      epsilon=1e-6,
      verbose=0):
    """nll = -y log lambda + lambda, lambda = X beta + offset."""
    clip_min = 1e-12
    num_samples, num_basis = X.shape
    nll = []

    if beta is None:
      beta = np.ones([num_basis, 1]) * 0.05
    # Check if X_i is sparse.
    sparsity = X.T @ y
    sparsity = sparsity.reshape(-1)
    sparsity = sparsity < 2
    beta[sparsity,:] = 1e-6

    # Skip the veriables where all the regressors are 0.
    if np.isscalar(offset) and offset != 0:
      supp = np.arange(len(y))
    elif np.isscalar(offset) and offset == 0:
      supp = np.where(X.sum(axis=1) != 0)[0]
    elif isinstance(offset, np.ndarray):
      X_supp = np.where(X.sum(axis=1) != 0)[0]
      offset_supp = np.where(supp != 0)[0]
      supp = np.union1d(X_supp, offset_supp)

    X_original = X
    offset_original = offset
    X_sum = X.sum(axis=0).reshape(-1,1)
    y = y[supp,:]
    X = X[supp,:]

    beta_old = beta
    for itr in range(max_num_itrs):
      # lmbd = np.clip(X @ beta + offset, a_min=clip_min, a_max=np.float('inf'))
      lmbd = X @ beta + offset

      if verbose == 2:
        log_lmbd = np.log(lmbd).reshape(-1)
        curr_nll = cls.spike_trains_neg_log_likelihood(log_lmbd, y.T)
        nll.append(curr_nll)

      # gradient = - X.T @ (y / lmbd) + X.sum(axis=0).reshape(-1,1)
      gradient = - X.T @ (y / lmbd) + X_sum
      # delta = gradient
      hessian = X.T @ (1 / np.square(lmbd) * X)
      delta = np.linalg.inv(hessian) @ gradient
      beta = beta - learning_rate * delta

      # Check convergence.
      beta_err = np.sum(np.abs(beta_old - beta))
      if beta_err < epsilon:
        break
      beta_old = beta

    # Final check of the likelihood.
    # lmbd = np.clip(X @ beta + offset, a_min=clip_min, a_max=np.float('inf'))
    lmbd = X @ beta + offset
    log_lmbd = np.log(lmbd).reshape(-1)
    curr_nll = cls.spike_trains_neg_log_likelihood(log_lmbd, y.T)
    nll.append(curr_nll)

    if itr == max_num_itrs-1:
      warnings.warn(f'Reach max iterations. Last err:{beta_err:.3e}')

    if verbose == 1:
      print('X_original.shape', X_original.shape)
      print('X.shape', X.shape)
      print('num iterations: ', itr)
    if verbose == 2:
      print('X_original.shape', X_original.shape)
      print('X.shape', X.shape)
      print('num iterations: ', itr)
      print('beta')
      print(beta)

      plt.figure(figsize=[6,3])
      plt.plot(nll)
      plt.title(f'nll final: {nll[-1]:.5f}')

    lmbd = X_original @ beta + offset_original
    return lmbd.reshape(-1), beta, nll[-1]


  @classmethod
  def optimize_spike_hist_log_lambda_linear_model(
      cls,
      y,
      X,
      beta=None,
      offset=0,
      learning_rate=0.8,
      max_num_itrs=2000,
      epsilon=1e-8,
      verbose=False):
    """nll = -y log lambda + lambda, lambda = exp(X beta)."""
    num_samples, num_basis = X.shape
    nll = []

    if beta is None:
      beta = np.ones([num_basis, 1]) * 2
    beta_old = beta

    for itr in range(max_num_itrs):
      log_lmbd = (X @ beta + offset).reshape(-1)
      curr_nll = cls.spike_trains_neg_log_likelihood(log_lmbd, y.T)
      nll.append(curr_nll)

      eta = X @ beta + offset
      mu = np.exp(eta)
      gradient = X.T @ (-y + mu)
      hessian = X.T @ (mu * X)
      # delta = gradient
      delta = np.linalg.inv(hessian) @ gradient
      beta = beta - learning_rate * delta

      # Check convergence.
      beta_err = np.sum(np.abs(beta_old - beta))
      if beta_err < epsilon:
        break
      beta_old = beta

    if itr == max_num_itrs-1:
      warnings.warn(f'Reach max iterations. Last err:{beta_err:.3e}')

    if verbose:
      print('num iterations: ', itr)
      plt.figure(figsize=[6,3])
      plt.plot(nll)
      plt.title(f'nll final: {nll[-1]}')
      print(beta)

    return mu.reshape(-1), beta, nll[-1]


  @classmethod
  def optimize_spike_hist_gaussian_linear_model(
      cls,
      y,
      X,
      beta=None):
    """nll = (lmbd - y)^2"""
    learning_rate = 0.00001
    num_samples, num_basis = X.shape

    if beta is None:
      beta = np.ones([num_basis, 1]) * 0.5

    for itr in range(800):
      eta = X @ beta
      mu = eta

      nll = np.sum(np.square(y - mu))
      if itr % 100 == 0:
        print(beta)
        print(nll)

      gradient = 2 * X.T @ (mu - y)
      beta = beta - learning_rate * gradient

    return mu.reshape(-1), beta


  def models_comparision(
      self,
      spike_times_x,
      spike_times_y,
      trial_length,
      jitter_window_width,
      spk_bin_width):
    """Models comparison."""
    p_vals = np.zeros(5)

    # Xorr.
    lag_range = [-spk_bin_width/2, spk_bin_width/2]
    _, _, _, _, _, p_vals[0], _ = self.cross_correlation_jitter(
        spike_times_x, spike_times_y, spk_bin_width, trial_length,
        lag_range, jitter_window_width, distribution_type='binom', verbose=False)

    # LRT method.
    p_vals[1] = self.bivariate_spike_hist_coupling_filter_regression_lrt(
        spike_times_x, spike_times_y, trial_length, spk_bin_width,
        jitter_window_width, jitter_type='interval', basis_type='B0B1',
        epsilon=1e-10, cache_parameters=True)

    # # Jitter regression spike cnt.
    # p_vals[2], _ = self.bivariate_spike_hist_coupling_filter_regression_jitter(
    #     spike_times_x, spike_times_y, spk_bin_width, trial_length,
    #     jitter_window_width, link_func='linear', distribution_type='binom',
    #     binarize=False, verbose=False)

    # Jitter regression spike binarized.
    p_vals[3], _ = self.bivariate_spike_hist_coupling_filter_regression_jitter(
        spike_times_x, spike_times_y, spk_bin_width, trial_length,
        jitter_window_width, link_func='linear', distribution_type='binom',
        binarize=True, verbose=False)

    # Regression B0B1 + centered Binomial distribution inference
    p_vals[4], _ = self.bivariate_spike_hist_coupling_filter_regression_jitter(
        spike_times_x, spike_times_y, spk_bin_width, trial_length,
        jitter_window_width, link_func='linear', distribution_type='binom',
        binarize=True, calculation_type='optimize', mean_correct_type='binary',
        verbose=False)

    return p_vals


  @classmethod
  def roc(
      cls,
      neg_scores=None,
      pos_scores=None,
      neg_pos_score_pairs=None,
      score_range=[0, 1],
      model_labels=None,
      verbose=False,
      file_path=None):
    """ROC analysis.

    Args:
      The function takes two types of inputs, either `neg_scores_pos_scores` or
      `neg_pos_score_pairs`.

      neg_scores: num_trias x models.
      pos_scores: num_trias x models.
      neg_pos_score_pairs: [(neg_score, pos_score), ...]. num_models pairs.
    """
    # linetype = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'r']
    # linetype = ['k', 'k--', 'tab:green', 'tab:orange', 'r']
    linetype = ['k', 'k--', 'k:']

    if (neg_pos_score_pairs is None and pos_scores is not None and
        neg_scores is not None):
      neg_scores = np.array(neg_scores)
      pos_scores = np.array(pos_scores)
      num_roc = neg_scores.shape[0]
      neg_pos_score_pairs = []
      for m in range(num_roc):
        if np.sum(np.abs(neg_scores[m,:])) == 0:
          continue
        neg_pos_score_pairs.append((neg_scores[m,:], pos_scores[m,:]))
    else:
      num_roc = len(neg_pos_score_pairs)

    num_thresholds = 501
    thresholds = np.linspace(score_range[0], score_range[1], num_thresholds)

    gs_kw = dict(width_ratios=[1], height_ratios=[1]*1)
    fig, axs = plt.subplots(figsize=(3.3, 3), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    ax = fig.add_subplot(axs)
    for roc_id in range(num_roc):
      fpr = np.zeros(num_thresholds)
      tpr = np.zeros(num_thresholds)
      scores_neg_list = neg_pos_score_pairs[roc_id][0]
      scores_pos_list = neg_pos_score_pairs[roc_id][1]
      num_real_neg = len(scores_neg_list)
      num_real_pos = len(scores_pos_list)
      print(f'num_neg:{num_real_neg}\tum_pos:{num_real_pos}\t')
      for r, threshold in enumerate(thresholds):
        num_false_pos = sum(scores_neg_list < threshold)
        num_true_pos = sum(scores_pos_list < threshold)
        fpr[r] = num_false_pos / num_real_neg
        tpr[r] = num_true_pos / num_real_pos
      plt.plot(fpr, tpr, linetype[roc_id], label=model_labels[roc_id])
      # plt.plot(fpr, tpr, 'k')
    plt.plot([-1, 2], [-1, 2], ls='--', c='lightgrey')
    plt.axis([-0.015, 1.01, -0.01, 1.01])
    plt.xticks([0, 1], [0, 1])
    plt.yticks([0, 1], [0, 1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc='lower right')

    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      print('save figure:', file_path)
    plt.show()

    return

    gs_kw = dict(width_ratios=[1]*2, height_ratios=[1]*1)
    fig, axs = plt.subplots(figsize=(7, 3), gridspec_kw=gs_kw,
        nrows=1, ncols=2)
    ax = fig.add_subplot(axs[1])
    ax.tick_params(labelbottom=True, labelleft=False)
    for roc_id in range(num_roc):
      uniform = np.zeros(num_thresholds)
      fpr = np.zeros(num_thresholds)
      scores_neg_list = neg_pos_score_pairs[roc_id][0]
      num_real_pos = len(scores_neg_list)
      for r, threshold in enumerate(thresholds):
        num_false_pos = sum(scores_neg_list < threshold)
        uniform[r] = threshold
        fpr[r] = num_false_pos / num_real_pos
      plt.plot(uniform, fpr, linetype[roc_id], label=model_labels[roc_id])
    plt.plot([-1, 2], [-1, 2], ls='--', c='grey')
    plt.axis([-0.015, 1.01, -0.01, 1.01])
    plt.xticks([0, 1], [0, 1])
    plt.yticks([0, 1], [0, 1])
    plt.xlabel('Uniform')
    plt.ylabel('FPR')
    plt.show()


    # ax = fig.add_subplot(axs[2])
    # ax.tick_params(labelbottom=True, labelleft=False)
    # for roc_id in range(num_roc):
    #   uniform = np.zeros(num_thresholds)
    #   tpr = np.zeros(num_thresholds)
    #   scores_pos_list = neg_pos_score_pairs[roc_id][1]
    #   num_real_pos = len(scores_pos_list)
    #   for r, threshold in enumerate(thresholds):
    #     num_true_pos = sum(scores_pos_list < threshold)
    #     uniform[r] = threshold
    #     tpr[r] = num_true_pos / num_real_pos
    #   plt.plot(uniform, tpr, linetype[roc_id])
    # plt.plot([-1, 2], [-1, 2], ls='--', c='grey')
    # plt.axis([-0.02, 1.02, 0.02, 1.02])
    # plt.xticks([0, 1], [0, 1])
    # plt.yticks([0, 1], [0, 1])
    # plt.xlabel('Uniform')
    # plt.ylabel('TPR')
    # plt.show()

    # Plot distributions of the scores.
    bins = np.linspace(0, 1, 21)
    gs_kw = dict(width_ratios=[1]*num_roc, height_ratios=[1]*2)
    fig, axs = plt.subplots(figsize=(3*num_roc, 5), gridspec_kw=gs_kw,
        nrows=2, ncols=num_roc)
    # plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
    if num_roc == 1:
      axs = axs.reshape(-1,1)

    for roc_id in range(num_roc):
      scores_neg_list = neg_pos_score_pairs[roc_id][0]
      scores_pos_list = neg_pos_score_pairs[roc_id][1]

      ax = fig.add_subplot(axs[0, roc_id])
      if roc_id == 0:
        ax.tick_params(labelbottom=False, labelleft=True)
      else:
        ax.tick_params(labelbottom=False, labelleft=False)
      seaborn.distplot(scores_neg_list, bins=bins,
          kde=False, norm_hist=True, color='grey')
      ax.axhline(y=1, ls='--', c='k')
      plt.xlim(0,1)
      plt.ylim(0, 3)
      plt.yticks([0, 1, 2], [0, 1, 2])
      plt.xticks([0, 1], [0, 1])
      plt.title(f'{model_labels[roc_id]}')
      plt.text(0.85, 0.9, 'Neg', color='k', size=10, transform=ax.transAxes)

      ax = fig.add_subplot(axs[1, roc_id])
      if roc_id == 0:
        ax.tick_params(labelbottom=True, labelleft=True)
      else:
        ax.tick_params(labelbottom=False, labelleft=False)
      seaborn.distplot(scores_pos_list, bins=bins,
        kde=False, norm_hist=True, color='grey')
      plt.text(0.85, 0.9, 'Pos', color='k', size=10, transform=ax.transAxes)
      plt.xlim(0,1)
      plt.ylim(0, 15)
      plt.yticks([0, 5, 10], [0, 5, 10])
      plt.xticks([0, 1], [0, 1])


  @classmethod
  def allocate_basis_knots(
      cls,
      num_knots,
      filter_length,
      offset=0.008,
      verbose=False):
    """Allocate knots similar to Pillow et. al. 2008 p.6."""
    # nonlinearity for stretching x axis (and its inverse)
    x_range = np.array([0, filter_length])
    y_range = np.log(x_range + offset)
    y_knots = np.linspace(y_range[0], y_range[1], num_knots)
    knots = np.exp(y_knots) - offset
    knots[0] = 0

    if verbose:
      _, auc, _ = cls.bspline_basis(4, knots, [0], num_tail_drop=2, verbose=True)
      print('knots', knots)
      print('AUC', auc)
    return knots


  @classmethod
  def bspline_basis(
      cls,
      spline_order,
      distinct_knots,
      x,
      derivative_ord=0,
      num_tail_drop=0,
      verbose=False):
    """Creates B-spline basis.

    Details see Hastie Tibshirani Friedman 2009 Springer 
    - The elements of statistical learning p. 189.

    Args:
      spline_order: cubic spline order is 4, degree 3 polynomial.
          piecewise constant is order 1, degree 0 polynomial.
      distinct_knots: Includes two ends (no padding).
      x:
    """
    if np.isscalar(x):
      x = [x]

    spline_degree = spline_order - 1
    # t: vector of knots, c: coefficients, k: spline polynomial degree.
    tck=[0, 0, spline_degree]
    num_basis = len(distinct_knots) + spline_degree - 1
    # Pad the boundary knots.
    knots_pad = np.hstack(([distinct_knots[0]] * spline_degree,
                            distinct_knots,
                           [distinct_knots[-1]] * spline_degree))
    tck[0] = knots_pad

    # Basis or derivative of basis.
    basis = np.zeros((len(x), num_basis-num_tail_drop))
    for i in range(num_basis-num_tail_drop):
      vec = [0] * num_basis
      vec[i] = 1.0
      tck[1] = vec
      x_i = scipy.interpolate.splev(x, tck, der=derivative_ord)
      basis[:,i] = x_i

    # Integral of basis.
    basis_integral = np.zeros(num_basis-num_tail_drop)
    for i in range(num_basis-num_tail_drop):
      left_end = knots_pad[i]
      right_end = knots_pad[i+spline_degree+1]
      # The derivation for this shortcut is in my report.
      basis_integral[i] = (right_end - left_end) / spline_order
      # vec = [0] * num_basis
      # vec[i] = 1.0
      # tck[1] = vec
      # x_i = scipy.interpolate.splint(left_end, right_end, tck)
      # basis_integral[i] = x_i

    if verbose:
      dt = 0.0002
      t = np.linspace(distinct_knots[0], distinct_knots[-1],
                      int(np.round(distinct_knots[-1]/dt)) + 1)
      y = np.zeros((len(t), num_basis-num_tail_drop))
      for i in range(num_basis-num_tail_drop):
        vec = [0] * num_basis
        vec[i] = 1.0
        tck[1] = vec
        x_i = scipy.interpolate.splev(t, tck, der=derivative_ord)
        # y[:,i] = x_i / basis_integral[i]  # Normalize basis.
        y[:,i] = x_i
      plt.figure(figsize=[8, 2])
      plt.plot(t, y)
      plt.plot(distinct_knots, np.zeros(len(distinct_knots)), 'rx')
      # plt.title('Basis splines')
      for sample in x:
        plt.axvline(x=sample, c='lightgrey')
      plt.xlabel('Time [sec]')
      plt.show()

      # Integral of the basis.
      plt.figure(figsize=[8, 2])
      left_end = distinct_knots[0]
      basis_integral_plot = np.zeros((len(t), num_basis-num_tail_drop))
      for i in range(num_basis-num_tail_drop):
        vec = [0] * num_basis
        vec[i] = 1.0
        tck[1] = vec
        for j, right_end in enumerate(t):
          x_ji = scipy.interpolate.splint(left_end, right_end, tck)
          basis_integral_plot[j,i] = x_ji

        left_end = knots_pad[i]
        right_end = knots_pad[i+spline_degree+1]
        plt.plot([left_end, right_end], [basis_integral[i]]*2,
                 lw=0.5, c='lightgrey')
        plt.scatter([left_end, right_end], [basis_integral[i]]*2, marker='s')
      plt.plot(t, basis_integral_plot)
      plt.plot(distinct_knots, np.zeros(len(distinct_knots)), 'rx')
      for sample in x:
        plt.axvline(x=sample, c='lightgrey')
      plt.xlabel('Time [sec]')
      plt.show()

    return basis, basis_integral, tck


  @classmethod
  def spike_times_neg_log_likelihood(
      cls,
      lmbd_samples,
      lmbd_integral):
    """Negative log-likelihood of continuous spike times."""
    err_ind = np.where(lmbd_samples <= 0)[0]
    if len(err_ind) > 0:
      warnings.warn(f'NLL error: {len(err_ind)}/{len(lmbd_samples)} non-pos.')
    nll = -np.sum(np.log(lmbd_samples)) + lmbd_integral
    return nll


  @classmethod
  def construct_regressors_const(
      cls,
      spike_times_x,
      spike_times_y,
      trial_window,
      verbose=False):
    """Build GLM regressors using square basis."""
    num_trials = len(spike_times_x)
    num_spikes_y = [len(spikes) for spikes in spike_times_y]
    num_spikes_y = np.sum(num_spikes_y)
    X = np.ones([num_spikes_y, 1])
    trial_length = trial_window[1] - trial_window[0]
    basis_integral = np.zeros(1) + trial_length * num_trials

    return X, basis_integral


  @classmethod
  def construct_regressors_square(
      cls,
      spike_times_x,
      spike_times_y,
      filter_length,
      trial_length=None,
      trial_window=None,
      mean_norm=False,
      verbose=False):
    """Build GLM regressors using square basis.

    We assume the spike_times are already in the trial window, no need to clip
    them further.
    """
    num_trials = len(spike_times_x)
    num_spikes_x = [len(spikes) for spikes in spike_times_x]
    num_spikes_x = np.sum(num_spikes_x)
    num_spikes_y = [len(spikes) for spikes in spike_times_y]
    num_spikes_y = np.sum(num_spikes_y)
    if trial_window is None and trial_length is not None:
      trial_window = [0, trial_length]
    elif trial_window is not None and trial_length is None:
      trial_length = trial_window[1] - trial_window[0]

    X = np.zeros([num_spikes_y, 1])
    integral_error = 0
    spk_cnt = -1
    for r in range(num_trials):
      spikes_x = spike_times_x[r]
      spikes_y = spike_times_y[r]
      # Spikes near right boundary. No need to consider the left boundary error
      # since the filter is always on the right side of the spikes.
      boundary_gaps_r = trial_window[1] - spikes_x[spikes_x >
          trial_window[1] - filter_length]
      error_r = filter_length - boundary_gaps_r
      integral_error += error_r.sum()

      for spk_y in spikes_y:
        spk_cnt = spk_cnt + 1
        delays = spk_y - spikes_x
        # Can't use delays>=0, don't include current spike. This will be
        # more problematic when doing self-coupling effect.
        delays = delays[(delays>0) & (delays<=filter_length)]
        X[spk_cnt] = len(delays)

    # Get the integral.
    basis_integral = num_spikes_x * filter_length - integral_error

    if mean_norm and trial_length is not None:
      X = X - basis_integral / num_trials / trial_length
      basis_integral = np.array([0])

    return X, basis_integral


  @classmethod
  def construct_regressors_bspline(
      cls,
      spike_times_x,
      spike_times_y,
      num_knots,
      filter_length,
      space_par,
      num_tail_drop,
      verbose=False):
    """Build GLM regressors."""
    knots = cls.allocate_basis_knots(num_knots, filter_length, space_par,
        verbose=False)
    # knots=[0.0, 0.025, 0.029, 0.0295, 0.03, 0.04, 0.08]
    num_trials = len(spike_times_x)
    num_spikes_y = [len(spikes) for spikes in spike_times_y]
    num_spikes_y = np.sum(num_spikes_y)
    num_spikes_x = [len(spikes) for spikes in spike_times_x]
    num_spikes_x = np.sum(num_spikes_x)

    X = np.zeros([num_spikes_y, num_knots+2-num_tail_drop])
    spk_cnt = -1
    for r in range(num_trials):
      spikes_x = spike_times_x[r]
      spikes_y = spike_times_y[r]
      for spk_y in spikes_y:
        spk_cnt = spk_cnt + 1
        delays = spk_y - spikes_x
        # Can't use delays>=0, don't include current spike.
        delays = delays[(delays>0) & (delays<=filter_length)]
        if len(delays) == 0:
          continue
        samples,_,_ = cls.bspline_basis(4, knots, delays,
            num_tail_drop=num_tail_drop, verbose=False)
        X[spk_cnt] = samples.sum(axis=0)

    # Get the integral.
    _, basis_integral, tck = cls.bspline_basis(4, knots, [0],
        num_tail_drop=num_tail_drop, verbose=verbose)

    return X, basis_integral * num_spikes_x, tck


  @classmethod
  def construct_regressors_exp(
      cls,
      spike_times_x,
      spike_times_y,
      beta,
      trial_length=None,
      trial_window=None,
      mean_norm=False,
      verbose=False):
    """Build GLM regressors using square basis.

    We assume the spike_times are already in the trial window, no need to clip
    them further.
    """
    num_trials = len(spike_times_x)
    num_spikes_x = [len(spikes) for spikes in spike_times_x]
    num_spikes_x = np.sum(num_spikes_x)
    num_spikes_y = [len(spikes) for spikes in spike_times_y]
    num_spikes_y = np.sum(num_spikes_y)
    if trial_window is None and trial_length is not None:
      trial_window = [0, trial_length]
    elif trial_window is not None and trial_length is None:
      trial_length = trial_window[1] - trial_window[0]

    X = np.zeros([num_spikes_y, 1])
    spk_cnt = -1
    basis_integral = 0
    for r in range(num_trials):
      spikes_x = spike_times_x[r]
      spikes_y = spike_times_y[r]

      for spk_y in spikes_y:
        spk_cnt = spk_cnt + 1
        delays = spk_y - spikes_x
        # Can't use delays>=0, don't include current spike.
        delays = delays[(delays>0)]
        if len(delays) == 0:
          X[spk_cnt] = 0
        else:
          X[spk_cnt] = np.exp(-beta*delays).sum()

      # Get the integral.
      trial_integral = 1/beta*(1-np.exp(-beta*(trial_window[1]-spikes_x)))
      basis_integral += trial_integral.sum()

    # if mean_norm and trial_length is not None:
      # X = X - basis_integral / num_trials / trial_length
      # basis_integral = np.array([0])

    return X, basis_integral


  @classmethod
  def construct_regressors_exp_derivative(
      cls,
      spike_times_x,
      spike_times_y,
      beta,
      trial_window=None,
      verbose=False):
    """Build GLM regressors using square basis.

    We assume the spike_times are already in the trial window, no need to clip
    them further.
    """
    num_trials = len(spike_times_x)
    num_spikes_x = [len(spikes) for spikes in spike_times_x]
    num_spikes_x = np.sum(num_spikes_x)
    num_spikes_y = [len(spikes) for spikes in spike_times_y]
    num_spikes_y = np.sum(num_spikes_y)
    trial_length = trial_window[1] - trial_window[0]

    X = np.zeros([num_spikes_y, 1])
    delay_basis = np.zeros(num_spikes_y)
    spk_cnt = -1
    basis_integral = 0
    delay_integral = 0
    for r in range(num_trials):
      spikes_x = spike_times_x[r]
      spikes_y = spike_times_y[r]

      for spk_y in spikes_y:
        spk_cnt = spk_cnt + 1
        delays = spk_y - spikes_x
        # Can't use delays>=0, don't include current spike.
        delays = delays[(delays>0)]
        if len(delays) == 0:
          X[spk_cnt] = 0
          delay_basis[spk_cnt] = 0
        else:
          X[spk_cnt] = np.exp(-beta*delays).sum()
          delay_basis[spk_cnt] = np.sum(delays * np.exp(-beta*delays))

      # Get the integral.
      trial_integral = 1/beta*(1-np.exp(-beta*(trial_window[1]-spikes_x)))
      basis_integral += trial_integral.sum()
      d_int = (trial_window[1]-spikes_x) * np.exp(-beta*(trial_window[1]-spikes_x))
      delay_integral += d_int.sum()

    return X, basis_integral, delay_basis, delay_integral


  @classmethod
  def compare_nuisance_regressor_baseline(
      cls,
      spike_times_x,
      trial_length,
      nuisance_type,
      model_par,
      file_path=None):
    """Compare the nuisance baseline v.s. true baseline.

    We assume we regress y on x, so no y spike train input.

    Args:
      nuisance_type: Select nuisance type.
    """
    spikes = spike_times_x[5]

    if nuisance_type == 'triangle_kernel':
      sample_y = np.arange(0, trial_length, 0.005)
      nuisance, nuisance_integral, = cls.construct_regressors_triangle_kernel(
          [spikes], sample_y.reshape(1,-1), 0.12, verbose=False)
    elif nuisance_type == 'jitter_mean':
      sample_y = np.arange(0, trial_length, 0.005)
      nuisance, nuisance_integral, = cls.construct_regressors_jitter_mean(
          [spikes], sample_y.reshape(1,-1), trial_length, 0.06, verbose=False)
    elif nuisance_type == 'gaussian_kernel':
      sample_y = np.arange(0, trial_length, 0.005)
      nuisance, nuisance_integral, = cls.construct_regressors_gaussian_kernel(
          [spikes], sample_y.reshape(1,-1), 0.04, [0, trial_length], verbose=False)
    elif nuisance_type == 'square_kernel':
      sample_y = np.arange(0, trial_length, 0.005)
      nuisance, nuisance_integral, = cls.construct_regressors_square_kernel(
          [spikes], sample_y.reshape(1,-1), 0.05, [0, trial_length], verbose=False)

    # This part is copy-pasted from generator class
    # `generate_amarasingham_coupling_filter_spike_times`.
    np.random.seed(model_par['random_seed'])
    trial_length = model_par['trial_length']
    num_trials = model_par['num_trials']
    num_nodes = model_par['num_nodes']
    num_peaks = model_par['num_peaks']
    baseline = model_par['baseline']
    sigma = model_par['sigma']
    peaks = np.random.rand(num_peaks) * trial_length  # Uniform peak positions.
    peaks = np.sort(peaks)
    def intensity_func(t):
      """Intensity function with mixture of Laplacian's."""
      t = np.array(t)
      if np.ndim(t) == 0:
        sample_points = scipy.stats.laplace.pdf(t - peaks,
            loc=0, scale=sigma/np.sqrt(2))
        intensity = sample_points.sum() + baseline
      else:
        num_t = len(t)
        intensity = np.zeros(num_t)
        for i in range(num_t):
          sample_points = scipy.stats.laplace.pdf(t[i] - peaks,
              loc=0, scale=sigma/np.sqrt(2))
          intensity[i] = sample_points.sum() + baseline
      return intensity

    gs_kw = dict(width_ratios=[1], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(6, 2.5), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    # plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
    axs = [axs]
    ax = fig.add_subplot(axs[0])
    for b in spikes:
      plt.axvline(x=b, c='grey', lw=0.1)
    plt.axhline(y=0, c='grey', lw=0.1)
    plt.plot(spikes, np.zeros(len(spikes))-10, 'k+', ms=4, label='Spikes')
    plt.plot(sample_y, nuisance.reshape(-1), 'k', label='Nuisance regressor')
    plt.plot(sample_y, intensity_func(sample_y), 'g', label='True intensity')
    plt.xlabel('Time [s]')
    plt.ylabel('Firing rate [spikes/s]')
    # plt.legend(loc=(0, 1.03), ncol=3)
    plt.xlim(0, trial_length)
    plt.ylim(-15, 80)
    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      print('save figure:', file_path)
    plt.show()


  @classmethod
  def construct_regressors_jitter_mean(
      cls,
      spike_times_x,
      spike_times_y,
      trial_length,
      jitter_window_width,
      verbose=False):
    """x^jitter basis, which is the mean of x over jitter null distribution."""
    baseline_shift = 0  # by 1 spike count.
    num_trials = len(spike_times_x)
    num_spikes_x = [len(spikes) for spikes in spike_times_x]
    num_spikes_x = np.sum(num_spikes_x)
    num_spikes_y = [len(spikes) for spikes in spike_times_y]
    num_spikes_y = np.sum(num_spikes_y)

    x_cnt, bins = cls.bin_spike_times(
        spike_times_x, jitter_window_width, trial_length)
    y_cnt, _ = cls.bin_spike_times(
        spike_times_y, jitter_window_width, trial_length)
    x_cnt = x_cnt.astype(int).reshape(-1)
    y_cnt = y_cnt.astype(int).reshape(-1)
    basis_cnt = np.zeros(num_spikes_y)
    y_ind = 0
    for i, cnt in enumerate(y_cnt.astype(int)):
      basis_cnt[y_ind:(y_ind+cnt)] = x_cnt[i]
      y_ind += cnt
    # The integral is over the whole continuous basis, not just at y time points.
    basis_cnt = basis_cnt + baseline_shift
    basis = basis_cnt / jitter_window_width
    basis_integral = num_spikes_x
    basis_integral += baseline_shift/jitter_window_width*num_trials*trial_length

    if verbose:
      trial_id = 0
      plt.figure(figsize=[18, 2])
      for b in bins:
        plt.axvline(x=b, c='grey', lw=0.2)
      plt.axhline(y=0, c='grey', lw=0.2)
      plt.plot(spike_times_x[0], np.zeros(len(spike_times_x[0]))-10, 'k+')
      plt.plot(spike_times_y[0], basis[:len(spike_times_y[0])], 'b.', ms=0.2)

    return basis.reshape(num_spikes_y, 1), basis_integral


  @classmethod
  def construct_regressors_square_kernel(
      cls,
      spike_times_x,
      spike_times_y,
      kernel_width,
      trial_window,
      mean_norm=False,
      verbose=False):
    """x^jitter basis, which is the mean of x over jitter null distribution."""
    num_trials = len(spike_times_x)
    num_spikes_x = [len(spikes) for spikes in spike_times_x]
    num_spikes_x = np.sum(num_spikes_x)
    num_spikes_y = [len(spikes) for spikes in spike_times_y]
    num_spikes_y = np.sum(num_spikes_y)
    trial_length = trial_window[1] - trial_window[0]

    half_kernel_width = kernel_width / 2
    kernel_height = 1 / kernel_width
    basis = np.zeros(num_spikes_y)
    integral_error = 0
    spk_cnt = -1
    for r in range(num_trials):
      spikes_x = spike_times_x[r]
      spikes_y = spike_times_y[r]
      boundary_gaps_l = - trial_window[0] + spikes_x[spikes_x <=
          trial_window[0] + half_kernel_width]
      boundary_gaps_r = trial_window[1] - spikes_x[spikes_x >
          trial_window[1] - half_kernel_width]
      error_l = 0.5 - boundary_gaps_l / kernel_width
      error_r = 0.5 - boundary_gaps_r / kernel_width
      integral_error += error_l.sum() + error_r.sum()

      for spk_y in spikes_y:
        spk_cnt = spk_cnt + 1
        delays = spk_y - spikes_x
        delays = delays[(delays>-half_kernel_width) & (delays<=half_kernel_width)]
        if len(delays) == 0:
          continue
        basis[spk_cnt] = len(delays) * kernel_height

    basis_integral = num_spikes_x - integral_error
    if mean_norm:
      basis = basis - basis_integral / num_trials / trial_length
      basis_integral = 0

    if verbose:
      trial_id = 0
      plt.figure(figsize=[18, 2])
      for b in spike_times_x[0]:
        plt.axvline(x=b, c='grey', lw=0.2)
      plt.axhline(y=0, c='grey', lw=0.2)
      plt.plot(spike_times_x[0], np.zeros(len(spike_times_x[0]))-10, 'k+')
      plt.plot(spike_times_y[0], basis[:len(spike_times_y[0])], 'b')

    return basis.reshape(num_spikes_y, 1), basis_integral


  @classmethod
  def construct_regressors_triangle_kernel(
      cls,
      spike_times_x,
      spike_times_y,
      kernel_width,
      trial_window,
      mean_norm=False,
      verbose=False):
    """x^jitter basis, which is the mean of x over jitter null distribution."""
    num_trials = len(spike_times_x)
    num_spikes_x = [len(spikes) for spikes in spike_times_x]
    num_spikes_x = np.sum(num_spikes_x)
    num_spikes_y = [len(spikes) for spikes in spike_times_y]
    num_spikes_y = np.sum(num_spikes_y)
    trial_length = trial_window[1] - trial_window[0]
    half_kernel_width = kernel_width / 2
    kernel_height = 2 / kernel_width
    basis = np.zeros(num_spikes_y)
    integral_error = 0
    spk_cnt = -1
    for r in range(num_trials):
      spikes_x = spike_times_x[r]
      spikes_y = spike_times_y[r]
      # Spikes near left boundary and right boundary.
      boundary_gaps_l = - trial_window[0] + spikes_x[spikes_x <=
          trial_window[0] + half_kernel_width]
      boundary_gaps_r = trial_window[1] - spikes_x[spikes_x >
          trial_window[1] - half_kernel_width]
      error_l = np.square(half_kernel_width - boundary_gaps_l) * 2 / kernel_width**2
      error_r = np.square(half_kernel_width - boundary_gaps_r) * 2 / kernel_width**2
      integral_error += error_l.sum() + error_r.sum()

      for spk_y in spikes_y:
        spk_cnt = spk_cnt + 1
        delays = spk_y - spikes_x
        delays = delays[(delays>-half_kernel_width) & (delays<=half_kernel_width)]
        if len(delays) == 0:
          continue
        delays = np.abs(delays)
        sample_points = (half_kernel_width - delays) / (half_kernel_width**2)
        basis[spk_cnt] = np.sum(sample_points)

    basis_integral = num_spikes_x - integral_error
    if mean_norm:
      basis = basis - basis_integral / num_trials / trial_length
      basis_integral = 0

    if verbose:
      trial_id = 0
      plt.figure(figsize=[18, 2])
      for b in spike_times_x[0]:
        plt.axvline(x=b, c='grey', lw=0.2)
      plt.axhline(y=0, c='grey', lw=0.2)
      plt.plot(spike_times_x[0], np.zeros(len(spike_times_x[0]))-10, 'k+')
      plt.plot(spike_times_y[0], basis[:len(spike_times_y[0])], 'b')

    return basis.reshape(num_spikes_y, 1), basis_integral


  @classmethod
  def construct_regressors_laplacian_kernel(
      cls,
      spike_times_x,
      spike_times_y,
      kernel_width,
      verbose=False):
    """x^jitter basis, which is the mean of x over jitter null distribution."""
    num_trials = len(spike_times_x)
    num_spikes_x = [len(spikes) for spikes in spike_times_x]
    num_spikes_x = np.sum(num_spikes_x)
    num_spikes_y = [len(spikes) for spikes in spike_times_y]
    num_spikes_y = np.sum(num_spikes_y)

    effect_kernel_ratio = 3
    basis = np.zeros(num_spikes_y)
    spk_cnt = -1
    for r in range(num_trials):
      spikes_x = spike_times_x[r]
      spikes_y = spike_times_y[r]
      for spk_y in spikes_y:
        spk_cnt = spk_cnt + 1
        delays = spk_y - spikes_x
        delays = delays[(delays>-effect_kernel_ratio * kernel_width) &
                        (delays<=effect_kernel_ratio * kernel_width)]
        if len(delays) == 0:
          continue
        sample_points = scipy.stats.laplace.pdf(
            delays, loc=0, scale=kernel_width/np.sqrt(2))
        basis[spk_cnt] = np.sum(sample_points)

    window_integral = 1 - 2 * scipy.stats.laplace.cdf(
        - effect_kernel_ratio * kernel_width, scale=kernel_width/np.sqrt(2))
    basis_integral = num_spikes_x * window_integral

    if verbose:
      print('laplacian window integral', window_integral)
      trial_id = 0
      plt.figure(figsize=[18, 2])
      for b in spike_times_x[0]:
        plt.axvline(x=b, c='grey', lw=0.2)
      plt.axhline(y=0, c='grey', lw=0.2)
      plt.plot(spike_times_x[0], np.zeros(len(spike_times_x[0]))-10, 'k+')
      plt.plot(spike_times_y[0], basis[:len(spike_times_y[0])], 'b')

    return basis.reshape(num_spikes_y, 1), basis_integral


  @classmethod
  def construct_regressors_gaussian_kernel(
      cls,
      spike_times_x,
      spike_times_y,
      kernel_width,
      trial_window,
      mean_norm=False,
      verbose=False):
    """x^jitter basis, which is the mean of x over jitter null distribution."""
    num_trials = len(spike_times_x)
    num_spikes_x = [len(spikes) for spikes in spike_times_x]
    num_spikes_x = np.sum(num_spikes_x)
    num_spikes_y = [len(spikes) for spikes in spike_times_y]
    num_spikes_y = np.sum(num_spikes_y)
    trial_length = trial_window[1] - trial_window[0]

    effect_kernel_ratio = 5
    basis = np.zeros(num_spikes_y)
    spk_cnt = -1
    # Correct the basis integral outside the trial window.
    integral_error = 0
    for r in range(num_trials):
      spikes_x = spike_times_x[r]
      spikes_y = spike_times_y[r]
      # Spikes near left boundary and right boundary.
      boundary_gaps_l = - trial_window[0] + spikes_x[spikes_x <=
          trial_window[0] + effect_kernel_ratio * kernel_width]
      boundary_gaps_r = trial_window[1] - spikes_x[spikes_x >
          trial_window[1] - effect_kernel_ratio * kernel_width]
      error_l = scipy.stats.norm.cdf(-boundary_gaps_l, scale=kernel_width)
      error_r = scipy.stats.norm.cdf(-boundary_gaps_r, scale=kernel_width)
      integral_error += error_l.sum() + error_r.sum()

      for spk_y in spikes_y:
        spk_cnt = spk_cnt + 1
        delays = spk_y - spikes_x
        delays = delays[(delays>-effect_kernel_ratio * kernel_width) &
                        (delays<=effect_kernel_ratio * kernel_width)]
        if len(delays) == 0:
          continue
        sample_points = scipy.stats.norm.pdf(
            delays, loc=0, scale=kernel_width)
        basis[spk_cnt] = np.sum(sample_points)

    window_integral = 1 - 2 * scipy.stats.norm.cdf(-effect_kernel_ratio)
    # if mean_norm is False:
    basis_integral = num_spikes_x * window_integral - integral_error

    if mean_norm:
      basis = basis - basis_integral / num_trials / trial_length
      basis_integral = 0

    if verbose:
      print('gaussian window integral', window_integral)
      print('np.sum(basis):', np.sum(basis))
      plt.figure(figsize=[18, 2])
      for b in spike_times_x[0]:
        plt.axvline(x=b, c='grey', lw=0.2)
      plt.axhline(y=0, c='grey', lw=0.2)
      plt.plot(spike_times_x[0], np.zeros(len(spike_times_x[0]))-10, 'k+')
      plt.plot(spike_times_y[0], basis[:len(spike_times_y[0])], 'b')

    return basis.reshape(num_spikes_y, 1), basis_integral


  @classmethod
  def plot_kernels(
      cls,
      file_path=None):
    """Visualize the kernel shapes"""
    t = np.linspace(-0.15, 0.15, 500)
    spike_times_x = [np.array([0])]
    spike_times_y = [t]
    trial_window = [0, 5]

    kernel_width0 = 0.1
    y0, integral0 = cls.construct_regressors_square_kernel(
      spike_times_x, spike_times_y, kernel_width0, trial_window=trial_window)

    kernel_width1 = 0.2
    y1, integral1 = cls.construct_regressors_triangle_kernel(
      spike_times_x, spike_times_y, kernel_width1, trial_window=trial_window)

    kernel_width2 = 0.035
    y2, integral2 = cls.construct_regressors_gaussian_kernel(
      spike_times_x, spike_times_y, kernel_width2, trial_window=trial_window)

    kernel_width3 = 0.06
    y3, integral3 = cls.construct_regressors_laplacian_kernel(
      spike_times_x, spike_times_y, kernel_width3)

    y0 = y0.reshape(-1) / integral0
    plot_t = t * 1000

    fig, axs = plt.subplots(figsize=(16, 2.5), nrows=1, ncols=4)
    plt.subplots_adjust(hspace=0, wspace=0.2)
    ax = fig.add_subplot(axs[0])
    ax.tick_params(left=False, labelleft=False)
    plt.plot(plot_t, y0, 'k', lw=2)
    # plt.axvline(-50, color='grey', lw=0.5)
    # plt.axvline(50, color='grey', lw=0.5)
    plt.ylim(-1, 30)
    plt.xlabel('Time [ms]')
    plt.title(f'square kernel width = {kernel_width0}')

    ax = fig.add_subplot(axs[1])
    ax.tick_params(left=False, labelleft=False)
    plt.plot(plot_t, y1, 'k', lw=2)
    # plt.axvline(-50, color='grey', lw=0.5)
    # plt.axvline(50, color='grey', lw=0.5)
    plt.ylim(-1, 15)
    plt.xlabel('Time [ms]')
    plt.title(f'triangle kernel width = {kernel_width1}')

    ax = fig.add_subplot(axs[2])
    ax.tick_params(left=False, labelleft=False)
    plt.plot(plot_t, y2, 'k', lw=2)
    # plt.axvline(-50, color='grey', lw=0.5)
    # plt.axvline(50, color='grey', lw=0.5)
    plt.ylim(-1, 15)
    plt.xlabel('Time [ms]')
    plt.title(f'Gaussian kernel width = {kernel_width2}')

    ax = fig.add_subplot(axs[3])
    ax.tick_params(left=False, labelleft=False)
    plt.plot(plot_t, y3, 'k', lw=2)
    # plt.axvline(-50, color='grey', lw=0.5)
    # plt.axvline(50, color='grey', lw=0.5)
    plt.ylim(-1, 15)
    plt.xlabel('Time [ms]')
    plt.title(f'Laplace kernel width = {kernel_width3}')

    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      print('save figure:', file_path)
    plt.show()


  @classmethod
  def bivariate_continuous_time_coupling_filter_build_regressors(
      cls,
      spike_times_x,
      spike_times_y,
      trial_window,
      model_par,
      mean_norm=True):
    """Bivariate continuous-time PP-GLM."""
    num_trials = len(spike_times_y)

    # Cross-coupling filter.
    if 'filter_type' not in model_par or model_par['filter_type'] == 'none':
      num_spikes_y = [len(spikes) for spikes in spike_times_y]
      num_spikes_y = np.sum(num_spikes_y)
      X, basis_integral = np.empty([num_spikes_y,0]), np.empty(0)
    elif model_par['filter_type'] == 'bspline':
      X, basis_integral, tck = cls.construct_regressors_bspline(
          spike_times_x, spike_times_y, num_knots=model_par['num_knots'],
          filter_length=model_par['filter_length'],
          space_par=model_par['knot_space_par'],
          num_tail_drop=model_par['num_tail_drop'], verbose=False)
      model_par['tck'] = tck
    elif model_par['filter_type'] == 'square':
      X, basis_integral = cls.construct_regressors_square(
          spike_times_x, spike_times_y, filter_length=model_par['filter_length'],
          trial_length=None,  # trial_window[1]-trial_window[0],
          trial_window=trial_window, mean_norm=True, verbose=False)
    elif model_par['filter_type'] == 'const':
      X, basis_integral = cls.construct_regressors_const(
          spike_times_x, spike_times_y, trial_window, verbose=False)
    elif model_par['filter_type'] == 'exp':
      X, basis_integral = cls.construct_regressors_exp(
          spike_times_x, spike_times_y, model_par['filter_beta'],
          trial_length=None, trial_window=trial_window, verbose=False)

    # Set filter basis as offset.
    if ('filter_type' in model_par and
        model_par['filter_type'] != 'none' and
        'fix_filter' in model_par and model_par['fix_filter']):
      X_filter, basis_integral_filter = X.copy(), basis_integral.copy()
      num_spikes_y = [len(spikes) for spikes in spike_times_y]
      num_spikes_y = np.sum(num_spikes_y)
      X, basis_integral = np.empty([num_spikes_y,0]), np.empty(0)

    num_samples, num_basis = X.shape
    model_par['num_samples'] = num_samples
    model_par['num_basis'] = num_basis

    # Append nuisance variable.
    if 'jitter_mean' in model_par['append_nuisance']:
      jitter_window_width = model_par['jitter_window_width']
      # TODO: make trial_length into trial_window.
      nuisance, nuisance_integral = cls.construct_regressors_jitter_mean(
          spike_times_x, spike_times_y, trial_window[1], jitter_window_width)
      X = np.hstack([nuisance, X])
      basis_integral = np.hstack([nuisance_integral, basis_integral])
      num_basis = num_basis + 1
      model_par['num_basis'] = num_basis
    if 'square_kernel' in model_par['append_nuisance']:
      kernel_width = model_par['kernel_width']
      nuisance, nuisance_integral = cls.construct_regressors_square_kernel(
          spike_times_x, spike_times_y, kernel_width,
          trial_window=trial_window, mean_norm=mean_norm)
      X = np.hstack([nuisance, X])
      basis_integral = np.hstack([nuisance_integral, basis_integral])
      num_basis = num_basis + 1
      model_par['num_basis'] = num_basis
    if 'triangle_kernel' in model_par['append_nuisance']:
      kernel_width = model_par['kernel_width']
      nuisance, nuisance_integral = cls.construct_regressors_triangle_kernel(
          spike_times_x, spike_times_y, kernel_width,
          trial_window=trial_window, mean_norm=mean_norm)
      X = np.hstack([nuisance, X])
      basis_integral = np.hstack([nuisance_integral, basis_integral])
      num_basis = num_basis + 1
      model_par['num_basis'] = num_basis
    if 'laplacian_kernel' in model_par['append_nuisance']:
      kernel_width = model_par['kernel_width']
      nuisance, nuisance_integral = cls.construct_regressors_laplacian_kernel(
          spike_times_x, spike_times_y, kernel_width)
      X = np.hstack([nuisance, X])
      basis_integral = np.hstack([nuisance_integral, basis_integral])
      num_basis = num_basis + 1
      model_par['num_basis'] = num_basis
    if 'gaussian_kernel' in model_par['append_nuisance']:
      kernel_width = model_par['kernel_width']
      nuisance, nuisance_integral = cls.construct_regressors_gaussian_kernel(
          spike_times_x, spike_times_y, kernel_width,
          trial_window=trial_window, mean_norm=mean_norm)
      X = np.hstack([nuisance, X])
      basis_integral = np.hstack([nuisance_integral, basis_integral])
      num_basis = num_basis + 1
      model_par['num_basis'] = num_basis
    # This has to be the convention that the constant nuisance is
    # always at the beginning. It is related to the initialization.
    if 'const' in model_par['append_nuisance']:
      nuisance = np.ones([num_samples, 1])
      nuisance_integral = np.zeros(1) + (trial_window[1]-trial_window[0]) * num_trials
      X = np.hstack([nuisance, X])
      basis_integral = np.hstack([nuisance_integral, basis_integral])
      num_basis = num_basis + 1
      model_par['num_basis'] = num_basis

    basis_integral = basis_integral.reshape(-1,1)

    # Baseline constant offset.
    offset = model_par['const_offset']
    offset_integral = offset * (trial_window[1]-trial_window[0]) * num_trials
    # Set filter basis as offset.
    if ('filter_type' in model_par and model_par['filter_type'] != 'none' and
        'fix_filter' in model_par and model_par['fix_filter']):
      num_nuisance = len(model_par['append_nuisance'])
      beta_filter = model_par['beta_fix'][num_nuisance:]
      offset += X_filter @ beta_filter
      # Vec x Vec.
      offset_integral += np.dot(basis_integral_filter, beta_filter)[0]

    return num_basis, num_samples, X, basis_integral, offset, offset_integral


  # TODO: merge similar functions.
  @classmethod
  def bivariate_continuous_time_coupling_filter_self_coupling_build_regressors(
      cls,
      spike_times_x,
      spike_times_y,
      trial_window,
      model_par,
      mean_norm=True):
    """Bivariate continuous-time PP-GLM."""
    num_trials = len(spike_times_y)

    # Cross-coupling filter.
    if 'filter_type' not in model_par or model_par['filter_type'] == 'none':
      num_spikes_y = [len(spikes) for spikes in spike_times_y]
      num_spikes_y = np.sum(num_spikes_y)
      X, basis_integral = np.empty([num_spikes_y,0]), np.empty(0)
    elif model_par['filter_type'] == 'bspline':
      X, basis_integral, tck = cls.construct_regressors_bspline(
          spike_times_x, spike_times_y, num_knots=model_par['num_knots'],
          filter_length=model_par['filter_length'],
          space_par=model_par['knot_space_par'],
          num_tail_drop=model_par['num_tail_drop'], verbose=False)
      model_par['tck'] = tck
    elif model_par['filter_type'] == 'square':
      X, basis_integral = cls.construct_regressors_square(
          spike_times_x, spike_times_y, filter_length=model_par['filter_length'],
          trial_length=None,  # trial_window[1]-trial_window[0],
          trial_window=trial_window, mean_norm=True, verbose=False)
    elif model_par['filter_type'] == 'const':
      X, basis_integral = cls.construct_regressors_const(
          spike_times_x, spike_times_y, trial_window, verbose=False)
    elif model_par['filter_type'] == 'exp':
      X, basis_integral = cls.construct_regressors_exp(
          spike_times_x, spike_times_y, model_par['filter_beta'],
          trial_length=None, trial_window=trial_window, verbose=False)

    # Self coupling filter.
    if model_par['self_filter_type'] == 'square':
      X_filter, X_integral = cls.construct_regressors_square(
          spike_times_y, spike_times_y, filter_length=model_par['self_filter_length'],
          trial_length=None,  # assuming trial_window[1]-trial_window[0],
          trial_window=trial_window, mean_norm=True, verbose=False)
      X = np.hstack([X, X_filter])
      basis_integral = np.hstack([basis_integral, X_integral])

    num_samples, num_basis = X.shape
    model_par['num_samples'] = num_samples
    model_par['num_basis'] = num_basis

    # Append nuisance variable.
    if 'jitter_mean' in model_par['append_nuisance']:
      jitter_window_width = model_par['jitter_window_width']
      # TODO: make trial_length into trial_window.
      nuisance, nuisance_integral = cls.construct_regressors_jitter_mean(
          spike_times_x, spike_times_y, trial_window[1], jitter_window_width)
      X = np.hstack([nuisance, X])
      basis_integral = np.hstack([nuisance_integral, basis_integral])
      num_basis = num_basis + 1
      model_par['num_basis'] = num_basis
    if 'square_kernel' in model_par['append_nuisance']:
      kernel_width = model_par['kernel_width']
      nuisance, nuisance_integral = cls.construct_regressors_square_kernel(
          spike_times_x, spike_times_y, kernel_width,
          trial_window=trial_window, mean_norm=mean_norm)
      X = np.hstack([nuisance, X])
      basis_integral = np.hstack([nuisance_integral, basis_integral])
      num_basis = num_basis + 1
      model_par['num_basis'] = num_basis
    if 'triangle_kernel' in model_par['append_nuisance']:
      kernel_width = model_par['kernel_width']
      nuisance, nuisance_integral = cls.construct_regressors_triangle_kernel(
          spike_times_x, spike_times_y, kernel_width,
          trial_window=trial_window, mean_norm=mean_norm)
      X = np.hstack([nuisance, X])
      basis_integral = np.hstack([nuisance_integral, basis_integral])
      num_basis = num_basis + 1
      model_par['num_basis'] = num_basis
    if 'laplacian_kernel' in model_par['append_nuisance']:
      kernel_width = model_par['kernel_width']
      nuisance, nuisance_integral = cls.construct_regressors_laplacian_kernel(
          spike_times_x, spike_times_y, kernel_width)
      X = np.hstack([nuisance, X])
      basis_integral = np.hstack([nuisance_integral, basis_integral])
      num_basis = num_basis + 1
      model_par['num_basis'] = num_basis
    if 'gaussian_kernel' in model_par['append_nuisance']:
      kernel_width = model_par['kernel_width']
      nuisance, nuisance_integral = cls.construct_regressors_gaussian_kernel(
          spike_times_x, spike_times_y, kernel_width,
          trial_window=trial_window, mean_norm=mean_norm)
      X = np.hstack([nuisance, X])
      basis_integral = np.hstack([nuisance_integral, basis_integral])
      num_basis = num_basis + 1
      model_par['num_basis'] = num_basis
    # This has to be the convention that the constant nuisance is
    # always at the beginning. It is related to the initialization.
    if 'const' in model_par['append_nuisance']:
      nuisance = np.ones([num_samples, 1])
      nuisance_integral = np.zeros(1) + (trial_window[1]-trial_window[0]) * num_trials
      X = np.hstack([nuisance, X])
      basis_integral = np.hstack([nuisance_integral, basis_integral])
      num_basis = num_basis + 1
      model_par['num_basis'] = num_basis

    basis_integral = basis_integral.reshape(-1,1)

    # Baseline constant offset.
    offset = model_par['const_offset']
    offset_integral = offset * (trial_window[1]-trial_window[0]) * num_trials
    # Set filter basis as offset.
    if ('filter_type' in model_par and model_par['filter_type'] != 'none' and
        'fix_filter' in model_par and model_par['fix_filter']):
      num_nuisance = len(model_par['append_nuisance'])
      beta_filter = model_par['beta_fix'][num_nuisance:]
      offset += X_filter @ beta_filter
      # Vec x Vec.
      offset_integral += np.dot(basis_integral_filter, beta_filter)[0]

    return num_basis, num_samples, X, basis_integral, offset, offset_integral


  @classmethod
  def plot_bivariate_continuous_time_coupling_filter_regressors(
      cls,
      spike_times_x,
      spike_times_y,
      trial_window,
      dt=0.001,
      model_par=None,
      verbose=False):
    """Bivariate continuous-time PP-GLM."""
    num_trials = len(spike_times_y)
    T = trial_window[1]
    t_sample_single = np.arange(0, T, dt)
    t_sample = [t_sample_single] * num_trials
    (num_basis, num_samples, X, basis_integral, offset, offset_integral
        ) = cls.bivariate_continuous_time_coupling_filter_build_regressors(
        spike_times_x, t_sample, trial_window, model_par)
    print('X.shape', X.shape)
    beta = model_par['beta']
    lambda_hat = (X @ beta).reshape(-1)

    gs_kw = dict(width_ratios=[1], height_ratios=[1]*3)
    fig, axs = plt.subplots(figsize=(20, 4), gridspec_kw=gs_kw,
        nrows=3, ncols=1)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
    ax = fig.add_subplot(axs[0])
    plt.plot(X[:,1])
    plt.xlim(0, 8000)
    ax.tick_params(labelbottom=False)

    ax = fig.add_subplot(axs[1])
    if X.shape[1] > 2:
      plt.plot(X[:,2])
      plt.xlim(0, 8000)
    ax.tick_params(labelbottom=False)

    ax = fig.add_subplot(axs[2])
    plt.plot(lambda_hat)
    plt.xlim(0, 8000)
    plt.ylim(0)
    plt.show()


  @classmethod
  def bivariate_continuous_time_coupling_filter_regression(
      cls,
      spike_times_x,
      spike_times_y,
      trial_window,
      model_par=None,
      mute_warning=False,
      cache=None,
      verbose=False):
    """Do actual regression for kernel width search.
    Using init doesn't reduce the number of iterations a lot, e.g. 22 --> 15.
    """
    num_trials = len(spike_times_y)
    learning_rate = model_par['learning_rate']
    max_num_itrs = model_par['max_num_itrs']
    epsilon = model_par['epsilon']
    if cache is not None:
      num_basis, num_samples, X, basis_integral, offset, offset_integral = cache
    else:
      (num_basis, num_samples, X, basis_integral, offset, offset_integral
          ) = cls.bivariate_continuous_time_coupling_filter_build_regressors(
          spike_times_x, spike_times_y, trial_window, model_par, mean_norm=True)

    # Beta offset.
    if 'beta_offset' in model_par:
      beta_offset = model_par['beta_offset']
    else:
      beta_offset = 0

    # Initialize parameters.
    if 'beta_init' in model_par:
      beta = model_par['beta_init'].copy()
    elif model_par['append_nuisance'] == 'const':
      beta = np.zeros([num_basis, 1]) + 1
      # Mean FR.
      beta[0] = num_samples / num_trials / (trial_window[1]-trial_window[0])
    elif 'const' in model_par['append_nuisance']:
      beta = np.zeros([num_basis, 1]) + 0.1
      beta[0] = num_samples / num_trials / (trial_window[1]-trial_window[0])
    else:
      beta = np.zeros([num_basis, 1]) + 0.1

    beta_old = beta
    if verbose:
      print('num_trials', num_trials)
      print(f'X.shape {X.shape}, basis_integral.shape {basis_integral.shape},' +
            f'beta.shape{beta.shape}')

    if hasattr(tqdm, '_instances'):
      tqdm._instances.clear()
    if verbose:
      trange = tqdm(range(max_num_itrs), ncols=100, file=sys.stdout)
    else:
      trange =range(max_num_itrs)
    for itr in trange:
      lmbd = X @ (beta + beta_offset) + offset
      non_zero_ind = np.where(lmbd > 0)[0]
      lmbd_integral = basis_integral.T @ (beta + beta_offset) + offset_integral
      nll = cls.spike_times_neg_log_likelihood(lmbd[non_zero_ind], lmbd_integral)
      # gradient = - X[non_zero_ind].T @ (1 / lmbd[non_zero_ind]) + basis_integral
      # hessian = X[non_zero_ind].T @ (X[non_zero_ind] / np.square(lmbd[non_zero_ind]))
      vec = X[non_zero_ind] / lmbd[non_zero_ind]
      gradient = - np.sum(vec, axis=0, keepdims=True).T + basis_integral
      hessian = vec.T @ vec

      # Newton's method.
      try:
        delta = np.linalg.inv(hessian) @ gradient
      except np.linalg.LinAlgError:
        hessian = hessian + np.eye(hessian.shape[0])*0.01
        delta = np.linalg.inv(hessian) @ gradient
      # Gradient descent.
      # delta = 0.0001*gradient

      # The threshold is set as 5 because we know the range of the optimal beta
      # is around 10.
      if any(np.abs(learning_rate*delta) > 5):
        lr = learning_rate * 0.2
        learning_rate = max(lr, 0.001)
      beta = beta - learning_rate * delta

      # Check convergence.
      beta_err = np.sum(np.abs(beta_old - beta))
      if beta_err < epsilon:
        break
      beta_old = beta

    if not mute_warning and itr == max_num_itrs-1 and max_num_itrs > 1:
      warnings.warn(f'Reach max itrs {max_num_itrs}. Last err:{beta_err:.3e}')
      model_par['warnings'] = 'itr_max'

    model_par['beta'] = beta
    model_par['beta_hessian'] = hessian
    model_par['num_itrs'] = itr
    model_par['nll'] = nll[0,0]
    if verbose:
      print('gradient', gradient.reshape(-1))
      print('num itr', itr, nll, beta_err)
      print('beta', beta.reshape(-1))

    return model_par.copy()


  @classmethod
  def bivariate_continuous_time_coupling_filter_full_regression(
      cls,
      spike_times_x,
      spike_times_y,
      trial_window,
      model_par=None,
      mute_warning=False,
      cache=None,
      verbose=False):
    """Add self-coupling component so called FULL."""
    num_trials = len(spike_times_y)
    learning_rate = model_par['learning_rate']
    max_num_itrs = model_par['max_num_itrs']
    epsilon = model_par['epsilon']
    if cache is not None:
      num_basis, num_samples, X, basis_integral, offset, offset_integral = cache
    else:
      (num_basis, num_samples, X, basis_integral, offset, offset_integral
          ) = cls.bivariate_continuous_time_coupling_filter_self_coupling_build_regressors(
          spike_times_x, spike_times_y, trial_window, model_par, mean_norm=True)

    # Beta offset.
    if 'beta_offset' in model_par:
      beta_offset = model_par['beta_offset']
    else:
      beta_offset = 0

    # Initialize parameters.
    if 'beta_init' in model_par:
      beta = model_par['beta_init'].copy()
    elif model_par['append_nuisance'] == 'const':
      beta = np.zeros([num_basis, 1]) + 1
      # Mean FR.
      beta[0] = num_samples / num_trials / (trial_window[1]-trial_window[0])
    elif 'const' in model_par['append_nuisance']:
      beta = np.zeros([num_basis, 1]) + 0.1
      beta[0] = num_samples / num_trials / (trial_window[1]-trial_window[0])
    else:
      beta = np.zeros([num_basis, 1]) + 0.1

    beta_old = beta
    if verbose:
      print('num_trials', num_trials)
      print(f'X.shape {X.shape}, basis_integral.shape {basis_integral.shape},' +
            f'beta.shape{beta.shape}')

    if hasattr(tqdm, '_instances'):
      tqdm._instances.clear()
    if verbose:
      trange = tqdm(range(max_num_itrs), ncols=100, file=sys.stdout)
    else:
      trange =range(max_num_itrs)
    for itr in trange:
      lmbd = X @ (beta + beta_offset) + offset
      non_zero_ind = np.where(lmbd > 0)[0]
      lmbd_integral = basis_integral.T @ (beta + beta_offset) + offset_integral
      nll = cls.spike_times_neg_log_likelihood(lmbd[non_zero_ind], lmbd_integral)
      # gradient = - X[non_zero_ind].T @ (1 / lmbd[non_zero_ind]) + basis_integral
      # hessian = X[non_zero_ind].T @ (X[non_zero_ind] / np.square(lmbd[non_zero_ind]))
      vec = X[non_zero_ind] / lmbd[non_zero_ind]
      gradient = - np.sum(vec, axis=0, keepdims=True).T + basis_integral
      hessian = vec.T @ vec

      # Newton's method.
      try:
        delta = np.linalg.inv(hessian) @ gradient
      except np.linalg.LinAlgError:
        hessian = hessian + np.eye(hessian.shape[0])*0.01
        delta = np.linalg.inv(hessian) @ gradient
      # Gradient descent.
      # delta = 0.0001*gradient

      # The threshold is set as 5 because we know the range of the optimal beta
      # is around 10.
      if any(np.abs(learning_rate*delta) > 5):
        lr = learning_rate * 0.2
        learning_rate = max(lr, 0.001)
      beta = beta - learning_rate * delta

      # Check convergence.
      beta_err = np.sum(np.abs(beta_old - beta))
      if beta_err < epsilon:
        break
      beta_old = beta

    if not mute_warning and itr == max_num_itrs-1 and max_num_itrs > 1:
      warnings.warn(f'Reach max itrs {max_num_itrs}. Last err:{beta_err:.3e}')
      model_par['warnings'] = 'itr_max'

    model_par['beta'] = beta
    model_par['beta_hessian'] = hessian
    model_par['num_itrs'] = itr
    model_par['nll'] = nll[0,0]
    if verbose:
      print('gradient', gradient.reshape(-1))
      print('num itr', itr, nll, beta_err)
      print('beta', beta.reshape(-1))

    return model_par.copy()


  @classmethod
  def bivariate_continuous_time_coupling_filter_regression_block(
      cls,
      spike_times_x,
      spike_times_y,
      trial_window,
      model_par=None,
      mute_warning=False,
      cache=None,
      verbose=False):
    """Bivariate continuous-time PP-GLM."""
    num_trials = len(spike_times_y)
    eta = 1e-2

    learning_rate = model_par['learning_rate']
    max_num_itrs = model_par['max_num_itrs']
    epsilon = model_par['epsilon']

    # Basis design.
    if model_par['filter_type'] == 'bspline':
      X, basis_integral, tck = cls.construct_regressors_bspline(
          spike_times_x, spike_times_y, num_knots=model_par['num_knots'],
          filter_length=model_par['filter_length'],
          space_par=model_par['knot_space_par'],
          num_tail_drop=model_par['num_tail_drop'], verbose=False)
      model_par['tck'] = tck
    elif model_par['filter_type'] == 'square':
      X, basis_integral = cls.construct_regressors_square(
          spike_times_x, spike_times_y, filter_length=model_par['filter_length'],
          verbose=False)
    elif model_par['filter_type'] == 'none':
      num_spikes_y = [len(spikes) for spikes in spike_times_y]
      num_spikes_y = np.sum(num_spikes_y)
      X, basis_integral = np.empty([num_spikes_y,0]), np.empty(0)

    num_samples, num_basis = X.shape
    model_par['num_samples'] = num_samples
    model_par['num_basis'] = num_basis
    num_nuisance = 0
    # Block baseline for each trial.
    if 'block_triangle_kernel' in model_par['append_nuisance']:
      kernel_width = model_par['kernel_width']
      nuisance = [0] * num_trials
      nuisance_integral = np.zeros(num_trials)
      for r in range(num_trials):
        nuisance[r], nuisance_integral[r] = cls.construct_regressors_triangle_kernel(
            [spike_times_x[r]], [spike_times_y[r]], kernel_width)
      nuisance = scipy.linalg.block_diag(*nuisance)
      X = np.hstack([nuisance, X])
      basis_integral = np.hstack([nuisance_integral, basis_integral])
      num_basis = num_basis + num_trials
      num_nuisance += num_trials
      model_par['num_basis'] = num_basis
      model_par['num_nuisance'] = num_nuisance

    # This is designed for baseline fitting for each trial separately.
    if 'block_const' in model_par['append_nuisance']:
      nuisance = [np.ones([len(spk), 1]) for spk in spike_times_y]
      trial_fr = [len(spk) / (trial_window[1]-trial_window[0])
                  for spk in spike_times_y]
      nuisance_integral = np.zeros(num_trials) + trial_window[1]-trial_window[0]
      nuisance = scipy.linalg.block_diag(*nuisance)
      X = np.hstack([nuisance, X])
      basis_integral = np.hstack([nuisance_integral, basis_integral])
      num_basis = num_basis + num_trials
      num_nuisance += num_trials
      model_par['num_basis'] = num_basis
      model_par['num_nuisance'] = num_nuisance
      print(trial_fr)

    basis_integral = basis_integral.reshape(-1,1)
    # Sparsify the matrix.
    # print('X mem size before sparsify MB', X.nbytes/1000000)
    X = scipy.sparse.csr_matrix(X)
    offset = model_par['const_offset']
    beta_offset = 0
    offset_integral = offset * (trial_window[1]-trial_window[0]) * num_trials

    # Initialize parameters.
    if 'beta_init' in model_par:
      beta = model_par['beta_init'].copy()
    elif 'block_const' in model_par['append_nuisance']:
      beta = np.zeros([num_basis, 1]) + 0.1
      beta[:num_trials,0] = trial_fr

    beta_old = beta
    if verbose:
      print(f'X.shape {X.shape}\tX mem {X.data.nbytes / 1000000}MB\t'+
            f'basis_integral.shape {basis_integral.shape}\t' +
            f'beta.shape{beta.shape}')

    nll_list = np.zeros(max_num_itrs)
    trange = tqdm(range(max_num_itrs), ncols=100) if verbose else range(max_num_itrs)
    for itr in trange:
      lmbd = X @ (beta + beta_offset) + offset
      non_zero_ind = np.where(lmbd > 0)[0]
      lmbd_integral = basis_integral.T @ (beta + beta_offset) + offset_integral
      nll = cls.spike_times_neg_log_likelihood(lmbd[non_zero_ind], lmbd_integral)
      gradient = - X[non_zero_ind].T @ (1 / lmbd[non_zero_ind]) + basis_integral
      gradient[:num_nuisance] = gradient[:num_nuisance] + eta * beta[:num_nuisance]
      hessian = X[non_zero_ind].T @ (X[non_zero_ind] / np.square(lmbd[non_zero_ind]))
      hessian[:num_nuisance,:num_nuisance] = (
          hessian[:num_nuisance,:num_nuisance] + eta * np.eye(num_nuisance))

      try:
        delta = np.linalg.inv(hessian) @ gradient
      except np.linalg.LinAlgError:
        hessian = hessian + np.eye(hessian.shape[0])*0.01
        delta = np.linalg.inv(hessian) @ gradient
      # The threshold is set as 5 because we know the range of the optimal beta
      # is around 10.
      # if any(np.abs(learning_rate*delta) > 5):
      #   lr = learning_rate * 0.2
      #   learning_rate = max(lr, 0.001)
      beta = beta - learning_rate * delta

      # Check convergence.
      nll_list[itr] = nll
      beta_err = np.sum(np.abs(beta_old - beta))
      if beta_err < epsilon:
        break
      beta_old = beta

    if not mute_warning and itr == max_num_itrs-1 and max_num_itrs > 1:
      warnings.warn(f'Reach max itrs {max_num_itrs}. Last err:{beta_err:.3e}')
      model_par['warnings'] = 'itr_max'

    model_par['beta'] = beta
    model_par['beta_hessian'] = hessian
    model_par['num_itrs'] = itr
    model_par['nll'] = nll[0,0]
    if verbose:
      print('num itr', itr, nll, beta_err)
      # print('beta', beta.reshape(-1))
      plt.figure(figsize=[4, 2.5])
      plt.plot(nll_list[:itr])
      plt.show()

    return model_par.copy()


  @classmethod
  def bivariate_continuous_time_coupling_filter_regression_batch(
      cls,
      spike_times_x,
      spike_times_y,
      trial_window,
      model_par=None,
      verbose=False):
    """Batch training for bivariate continuous-time PP-GLM."""
    num_trials = len(spike_times_y)
    max_num_itrs = model_par['max_num_itrs']
    batch_size = model_par['batch_size']
    epsilon = model_par['epsilon'] * batch_size
    if 'random_seed' in model_par:
      random.seed(model_par['random_seed'])
    nll_list = np.zeros(max_num_itrs)
    nll_old = 1e20

    model_par['max_num_itrs'] = 18  # Quickly get to the optimal.
    trange = tqdm(range(max_num_itrs), ncols=100) if verbose else range(max_num_itrs)
    for itr in trange:
      if itr > 0:  # Batch training.
        model_par['max_num_itrs'] = 1
      if num_trials > batch_size:
        batch_ids = random.sample(range(num_trials), batch_size)
      else:
        batch_ids = list(range(num_trials))
      spike_times_x_batch = [spike_times_x[idx] for idx in batch_ids]
      spike_times_y_batch = [spike_times_y[idx] for idx in batch_ids]
      model_par = cls.bivariate_continuous_time_coupling_filter_regression(
          spike_times_x_batch, spike_times_y_batch, trial_window, model_par)
      model_par['beta_init'] = model_par['beta']
      nll_list[itr] = model_par['nll']
      if batch_size > num_trials:
        break

      # Check convergence.
      nll_err = np.sum(np.abs(nll_old - nll_list[itr]))
      # print(nll_err)
      if nll_err < epsilon:
        break
      nll_old = nll_list[itr]

    model_par['max_num_itrs'] = max_num_itrs
    model_par['num_itrs'] = itr
    if verbose:
      plt.figure(figsize=[3, 1.6])
      plt.plot(nll_list[:itr+1], '.:')
      plt.show()
    return model_par.copy()


  @classmethod
  def bivariate_continuous_time_exp_coupling_filter_regression(
      cls,
      spike_times_x,
      spike_times_y,
      trial_window,
      model_par=None,
      update_gamma_only=False,
      mute_warning=False,
      verbose=False):
    """Do actual regression for kernel width search.
    Using init doesn't reduce the number of iterations a lot, e.g. 22 --> 15.
    """
    num_trials = len(spike_times_y)
    learning_rate = model_par['learning_rate']
    max_num_itrs = model_par['max_num_itrs']
    epsilon = model_par['epsilon']
    num_nuisance = len(model_par['append_nuisance'])
    (num_basis, num_samples, X, basis_integral, offset, offset_integral
        ) = cls.bivariate_continuous_time_coupling_filter_build_regressors(
        spike_times_x, spike_times_y, trial_window, model_par, mean_norm=False)

    # Initialize parameters.
    if 'beta_init' in model_par:
      beta = model_par['beta_init'].copy()
    elif model_par['append_nuisance'] == 'const':
      beta = np.zeros([num_basis, 1]) + 1
      # Mean FR.
      beta[0] = num_samples / num_trials / (trial_window[1]-trial_window[0])
    elif 'const' in model_par['append_nuisance']:
      beta = np.zeros([num_basis, 1]) + 0.1
      beta[0] = num_samples / num_trials / (trial_window[1]-trial_window[0])
    else:
      beta = np.zeros([num_basis, 1]) + 0.1

    gamma = model_par['filter_beta']

    beta_old = beta
    if verbose:
      print('num_trials', num_trials)
      print(f'X.shape {X.shape}, basis_integral.shape {basis_integral.shape},' +
            f'beta.shape{beta.shape}')

    if hasattr(tqdm, '_instances'):
      tqdm._instances.clear()
    if verbose:
      trange = tqdm(range(max_num_itrs), ncols=80, file=sys.stdout)
    else:
      trange =range(max_num_itrs)
    for itr in trange:
      # As gamma is updated, need to rebuild X every time.
      # Don't need to rebuild all basis, only the exp basis.
      # model_par_tmp = model_par.copy()
      # model_par['filter_beta'] = gamma
      # (num_basis, num_samples, X, basis_integral, offset, offset_integral
      #     ) = cls.bivariate_continuous_time_coupling_filter_build_regressors(
      #     spike_times_x, spike_times_y, trial_window, model_par, mean_norm=False)

      (X_exp, X_exp_integral, delay_basis, delay_integral
          ) = cls.construct_regressors_exp_derivative(
          spike_times_x, spike_times_y, gamma, trial_window, verbose=False)
      X[:,num_nuisance] = X_exp.reshape(-1)
      basis_integral[num_nuisance,0] = X_exp_integral

      lmbd = X @ beta
      # print('lmbd neg', len(lmbd[lmbd<=0]))
      non_zero_ind = np.where(lmbd > 0)[0]
      lmbd_integral = basis_integral.T @ beta
      nll = cls.spike_times_neg_log_likelihood(lmbd[non_zero_ind], lmbd_integral)
      vec = X[non_zero_ind] / lmbd[non_zero_ind]
      gradient = - np.sum(vec, axis=0, keepdims=True).T + basis_integral
      hessian = vec.T @ vec

      # Newton's method.
      # try:
      #   delta = np.linalg.inv(hessian) @ gradient
      # except np.linalg.LinAlgError:
      #   hessian = hessian + np.eye(hessian.shape[0])*0.01
      #   delta = np.linalg.inv(hessian) @ gradient
      # Gradient descent.
      delta = np.zeros_like(gradient) if update_gamma_only else gradient

      # Gradient of gamma
      alpha_h = beta[num_nuisance,0]
      gamma_gradien_spk = alpha_h * delay_basis[non_zero_ind]/lmbd[non_zero_ind].reshape(-1)
      gamma_integral = -alpha_h/gamma*X_exp_integral + alpha_h/gamma*delay_integral
      gamma_gradien = gamma_gradien_spk.sum() + gamma_integral

      # The threshold is set as 5.
      if any(np.abs(learning_rate*delta) > 5):
        lr = learning_rate * 0.2
        learning_rate = max(lr, 0.001)

      beta = beta - learning_rate * delta
      gamma = gamma - 5*learning_rate*gamma_gradien
      print('---', gamma, gamma_gradien)
      if verbose:
        trange.set_description(f'gamma {gamma:.3f}')

      # Check convergence.
      beta_err = np.sum(np.abs(beta_old - beta))
      if beta_err < epsilon and not update_gamma_only:
        break
      beta_old = beta

    if not mute_warning and itr == max_num_itrs-1 and max_num_itrs > 1:
      warnings.warn(f'Reach max itrs {max_num_itrs}. Last err:{beta_err:.3e}')
      model_par['warnings'] = 'itr_max'

    model_par['beta'] = beta
    model_par['beta_hessian'] = hessian
    model_par['num_itrs'] = itr
    model_par['nll'] = nll[0,0]
    if verbose:
      print('gradient', gradient.reshape(-1))
      print('num itr', itr, nll, beta_err)
      print('beta', beta.reshape(-1))

    return model_par.copy()


  @classmethod
  def bivariate_continuous_time_coupling_filter_regression_inference_single(
      cls,
      model_par_hat,
      verbose=False):
    """Inference for the regression."""
    filter_length = model_par_hat['filter_length']
    num_nuisance = len(model_par_hat['append_nuisance'])

    if model_par_hat['filter_type'] == 'square':
      hessian = model_par_hat['beta_hessian']
      beta_cov = np.linalg.inv(hessian)
      beta_cov = beta_cov[num_nuisance, num_nuisance]
      h_var = beta_cov
      h_std = np.sqrt(h_var)
      h = model_par_hat['beta'][num_nuisance,0]
      # print(h, h_std)
      cdf = scipy.stats.norm.cdf(h, scale=h_std)
      one_side_p_val = min(cdf, 1 - cdf)
      two_side_p_val = one_side_p_val * 2

      if verbose:
        print(f'p-val:{two_side_p_val:.2e}')
      return two_side_p_val


  @classmethod
  def bivariate_continuous_time_coupling_filter_regression_inference(
      cls,
      model_par_list,
      verbose=False):
    """Inference for the regression."""
    num_models = len(model_par_list)
    p_vals = np.zeros(num_models)
    print(f'num_models:{num_models}')

    for m in range(num_models):
      p_vals[m] = cls.bivariate_continuous_time_coupling_filter_regression_inference_single(
          model_par_list[m])

    return p_vals


  @classmethod
  def bivariate_continuous_time_coupling_filter_nll(
      cls,
      spike_times_x,
      spike_times_y,
      trial_window,
      model_par=None,
      mute_warning=False,
      cache=None,
      verbose=False):
    """PP-GLM negative log-likelihood."""
    num_trials = len(spike_times_y)
    (num_basis, num_samples, X, basis_integral, offset, offset_integral
        ) = cls.bivariate_continuous_time_coupling_filter_build_regressors(
        spike_times_x, spike_times_y, trial_window, model_par, mean_norm=True)
    beta = model_par['beta']
    lmbd = X @ beta + offset
    non_zero_ind = np.where(lmbd > 0)[0]
    lmbd_integral = basis_integral.T @ beta + offset_integral
    nll = cls.spike_times_neg_log_likelihood(lmbd[non_zero_ind], lmbd_integral)
    return nll[0,0]


  @classmethod
  def bivariate_continuous_time_coupling_filter_bayesian(
      cls,
      spike_times_x,
      spike_times_y,
      trial_window,
      model_par=None,
      num_samples=100,
      sample_kernel=True,
      verbose=False):
    """Sampling inference."""
    beta_samples = []
    kernel_samples = []
    accept_cnt = 0

    # Initialization.
    model_par = cls.bivariate_continuous_time_coupling_filter_regression(
        spike_times_x, spike_times_y, trial_window, model_par, verbose=False)
    beta_prev = model_par['beta']
    kernel_width_prev = model_par['kernel_width']
    log_likelihood_prev = -model_par['nll']
    hessian = model_par['beta_hessian']
    beta_cov = np.linalg.inv(hessian) * 0.6

    for s in tqdm(range(num_samples), ncols=100, file=sys.stdout):
      # Draw candidate.
      # beta_cnd = beta_prev + np.random.randn(3)*0.1
      beta_cnd = np.random.multivariate_normal(
          beta_prev.reshape(-1), beta_cov).reshape(-1,1)
      if sample_kernel:
        kernel_width_cnd = kernel_width_prev + np.random.randn()*0.005
      else:
        kernel_width_cnd = kernel_width_prev
      model_par['kernel_width'] = kernel_width_cnd
      model_par['beta'] = beta_cnd

      log_likelihood_cnd = -cls.bivariate_continuous_time_coupling_filter_nll(
          spike_times_x, spike_times_y, trial_window, model_par)
      a = np.exp(log_likelihood_cnd - log_likelihood_prev)
      u = np.random.rand()
      if u < a:  # Accept the sample.
        beta_samples.append(beta_cnd[2][0])
        kernel_samples.append(kernel_width_cnd)
        beta_prev = beta_cnd
        kernel_width_prev = kernel_width_cnd
        log_likelihood_prev = log_likelihood_cnd
        accept_cnt += 1
      else:
        beta_samples.append(beta_prev[2][0])
        kernel_samples.append(kernel_width_prev)

    print(f'acceptance ratio {accept_cnt/num_samples:.3f}')
    return beta_samples, kernel_samples


  @classmethod
  def plot_beta_kernel_samples(
      cls,
      beta_samples,
      kernel_samples,
      model_par_hat,
      generator_par,
      selected_kernel_width=None,
      experiment_name=None,
      output_dir=None):
    """Plot the output of `bivariate_continuous_time_coupling_filter_bayesian`."""
    # Traces.
    plt.figure(figsize=[12,3])
    plt.subplot(121)
    plt.plot(beta_samples)
    plt.subplot(122)
    plt.plot(kernel_samples)
    plt.show()

    beta_cov = np.linalg.inv(model_par_hat['beta_hessian'])[2,2]
    beta_std = np.sqrt(beta_cov)
    beta_hat = model_par_hat['beta'][2]
    x = np.linspace(beta_hat-4*beta_std, beta_hat+6*beta_std,200)
    y = scipy.stats.norm.pdf(x, loc=model_par_hat['beta'][2], scale=beta_std)

    CI_left = np.quantile(beta_samples, 0.025)
    CI_right = np.quantile(beta_samples, 0.975)
    gkde=scipy.stats.gaussian_kde(beta_samples)
    xx = np.linspace(CI_left, CI_right, 201)
    yy = gkde.evaluate(xx)
    mode = xx[np.argmax(yy)]
    err_left = mode - CI_left
    err_right = CI_right - mode
    print(f'CI_left {CI_left:.2f}, CI_right {CI_right:.2f}, mode {mode:.2f}')

    plt.figure(figsize=[4,2])
    plt.axvline(generator_par['alpha'][0][1],
        color='green', lw=1, label='True value')
    seaborn.distplot(beta_samples, bins=13, color='grey', label='Bayesian')
    plt.plot(x, y, 'k', lw=2, label='Regression')
    plt.errorbar([mode], [np.max(y)*1.4], xerr=[[err_left], [err_right]],
       fmt='+k', capsize=5, label='95% CI')
    plt.ylim(0, np.max(y)*1.7)
    plt.xlabel(r'$\hat\alpha_{i\to j}$ [spikes/sec]')
    plt.legend(fontsize=8)

    if output_dir is not None:
      file_path = output_dir + experiment_name + 'hatalphaij_posterior.pdf'
      plt.savefig(file_path)
      print('save figure:', file_path)
    plt.show()

    if all(x==kernel_samples[0] for x in kernel_samples):
      return
    kernel_samples = np.array(kernel_samples)*1000
    CI_left = np.quantile(kernel_samples, 0.025)
    CI_right = np.quantile(kernel_samples, 0.975)
    gkde=scipy.stats.gaussian_kde(kernel_samples)
    xx = np.linspace(CI_left, CI_right, 201)
    yy = gkde.evaluate(xx)
    mode = xx[np.argmax(yy)]
    err_left = mode - CI_left
    err_right = CI_right - mode
    print(f'CI_left {CI_left:.2f}, CI_right {CI_right:.2f}, mode {mode:.2f}')

    plt.figure(figsize=[4,2])
    x, y = seaborn.distplot(kernel_samples,
        bins=13, color='grey').get_lines()[0].get_data()
    mode_id = np.argmax(y)
    mode = x[mode_id]
    plt.errorbar([mode], [np.max(y)*1.4], xerr=[[err_left], [err_right]],
       fmt='+k', capsize=5, label='95% CI')
    if selected_kernel_width is not None:
      plt.axvline(selected_kernel_width, color='k', lw=1, label='Optimal')
    plt.xlabel(r'$\hat\sigma_w$ [ms]')
    plt.xlim(mode-40, mode+55)
    plt.ylim(0, np.max(y)*1.7)
    plt.legend(fontsize=8)

    if output_dir is not None:
      file_path = output_dir + experiment_name + 'sigmaW_posterior.pdf'
      plt.savefig(file_path)
      print('save figure:', file_path)
    plt.show()



  @classmethod
  def extract_filter_single(
      cls,
      model_par_hat,
      dt=0.001,
      verbose=False):
    """Inference for the regression."""
    filter_length = model_par_hat['filter_length']
    num_nuisance = len(model_par_hat['append_nuisance'])

    if model_par_hat['filter_type'] == 'square':
      hessian = model_par_hat['beta_hessian']
      beta_cov = np.linalg.inv(hessian)
      beta_cov = beta_cov[num_nuisance, num_nuisance]
      h_var = beta_cov
      h_std = np.sqrt(h_var)
      h = model_par_hat['beta'][num_nuisance,0]
      # print(h, h_std)
      return h, h_std

    elif model_par_hat['filter_type'] == 'bspline':
      t, h, h_std = cls.reconstruct_basis(model_par_hat, dt=dt)
      return t, h, h_std


  @classmethod
  def extract_filters(
      cls,
      model_par_list,
      dt=0.001,
      center=0,
      verbose=False):
    """Inference for the regression."""
    num_models = len(model_par_list)

    if model_par_list[0]['filter_type'] == 'square':
      h_vals = np.zeros(num_models)
      for m in range(num_models):
        h_vals[m], _ = cls.extract_filter_single(model_par_list[m])
    elif model_par_list[0]['filter_type'] == 'bspline':
      filter_length = model_par_list[0]['filter_length']
      num_bins = np.round(filter_length/dt).astype(int)+1
      h_vals = np.zeros([num_models, num_bins])
      for m in range(num_models):
        t, h_vals[m], _ = cls.extract_filter_single(model_par_list[m], dt)

    if verbose and model_par_list[0]['filter_type'] == 'square':
      print(f'num_models:{num_models}')
      mu = np.mean(h_vals)
      sigma = np.std(h_vals)
      plt.figure(figsize=(7, 2.5))
      seaborn.distplot(h_vals, bins=40, kde=True, norm_hist=True, color='grey')
      plt.axvline(mu, c='k')
      # x = np.linspace(-2, 4, 1001)
      # y = scipy.stats.norm.pdf(x, loc=mu, scale=sigma)
      # plt.plot(x, y, 'r')
      # plt.xlim(-6*sigma+mu, 6*sigma+mu)
      plt.xlim(center-2.5, center+2.5)
      plt.title(f'Mean {mu:.2f}  std {sigma:.2f}')
      plt.show()
    elif verbose and model_par_list[0]['filter_type'] == 'bspline':
      print(f'num_models:{num_models}')
      plt.figure(figsize=(5, 2))
      plt.plot(t, h_vals.T, c='lightgrey', lw=0.2)
      plt.plot(t, h_vals.mean(axis=0), c='k', lw=1.5)
      plt.axhline(0, color='lightgrey', lw=0.5)
      plt.show()

    if model_par_list[0]['filter_type'] == 'square':
      return h_vals
    elif model_par_list[0]['filter_type'] == 'bspline':
      return t, h_vals


  @classmethod
  def reconstruct_bspline_basis(
      cls,
      model_par,
      plot_err_band=True,
      dt=0.001,
      ylim=None,
      verbose=False):
    """Re-build curve using bspline beta."""
    beta = model_par['beta']
    tck = model_par['tck']
    num_basis = model_par['num_basis']
    num_tail_drop = model_par['num_tail_drop']
    knots = tck[0]
    spline_degree = tck[2]
    if 'append_nuisance' in model_par:
      start_ind = len(model_par['append_nuisance'])
    else:
      start_ind = 0

    bspline_beta = beta[start_ind:num_basis].reshape(-1)
    t = np.linspace(knots[0], knots[-1], int(np.round(knots[-1]/dt)) + 1)
    distinct_knots = knots[spline_degree:-spline_degree]
    basis, basis_integral,_ = cls.bspline_basis(
        spline_degree+1, distinct_knots, t, num_tail_drop=num_tail_drop)

    h = basis @ bspline_beta.reshape(-1,1)
    h = h.reshape(-1)
    # AUC of the filter.
    filter_auc = basis_integral.T @ bspline_beta.reshape(-1,1)

    if plot_err_band:
      if 'beta_hessian' in model_par:
        hessian = model_par['beta_hessian']
        beta_cov = np.linalg.inv(hessian)
      elif 'beta_cov' in model_par:
        beta_cov = model_par['beta_cov']
      beta_cov = beta_cov[np.ix_(range(start_ind, num_basis),
                                 range(start_ind, num_basis))]
      h_var = basis @ beta_cov @ basis.T
      h_std = np.sqrt(np.diag(h_var))
    else:
      h_std = None

    axs = None
    if verbose:
      gs_kw = dict(width_ratios=[1], height_ratios=[1])
      fig, axs = plt.subplots(figsize=(5, 3), gridspec_kw=gs_kw,
          nrows=1, ncols=1)
      # plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
      axs = [axs]
      ax = fig.add_subplot(axs[0])
      # plt.axhline(y=0, c='lightgrey')
      # plt.axvline(x=0, c='lightgrey')
      plt.plot(t, h, c='tab:blue', label='Estimator')
      if plot_err_band:
        CI_scale = scipy.stats.norm.ppf(0.975)  # 95% CI.
        # cornflowerblue, dodgerblue
        plt.fill_between(t, h+h_std*CI_scale, h-h_std*CI_scale,
                         facecolor='dodgerblue', alpha=0.3, label='95% CB')
      plt.ylim(ylim)
      plt.legend()

    return axs, t, h, h_std


  @classmethod
  def reconstruct_basis(
      cls,
      model_par,
      dt=0.001):
    """Reconstruct basis for all kinds."""
    filter_type = model_par['filter_type']
    num_basis = model_par['num_basis']

    if filter_type == 'bspline':
      _,t,h,h_std = cls.reconstruct_bspline_basis(model_par, dt=dt)
    elif filter_type == 'square':
      filter_length = model_par['filter_length']
      t = np.array([0, filter_length])
      h = model_par['beta'][-1,0]
      h = np.array([h, h])
      if 'append_nuisance' in model_par:
        start_ind = len(model_par['append_nuisance'])
      else:
        start_ind = 0
      if 'beta_hessian' in model_par:
        hessian = model_par['beta_hessian']
        beta_cov = np.linalg.inv(hessian)
      elif 'beta_cov' in model_par:
        beta_cov = model_par['beta_cov']
      else:
        beta_cov = np.zeros((num_basis, num_basis))
      beta_cov = beta_cov[np.ix_(range(start_ind, num_basis),
                                 range(start_ind, num_basis))]
      h_var = beta_cov
      h_std = np.sqrt(h_var[0])

    elif filter_type == 'none':
      t = np.array([0, 100])
      h = np.array([0, 0])
      h_std = 0
    else:
      raise ValueError(f'Add new type of filter of: {filter_type}')

    return t, h, h_std


  @classmethod
  def plot_continuous_time_bivariate_regression_model_par(
      cls,
      model_par,
      filter_par=None,
      ylim=None,
      xlim=None,
      file_path=None):
    """Visualize estimated filters.

    Args:
      model_par: estimated model.
      filter_par: true model.
    """
    num_basis = model_par['num_basis']
    beta = model_par['beta']
    if 'beta_hessian' in model_par:
      hessian = model_par['beta_hessian']
    elif 'beta_cov' in model_par:
      beta_cov = model_par['beta_cov']
      hessian = np.linalg.inv(beta_cov)
    if 'filter_length' in model_par:
      filter_length = model_par['filter_length']

    if model_par['filter_type'] == 'bspline':
      axs, t, h, h_std = cls.reconstruct_bspline_basis(model_par, verbose=True)
      # knots = tck[0]
      # plt.plot(knots, np.zeros(len(knots)), 'rx')

    if model_par['filter_type'] == 'square':
      if 'append_nuisance' in model_par:
        num_nuisance = len(model_par['append_nuisance'])
        filter_beta = beta[num_nuisance,0]
        beta_cov = np.linalg.inv(hessian)
        beta_cov = beta_cov[num_nuisance, num_nuisance]
      else:
        filter_beta = beta[0]
        beta_cov = np.linalg.inv(hessian)
      t = [0, model_par['filter_length']]
      h = np.array([filter_beta, filter_beta])
      CI_scale = scipy.stats.norm.ppf(0.975)  # 95% CI.
      h_var = beta_cov
      h_std = np.sqrt(h_var)
      gs_kw = dict(width_ratios=[1], height_ratios=[1])
      fig, axs = plt.subplots(figsize=(5, 3), gridspec_kw=gs_kw,
          nrows=1, ncols=1)
      # plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
      ax = fig.add_subplot(axs)
      # plt.axhline(y=0, c='lightgrey')
      # plt.axvline(x=0, c='lightgrey')
      plt.plot(t, h, 'k', label='Estimator')
      plt.fill_between(t, h+h_std*CI_scale, h-h_std*CI_scale,
                       facecolor='lightgrey', alpha=0.3, label='95% CB')

    if model_par['filter_type'] == 'none':
      print('none filter type.')
      return

    # Add true filter if `filter_par` is provided..
    if filter_par is not None and filter_par['type'] == 'triangle':
      filter_alpha = filter_par['alpha'][0][1]
      filter_beta = filter_par['beta'][0][1]
      ax = axs[0]
      ax.plot([0, filter_beta], [filter_alpha, 0], 'g', label='True filter')
      plt.ylim(-filter_alpha*0.3, filter_alpha*1.3)

    if filter_par is not None and filter_par['type'] == 'square':
      filter_alpha = filter_par['alpha'][0][1]
      filter_beta = filter_par['beta'][0][1]
      # plt.plot([0, filter_beta], [filter_alpha, filter_alpha], 'g', label='True filter')
      t = np.linspace(0, 0.07, 20000)
      y = np.zeros_like(t)
      y[t<=filter_beta] = filter_alpha
      plt.plot(t, y, 'g', label='True filter')

      if ylim is None:
        plt.ylim(-filter_alpha*0.3, filter_alpha*1.3)
      else:
        plt.ylim(ylim)
      plt.xlim(0)

    if filter_par is not None and filter_par['type'] == 'exp':
      filter_alpha = filter_par['alpha'][0][1]
      filter_beta = filter_par['beta'][0][1]
      ax = axs[0]
      t = np.linspace(0, filter_length, 1001)
      y = filter_alpha * np.exp(-filter_beta * t)
      ax.plot(t, y, 'g', label='True filter')
      plt.ylim(-filter_alpha*0.3, filter_alpha*1.3)

    plt.legend(loc='best', ncol=3)
    xticks = np.arange(0, filter_length*2, 0.01)
    plt.xticks(xticks, xticks*1000)
    plt.xlabel('Lag [ms]', fontsize=12)
    plt.ylabel('Firing rate [spikes/sec]', fontsize=12)
    plt.ylim(ylim)
    if xlim is not None:
      plt.xlim(xlim)
    else:
      plt.xlim(0)
    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      print('save figure:', file_path)
      # plt.close()
    plt.show()


  @classmethod
  def pulication_plot_continuous_time_bivariate_regression_model_par(
      cls,
      model_par,
      filter_par=None,
      ylim=None,
      xlim=None,
      file_path=None):
    """Visualize estimated filters.

    Args:
      model_par: estimated model.
      filter_par: true model.
    """
    num_basis = model_par['num_basis']
    beta = model_par['beta']
    if 'beta_hessian' in model_par:
      hessian = model_par['beta_hessian']
    elif 'beta_cov' in model_par:
      beta_cov = model_par['beta_cov']
      hessian = np.linalg.inv(beta_cov)
    if 'filter_length' in model_par:
      filter_length = model_par['filter_length']

    if model_par['filter_type'] == 'bspline':
      axs, t, h, h_std = cls.reconstruct_bspline_basis(model_par, verbose=True)
      # knots = tck[0]
      # plt.plot(knots, np.zeros(len(knots)), 'rx')

    if model_par['filter_type'] == 'square':
      if 'append_nuisance' in model_par:
        num_nuisance = len(model_par['append_nuisance'])
        filter_beta = beta[num_nuisance,0]
        beta_cov = np.linalg.inv(hessian)
        beta_cov = beta_cov[num_nuisance, num_nuisance]
      else:
        filter_beta = beta[0]
        beta_cov = np.linalg.inv(hessian)
      t = [0, model_par['filter_length']]
      h = np.array([filter_beta, filter_beta])
      CI_scale = scipy.stats.norm.ppf(0.975)  # 95% CI.
      h_var = beta_cov
      h_std = np.sqrt(h_var)
      gs_kw = dict(width_ratios=[1], height_ratios=[1])
      fig, axs = plt.subplots(figsize=(5, 3), gridspec_kw=gs_kw,
          nrows=1, ncols=1)
      # plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
      ax = fig.add_subplot(axs)
      # plt.axhline(y=0, c='lightgrey')
      # plt.axvline(x=0, c='lightgrey')
      plt.plot(t, h, 'k', label='Estimator')
      plt.fill_between(t, h+h_std*CI_scale, h-h_std*CI_scale,
                       facecolor='lightgrey', alpha=0.3, label='95% CB')

    if model_par['filter_type'] == 'none':
      print('none filter type.')
      return

    # Add true filter if `filter_par` is provided..
    if filter_par is not None and filter_par['type'] == 'triangle':
      filter_alpha = filter_par['alpha'][0][1]
      filter_beta = filter_par['beta'][0][1]
      ax = axs[0]
      ax.plot([0, filter_beta], [filter_alpha, 0], 'g', label='True filter')
      plt.ylim(-filter_alpha*0.3, filter_alpha*1.3)

    if filter_par is not None and filter_par['type'] == 'square':
      filter_alpha = filter_par['alpha'][0][1]
      filter_beta = filter_par['beta'][0][1]
      # plt.plot([0, filter_beta], [filter_alpha, filter_alpha], 'g', label='True filter')
      t = np.linspace(0, 0.07, 20000)
      y = np.zeros_like(t)
      y[t<=filter_beta] = filter_alpha
      plt.plot(t, y, '--', color='tab:red', label='True filter')

      if ylim is None:
        plt.ylim(-filter_alpha*0.3, filter_alpha*1.3)
      else:
        plt.ylim(ylim)
      plt.xlim(0)

    if filter_par is not None and filter_par['type'] == 'exp':
      filter_alpha = filter_par['alpha'][0][1]
      filter_beta = filter_par['beta'][0][1]
      ax = axs[0]
      t = np.linspace(0, filter_length, 1001)
      y = filter_alpha * np.exp(-filter_beta * t)
      ax.plot(t, y, 'g', label='True filter')
      plt.ylim(-filter_alpha*0.3, filter_alpha*1.3)

    plt.legend(loc='best', ncol=3)
    xticks = np.arange(0, filter_length*2, 0.01)
    plt.xticks(xticks, xticks*1000)
    plt.xlabel('Lag [ms]', fontsize=12)
    plt.ylabel('Firing rate [spikes/sec]', fontsize=12)
    plt.ylim(ylim)
    if xlim is not None:
      plt.xlim(xlim)
    else:
      plt.xlim(0)
    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      print('save figure:', file_path)
      # plt.close()
    plt.show()


  @classmethod
  def plot_continuous_time_bivariate_regression_jitter_model_fitted_psth(
      cls,
      model_par,
      trial_window,
      spike_times_x,
      spike_times_y,
      generator_par=None,
      file_path=None):
    """Plot fitted PSTH and study baseline nuisance beta."""
    import hierarchical_model_generator
    generator = hierarchical_model_generator.HierarchicalModelGenerator()

    _=generator.plot_psth([spike_times_x, spike_times_y],
        generator_par['trial_length'], 0.005, ylim=[0, 100])

    (num_basis, num_samples, X, basis_integral, offset, offset_integral
        ) = cls.bivariate_continuous_time_coupling_filter_build_regressors(
        spike_times_x, spike_times_y, trial_window, model_par, mean_norm=True)

    spike_times_y = np.concatenate(spike_times_y)
    const = X[:,0]
    nuisance = X[:,1]

    plt.figure()
    # plt.plot(spike_times_y, np.ones(len(spike_times_y)))
    plt.plot(spike_times_y, nuisance, 'k.')

    print(np.mean(nuisance), )

    # # This part is copy-pasted from generator class
    # # `generate_amarasingham_coupling_filter_spike_times`.
    # np.random.seed(model_par['random_seed'])
    # trial_length = model_par['trial_length']
    # num_trials = model_par['num_trials']
    # num_nodes = model_par['num_nodes']
    # num_peaks = model_par['num_peaks']
    # baseline = model_par['baseline']
    # sigma = model_par['sigma']
    # peaks = np.random.rand(num_peaks) * trial_length  # Uniform peak positions.
    # peaks = np.sort(peaks)
    # def intensity_func(t):
    #   """Intensity function with mixture of Laplacian's."""
    #   t = np.array(t)
    #   if np.ndim(t) == 0:
    #     sample_points = scipy.stats.laplace.pdf(t - peaks,
    #         loc=0, scale=sigma/np.sqrt(2))
    #     intensity = sample_points.sum() + baseline
    #   else:
    #     num_t = len(t)
    #     intensity = np.zeros(num_t)
    #     for i in range(num_t):
    #       sample_points = scipy.stats.laplace.pdf(t[i] - peaks,
    #           loc=0, scale=sigma/np.sqrt(2))
    #       intensity[i] = sample_points.sum() + baseline
    #   return intensity


  @classmethod
  def construct_regressors_square_discrete(
      cls,
      spike_hist_x,
      dt,
      filter_length,
      trial_length=None,
      mean_norm=False,
      verbose=False):
    """Build coupling filter basis.

    The height of the filter is one, not like square nuisance regressor, where
    the kernel is a unit window.
    """
    num_trials, num_bins = spike_hist_x.shape
    num_spikes_x = np.sum(spike_hist_x)
    kernel = np.ones(int(filter_length/dt))
    basis = np.zeros_like(spike_hist_x)

    for r in range(num_trials):
      spikes_x = spike_hist_x[r]
      # Note the coupling filter starts from t=0, it is not centered at t=0.
      basis[r] = np.convolve(spikes_x, kernel, 'full')[:num_bins]

    # In continuous model, the triggering happens after the spike time.
    basis = np.roll(basis, 1, axis=1)
    basis[:,0] = 0
    basis = basis.reshape(-1, 1)
    # basis_integral = np.array([filter_length]) * num_spikes_x
    basis_integral = np.sum(basis) * dt
    # Mean normlization.
    if mean_norm:
      basis = basis - np.mean(basis)
      basis_integral = np.array([0])

    return basis, basis_integral


  @classmethod
  def construct_regressors_bspline_discrete(
      cls,
      spike_hist_x,
      dt,
      filter_par,
      mean_norm=False,
      verbose=False):
    """Build coupling filter basis.

    The height of the filter is one, not like square nuisance regressor, where
    the kernel is a unit window.
    """
    filter_length = filter_par['filter_length']
    num_trials, num_bins = spike_hist_x.shape
    num_spikes_x = np.sum(spike_hist_x)
    kernel = np.ones(int(filter_length/dt))
    kernel_t = np.arange(0, filter_length+dt, dt)
    kernel, _, _ = cls.construct_regressors_bspline(
        [[0]], [kernel_t], num_knots=filter_par['num_knots'],
        filter_length=filter_par['filter_length'],
        space_par=filter_par['knot_space_par'],
        num_tail_drop=filter_par['num_tail_drop'], verbose=False)

    basis_list = [0] * num_trials
    for r in range(num_trials):
      spikes_x = spike_hist_x[r].reshape(-1,1)
      # Note the coupling filter starts from t=0, it is not centered at t=0.
      # basis = np.convolve(spikes_x, kernel, 'full')[:num_bins]
      basis = scipy.signal.convolve2d(spikes_x, kernel, 'full')[0:num_bins+0]
      # If take the convolution from index 1, then no need to roll.
      # basis = np.roll(basis, 1, axis=0)
      # basis[0,:] = 0
      basis_list[r] = basis

    # In continuous model, the triggering happens after the spike time.
    basis = np.vstack(basis_list)
    basis_integral = np.sum(basis, axis=1) * dt  # Discrete calculation.

    # Basis integral used in continuous regression.
    # knots = cls.allocate_basis_knots(filter_par['num_knots'],
    #     filter_par['filter_length'], filter_par['knot_space_par'])
    # _, basis_integral, _ = cls.bspline_basis(4, knots, [0],
    #     num_tail_drop=filter_par['num_tail_drop'], verbose=verbose)
    # basis_integral = basis_integral * num_spikes_x

    # Mean normlization.
    if mean_norm:
      basis = basis - np.mean(basis, axis=0)
      basis_integral = np.array([0])

    return basis, basis_integral


  @classmethod
  def construct_regressors_triangle_kernel_discrete(
      cls,
      spike_hist_x,
      dt,
      kernel_width,
      trial_length,
      mean_norm=False,
      verbose=False):
    """x^jitter basis, which is the mean of x over jitter null distribution."""
    num_trials, num_bins = spike_hist_x.shape
    num_spikes_x = np.sum(spike_hist_x)
    half_kernel_width = kernel_width / 2
    num = int(kernel_width/dt)
    if num % 2 == 0:
      num = num + 1
    kernel_t = np.linspace(-half_kernel_width, half_kernel_width, num=num,
        endpoint=True)
    kernel = 1 / half_kernel_width - np.abs(kernel_t) / half_kernel_width**2
    kernel_half_len = int(half_kernel_width/dt)
    basis = np.zeros_like(spike_hist_x)
    for r in range(num_trials):
      spikes_x = spike_hist_x[r]
      basis[r] = np.convolve(spikes_x, kernel, 'full')[kernel_half_len:-kernel_half_len]

    basis = basis.reshape(-1, 1)
    basis_integral = np.sum(basis) * dt

    if mean_norm:
      basis = basis - basis_integral / num_trials / trial_length
      basis_integral = 0

    return basis, basis_integral


  @classmethod
  def construct_regressors_gaussian_kernel_discrete(
      cls,
      spike_hist_x,
      dt,
      kernel_width,
      trial_length,
      mean_norm=False,
      verbose=False):
    """x^jitter basis, which is the mean of x over jitter null distribution."""
    num_trials, num_bins = spike_hist_x.shape
    num_spikes_x = np.sum(spike_hist_x)

    effect_kernel_ratio = 5.0
    num = np.round(effect_kernel_ratio*kernel_width/dt).astype(int) * 2 + 1
    kernel_t = np.linspace(-effect_kernel_ratio*kernel_width,
        effect_kernel_ratio*kernel_width, num=num, endpoint=True)
    kernel = scipy.stats.norm.pdf(kernel_t, loc=0, scale=kernel_width)
    kernel_half_len = np.round(effect_kernel_ratio*kernel_width/dt).astype(int)
    basis = np.zeros_like(spike_hist_x)

    for r in range(num_trials):
      spikes_x = spike_hist_x[r]
      basis[r] = np.convolve(spikes_x, kernel, 'full')[kernel_half_len:-kernel_half_len]

    # basis = np.roll(basis, -1, axis=1)
    # basis[:,-1] = basis[:,-2]
    basis = basis.reshape(-1, 1)
    basis_integral = np.sum(basis) * dt

    if mean_norm:
      basis = basis - basis_integral / num_trials / trial_length
      # basis = basis - np.mean(basis)
      basis_integral = 0

    return basis, basis_integral


  @classmethod
  def bivariate_discrete_time_coupling_filter_build_regressors(
      cls,
      spike_hist_x,
      trial_window,
      model_par,
      mean_norm=True):
    """Bivariate continuous-time PP-GLM."""
    num_trials, num_bins = spike_hist_x.shape
    dt = model_par['dt']
    trial_length = trial_window[1]

    # Basis design.
    if 'filter_type' not in model_par or model_par['filter_type'] == 'none':
      num_bins = num_bins * num_trials
      X, basis_integral = np.empty([num_bins,0]), np.empty(0)
    elif model_par['filter_type'] == 'bspline':
      X, basis_integral = cls.construct_regressors_bspline_discrete(
          spike_hist_x, dt, filter_par=model_par, mean_norm=False, verbose=False)
    elif model_par['filter_type'] == 'square':
      X, basis_integral = cls.construct_regressors_square_discrete(
          spike_hist_x, dt, filter_length=model_par['filter_length'],
          trial_length=trial_length, mean_norm=True, verbose=False)

    num_samples, num_basis = X.shape
    model_par['num_samples'] = num_samples
    model_par['num_basis'] = num_basis

    # Append nuisance variable.
    if 'triangle_kernel' in model_par['append_nuisance']:
      kernel_width = model_par['kernel_width']
      nuisance, nuisance_integral = cls.construct_regressors_triangle_kernel_discrete(
          spike_hist_x, dt, kernel_width, trial_length=trial_length,
          mean_norm=mean_norm)
      X = np.hstack([nuisance, X])
      basis_integral = np.hstack([nuisance_integral, basis_integral])
      num_basis = num_basis + 1
      model_par['num_basis'] = num_basis
    if 'gaussian_kernel' in model_par['append_nuisance']:
      kernel_width = model_par['kernel_width']
      nuisance, nuisance_integral = cls.construct_regressors_gaussian_kernel_discrete(
          spike_hist_x, dt, kernel_width, trial_length=trial_length,
          mean_norm=mean_norm)
      X = np.hstack([nuisance, X])
      basis_integral = np.hstack([nuisance_integral, basis_integral])
      num_basis = num_basis + 1
      model_par['num_basis'] = num_basis
    # This has to be the convention that the constant nuisance is
    # always at the beginning. It is related to the initialization.
    if 'const' in model_par['append_nuisance']:
      nuisance = np.ones([num_samples, 1])
      nuisance_integral = np.zeros(1) + trial_length * num_trials
      X = np.hstack([nuisance, X])
      basis_integral = np.hstack([nuisance_integral, basis_integral])
      num_basis = num_basis + 1
      model_par['num_basis'] = num_basis

    basis_integral = basis_integral.reshape(-1,1)
    return num_basis, num_samples, X, basis_integral, 0, 0


  @classmethod
  def bivariate_discrete_time_coupling_filter_self_coupling_build_regressors(
      cls,
      spike_hist_x,
      spike_hist_y,
      trial_window,
      model_par,
      mean_norm=True):
    """Bivariate continuous-time PP-GLM."""
    num_trials, num_bins = spike_hist_x.shape
    dt = model_par['dt']
    trial_length = trial_window[1]

    # Basis design.
    if 'filter_type' not in model_par or model_par['filter_type'] == 'none':
      num_bins = num_bins * num_trials
      X, basis_integral = np.empty([num_bins,0]), np.empty(0)
    elif model_par['filter_type'] == 'bspline':
      X, basis_integral = cls.construct_regressors_bspline_discrete(
          spike_hist_x, dt, filter_par=model_par, mean_norm=False, verbose=False)
    elif model_par['filter_type'] == 'square':
      X, basis_integral = cls.construct_regressors_square_discrete(
          spike_hist_x, dt, filter_length=model_par['filter_length'],
          trial_length=trial_length, mean_norm=True, verbose=False)

    if model_par['self_filter_type'] == 'square':
      X_filter, X_integral = cls.construct_regressors_square_discrete(
          spike_hist_y, dt, filter_length=model_par['self_filter_length'],
          trial_length=trial_length, mean_norm=True, verbose=False)
      X = np.hstack([X, X_filter])
      basis_integral = np.hstack([basis_integral, X_integral])

    num_samples, num_basis = X.shape
    model_par['num_samples'] = num_samples
    model_par['num_basis'] = num_basis

    # Append nuisance variable.
    if 'triangle_kernel' in model_par['append_nuisance']:
      kernel_width = model_par['kernel_width']
      nuisance, nuisance_integral = cls.construct_regressors_triangle_kernel_discrete(
          spike_hist_x, dt, kernel_width, trial_length=trial_length,
          mean_norm=mean_norm)
      X = np.hstack([nuisance, X])
      basis_integral = np.hstack([nuisance_integral, basis_integral])
      num_basis = num_basis + 1
      model_par['num_basis'] = num_basis
    if 'gaussian_kernel' in model_par['append_nuisance']:
      kernel_width = model_par['kernel_width']
      nuisance, nuisance_integral = cls.construct_regressors_gaussian_kernel_discrete(
          spike_hist_x, dt, kernel_width, trial_length=trial_length,
          mean_norm=mean_norm)
      X = np.hstack([nuisance, X])
      basis_integral = np.hstack([nuisance_integral, basis_integral])
      num_basis = num_basis + 1
      model_par['num_basis'] = num_basis
    # This has to be the convention that the constant nuisance is
    # always at the beginning. It is related to the initialization.
    if 'const' in model_par['append_nuisance']:
      nuisance = np.ones([num_samples, 1])
      nuisance_integral = np.zeros(1) + trial_length * num_trials
      X = np.hstack([nuisance, X])
      basis_integral = np.hstack([nuisance_integral, basis_integral])
      num_basis = num_basis + 1
      model_par['num_basis'] = num_basis

    basis_integral = basis_integral.reshape(-1,1)
    return num_basis, num_samples, X, basis_integral, 0, 0


  @classmethod
  def bivariate_discrete_time_coupling_filter_regression(
      cls,
      spike_times_x,
      spike_times_y,
      trial_window,
      model_par,
      verbose=True):
    """Verify optimization values."""
    model_par = model_par.copy()
    dt = model_par['dt']
    learning_rate = model_par['learning_rate']
    num_itrs = model_par['max_num_itrs']
    epsilon = model_par['epsilon']
    num_trials = len(spike_times_y)
    num_spikes_x = [len(spikes) for spikes in spike_times_x]
    num_spikes_x = np.sum(num_spikes_x)
    num_spikes_y = [len(spikes) for spikes in spike_times_y]
    num_spikes_y = np.sum(num_spikes_y)
    T = trial_window[1]
    trial_length = trial_window[1]

    # Build from continuous model (slow). Used for test.
    # t_sample_single = np.arange(0, T, dt)
    # t_sample = [t_sample_single] * num_trials
    # (num_basis, num_samples, X, basis_integral, offset, offset_integral
    #     ) = cls.bivariate_continuous_time_coupling_filter_build_regressors(
    #     spike_times_x, t_sample, trial_window, model_par, mean_norm=True)
    # basis_integral_discrete = np.sum(X, axis=0, keepdims=True).T * dt
    # print('discrete integral   ', basis_integral_discrete.T)
    # print('continuous integral ', basis_integral.T)
    # X1 = X

    spike_hist_stacked_x, _ = cls.bin_spike_times(spike_times_x, dt, trial_length)
    (num_basis, num_samples, X, basis_integral, offset, offset_integral
        ) = cls.bivariate_discrete_time_coupling_filter_build_regressors(
        spike_hist_stacked_x, trial_window, model_par, mean_norm=True)
    basis_integral_discrete = np.sum(X, axis=0, keepdims=True).T * dt
    print('discrete integral   ', basis_integral_discrete.T)
    print('continuous integral ', basis_integral.T)

    spike_hist_stacked_y, _ = cls.bin_spike_times(spike_times_y, dt, trial_length)
    spike_hist_y = spike_hist_stacked_y.reshape(-1, 1)

    # Append smoothed y.
    # kernel_width = 2
    # nuisance_y, nuisance_integral_y = cls.construct_regressors_gaussian_kernel_discrete(
    #     spike_hist_stacked_y, dt, kernel_width, trial_length=trial_length,
    #     mean_norm=True)
    # print('nuisance_y.shape', nuisance_y.shape,
    #       'nuisance_integral_y', nuisance_integral_y)
    # num_basis += 1
    # X = np.hstack((X, nuisance_y))
    # basis_integral_discrete = np.vstack((basis_integral_discrete, nuisance_integral_y))

    if verbose:
      print(f'num spikes x {num_spikes_x} y {num_spikes_y}')
      print('X.shape', X.shape)
      print('continuous integral ', basis_integral.T)
      print('discrete integral   ', basis_integral_discrete.T)

    basis_integral = basis_integral_discrete
    beta = np.zeros([num_basis, 1])
    beta[0] = num_spikes_y / num_trials / (trial_window[1]-trial_window[0])
    beta_old = beta

    for itr in range(num_itrs):
      lambda_hat = X @ beta
      vec = (X * spike_hist_y) / lambda_hat
      gradient = - np.sum(vec, axis=0, keepdims=True).T
      gradient = gradient + basis_integral
      hessian = vec.T @ vec
      beta = beta - learning_rate * np.linalg.inv(hessian) @ gradient

      # Check convergence.
      beta_err = np.sum(np.abs(beta_old - beta))
      if beta_err < epsilon:
        break
      beta_old = beta

      if itr % 1 == 0 and verbose:
      # if itr % (num_itrs//10) == 0:
        lambda_hat = np.clip(lambda_hat, a_min=1e-8, a_max=None)
        nll = - np.sum(spike_hist_y * np.log(lambda_hat))
        nll += lambda_hat.sum() * dt
        print(f'itr{itr}\tnll: {nll:.1f}\tbeta: {beta.reshape(-1)}')

    plt.plot(lambda_hat)

    model_par['beta'] = beta
    return model_par


  @classmethod
  def bivariate_discrete_time_coupling_filter_regression_L2_loss(
      cls,
      spike_times_x,
      spike_times_y,
      trial_window,
      model_par,
      verbose=False):
    """Verify optimization values."""
    dt = model_par['dt']
    learning_rate = model_par['learning_rate']
    num_itrs = model_par['max_num_itrs']
    epsilon = model_par['epsilon']
    num_trials = len(spike_times_y)
    num_spikes_x = [len(spikes) for spikes in spike_times_x]
    num_spikes_x = np.sum(num_spikes_x)
    num_spikes_y = [len(spikes) for spikes in spike_times_y]
    num_spikes_y = np.sum(num_spikes_y)
    T = trial_window[1]
    trial_length = trial_window[1]
    t_sample_single = np.arange(0, T, dt)
    t_sample = [t_sample_single] * num_trials

    # Obtained from continuous construction.
    # (num_basis, num_samples, X, basis_integral, offset, offset_integral
    #     ) = cls.bivariate_continuous_time_coupling_filter_build_regressors(
    #     spike_times_x, t_sample, trial_window, model_par, mean_norm=True)

    spike_hist_stacked_x, _ = cls.bin_spike_times(spike_times_x, dt, trial_length)
    (num_basis, num_samples, X, basis_integral, offset, offset_integral
        ) = cls.bivariate_discrete_time_coupling_filter_build_regressors(
        spike_hist_stacked_x, trial_window, model_par, mean_norm=True)

    spike_hist_stacked, _ = cls.bin_spike_times(spike_times_y, dt, trial_length)
    spike_hist = spike_hist_stacked.reshape(-1, 1)
    beta = np.zeros([num_basis, 1])
    beta[0] = num_spikes_y / num_trials / (trial_window[1]-trial_window[0])
    beta_old = beta

    if verbose:
      print(f'num spikes x {num_spikes_x} y {num_spikes_y}')
      print('X.shape', X.shape, 'beta.shape', beta.shape)
      print()

    for itr in range(num_itrs):
      lambda_hat = (X @ beta) * dt
      gradient = X.T @ (lambda_hat - spike_hist)
      hessian = 2 * dt * X.T @ X
      beta = beta - learning_rate * np.linalg.inv(hessian) @ gradient
      # Check convergence.
      beta_err = np.sum(np.abs(beta_old - beta))
      if beta_err < epsilon:
        break
      beta_old = beta
      if itr % 1 == 0 and verbose:
      # if itr % (num_itrs//10) == 0:
        lambda_hat = (X @ beta)*dt
        loss = np.sum(np.square(spike_hist - lambda_hat))
        print(f'itr{itr}\tloss: {loss:.1f}\tbeta: {beta.reshape(-1)}')

    model_par['beta'] = beta
    model_par['beta_hessian'] = hessian
    model_par['num_itrs'] = itr
    lambda_hat = (X @ beta)*dt
    loss = np.sum(np.square(spike_hist - lambda_hat))
    model_par['loss'] = loss
    return model_par.copy()


  @classmethod
  def bivariate_discrete_time_coupling_filter_regression_bases(
      cls,
      spike_times_x,
      spike_times_y,
      trial_window,
      model_par,
      verbose=False):
    """Analyze bases."""
    dt = model_par['dt']
    trial_length = trial_window[1]
    num_trials = len(spike_times_y)
    num_spikes_x = [len(spikes) for spikes in spike_times_x]
    num_spikes_x = np.sum(num_spikes_x)
    num_spikes_y = [len(spikes) for spikes in spike_times_y]
    num_spikes_y = np.sum(num_spikes_y)

    # Obtained from continuous construction.
    # T = trial_window[1]
    # t_sample_single = np.arange(0, T, dt)
    # t_sample = [t_sample_single] * num_trials
    # (num_basis, num_samples, X, basis_integral, offset, offset_integral
    #     ) = cls.bivariate_continuous_time_coupling_filter_build_regressors(
    #     spike_times_x, t_sample, trial_window, model_par, mean_norm=True)

    spike_hist_stacked_x, _ = cls.bin_spike_times(spike_times_x, dt, trial_length)
    (num_basis, num_samples, X, basis_integral, offset, offset_integral
        ) = cls.bivariate_discrete_time_coupling_filter_build_regressors(
        spike_hist_stacked_x, trial_window, model_par, mean_norm=True)

    spike_hist_stacked, _ = cls.bin_spike_times(spike_times_y, dt, trial_length)
    spike_hist = spike_hist_stacked.reshape(-1, 1)
    XY = np.hstack((X, spike_hist))
    S = XY.T @ XY
    model_par['S_xx'] = S

    if verbose:
      print()
      print(f'num spikes x {num_spikes_x} y {num_spikes_y}')
      print('X.shape', X.shape, 'spike_hist.shape', spike_hist.shape)
      print(S)

    return model_par.copy()


  @classmethod
  def covariance_density(
      cls,
      spike_times,
      model_par=None,
      true_par=None,
      verbose=False):
    """Estimate covariance density."""
    num_trials = len(spike_times)
    num_spikes = [len(spikes) for spikes in spike_times]
    num_spikes = np.sum(num_spikes)
    trial_window = model_par['trial_window']
    dt = model_par['dt']
    trial_length = trial_window[1] - trial_window[0]
    # cls.spike_times_statistics(spike_times, trial_length, verbose=1)
    bar_lambda = num_spikes / num_trials / trial_length
    print(f'num_trials {num_trials}  trial_length {trial_length}')

    if true_par is not None:
      rho = true_par['rho']
      mu = true_par['mu']
      alpha_h = true_par['alpha'][0][1]
      beta_h = true_par['beta'][0][1]
      baseline = true_par['baseline']
      sigma_h = beta_h
      sigma_I = true_par['sigma']
      bar_lambda_i = mu + baseline
      bar_lambda_j = bar_lambda_i * (1 + alpha_h * sigma_h)
      t = np.linspace(-1, 1, 1000)
      cov_density_mod = scipy.stats.norm.pdf(t, loc=0, scale=np.sqrt(2)*sigma_I)
      cov_density_mod = cov_density_mod * mu**2 / rho

    spike_train, time_bins = cls.bin_spike_times(spike_times, dt, trial_length)
    # spike_train = scipy.ndimage.gaussian_filter1d(spike_train, 5)
    # plt.plot(time_bins, spike_train.mean(axis=0))

    # x = spike_train[0]
    # y = np.convolve(x, np.flip(x), 'same')
    # # plt.plot(x)
    # plt.plot(y)
    # return

    xcorr, bins = util.cross_prod(spike_train, spike_train)
    m2_density = xcorr.mean(axis=0) / dt
    bins = bins * dt
    cov_density = m2_density - bar_lambda*bar_lambda

    # plt.plot(bins, xcorr.mean(axis=0))
    plt.figure(figsize=[8, 3])
    plt.plot(bins, m2_density)
    # plt.plot(t, cov_density_mod)
    # plt.xlim(-1, 1)
    # plt.ylim(0.055, 0.063)
    # plt.show()




  @classmethod
  def analytical_L2_loss_no_coupling(
      cls,
      spike_times_x,
      spike_times_y,
      trial_window,
      dt,
      model_par,
      verbose=True):
    """Verify optimization values."""
    num_trials = len(spike_times_y)
    num_spikes_x = [len(spikes) for spikes in spike_times_x]
    num_spikes_x = np.sum(num_spikes_x)
    num_spikes_y = [len(spikes) for spikes in spike_times_y]
    num_spikes_y = np.sum(num_spikes_y)

    T = trial_window[1]
    trial_length = trial_window[1]
    t_sample_single = np.arange(0, T, dt)
    t_sample = [t_sample_single] * num_trials
    (num_basis, num_samples, X, basis_integral, offset, offset_integral
        ) = cls.bivariate_continuous_time_coupling_filter_build_regressors(
        spike_times_x, t_sample, trial_window, model_par)
    spike_hist_stacked, _ = cls.bin_spike_times(spike_times_y, dt, trial_length)
    spike_hist = spike_hist_stacked.reshape(-1, 1)

    x1 = X[:,1].reshape(-1, 1)
    mfrx = num_spikes_x / num_trials / T
    mfry = num_spikes_y / num_trials / T

    beta_1_numer = x1.T @ spike_hist - num_spikes_x * num_spikes_y / num_trials / T
    beta_1_denom = x1.T @ x1 * dt - num_spikes_x * num_spikes_x / num_trials / T
    beta_1 = beta_1_numer / beta_1_denom
    beta_0 = mfry - beta_1[0,0] * mfrx
    beta = np.array([beta_0, beta_1[0,0]])
    print('beta:', beta)
    return beta


  @classmethod
  def basis_test(
      cls,
      spike_times_x,
      spike_times_y,
      trial_window,
      dt,
      model_par,
      verbose=True):
    """Verify optimization values."""
    num_trials = len(spike_times_y)
    num_spikes_x = [len(spikes) for spikes in spike_times_x]
    num_spikes_x = np.sum(num_spikes_x)
    num_spikes_y = [len(spikes) for spikes in spike_times_y]
    num_spikes_y = np.sum(num_spikes_y)

    T = trial_window[1]
    trial_length = trial_window[1]
    t_sample_single = np.arange(0, T, dt)
    t_sample = [t_sample_single] * num_trials
    (num_basis, num_samples, X, basis_integral, offset, offset_integral
        ) = cls.bivariate_continuous_time_coupling_filter_build_regressors(
        spike_times_x, t_sample, trial_window, model_par)
    spike_hist_stacked, _ = cls.bin_spike_times(spike_times_y, dt, trial_length)
    spike_hist = spike_hist_stacked.reshape(-1, 1)
    x1 = X[:,1].reshape(-1, 1)

    print(spike_hist.T @ spike_hist)
    x = spike_hist - np.sum(spike_hist)
    print(x.T @ x)


  @classmethod
  def plot_filter_estimator_asymptotic(
      cls,
      h_true,
      file_path_list,
      num_trials,
      file_path=None):
    """Plot the stacked distribution of estimators."""
    num_scenarios = len(file_path_list)
    y = np.linspace(-10, 10, 501)
    ybins = np.linspace(h_true-1.5, h_true+1.5, 51)
    bin_width = ybins[1] - ybins[0]

    gs_kw = dict(width_ratios=[1], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(7, 2.5), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    # plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
    axs = [axs]
    ax = fig.add_subplot(axs[0])
    plot_gap = 5
    for s in range(num_scenarios):
      model_par_list = util.load_variable(file_path_list[s])
      h_vals = cls.extract_filters(model_par_list, verbose=False)
      # x_offset = num_trials[s]
      x_offset = s*plot_gap

      xhist, _ = np.histogram(h_vals, ybins)
      xhist = xhist * bin_width
      plt.barh(ybins[1:]-0.5*bin_width, xhist, height=bin_width, left=x_offset,
          color='lightgrey')

      mean = np.mean(h_vals)
      std = np.std(h_vals)
      std_theoretical = 5/np.sqrt(num_trials[s])

      gkde=scipy.stats.gaussian_kde(h_vals)
      x = gkde.evaluate(y)
      plt.axvline(x_offset, lw=1, c='lightgrey')
      x_mean = gkde.evaluate(mean)

      if s == 0:
        plt.plot(x + x_offset, y, 'k', lw=1, label='Histogram')
        plt.plot(x_offset-0.2, mean, 'k+', ms=10, label='Mean')
        plt.plot([x_offset-0.6, x_offset-0.6], [mean-std, mean+std], 'b', lw=3,
            label='SE')
        plt.plot([x_offset-0.4, x_offset-0.4],
            [mean-std_theoretical, mean+std_theoretical],
            'lime', lw=3, label=r'$c/\sqrt{\mathrm{sample\; size}}$')
      else:
        plt.plot(x + x_offset, y, 'k', lw=1, label=None)
        plt.plot(x_offset-0.2, mean, 'k+', ms=10, label=None)
        plt.plot([x_offset-0.6, x_offset-0.6], [mean-std, mean+std], 'b', lw=3,
            label=None)
        plt.plot([x_offset-0.4, x_offset-0.4],
            [mean-std_theoretical, mean+std_theoretical],
            'lime', lw=3, label=None)

    plt.axhline(h_true, c='grey', lw=1)
    plt.text(-0.07, 0.48, 'True', color='k', size=10, transform=ax.transAxes)
    plt.legend(ncol=1, loc=(1.02, 0.5))
    plt.xticks(np.arange(0, num_scenarios*plot_gap, plot_gap), num_trials)
    # plt.yticks([-1,1,3], [-1,1,3])
    plt.yticks([ ], [ ])
    plt.ylim(h_true-1.5, h_true+1.5)
    plt.xlabel('Sample size [trials]')
    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      print('save figure:', file_path)
      # plt.close()
    plt.show()


  @classmethod
  def erf(
      cls,
      x):
    """Gaussian error function.

    Test cases using Taylor expansion.
    x = 0.5
    y = scipy.special.erf(x)
    z = 
    y_approx = 2/np.sqrt(np.pi) * x
    print(y, y_approx)
    """
    return scipy.special.erf(x)


  @classmethod
  def bias_theoretical(
      cls,
      par):
    """Analytical bias."""
    if (par['loss'] is not None and par['filter_type'] == 'square' and
        par['append_nuisance'][1] == 'gaussian_kernel' and
        par['background_window'] == 'gaussian'):
      rho = par['rho']
      mu = par['mu']
      baseline = par['baseline']
      bar_lambda_i = mu + baseline
      sigma_h = par['filter_length']
      sigma_w = par['kernel_width']
      sigma_I = par['background_window_sigma']

      S_ww = mu**2/(2*rho*np.sqrt(np.pi)*np.sqrt(sigma_w**2 + sigma_I**2)) + \
          bar_lambda_i/(2*np.sqrt(np.pi)*sigma_w)
      S_hy = mu**2/(2*rho)*cls.erf(sigma_h/2/sigma_I)
      S_hw = mu**2/(2*rho)*cls.erf(sigma_h/2/np.sqrt(sigma_w**2/2+sigma_I**2)) + \
          bar_lambda_i/(2)*cls.erf(sigma_h/np.sqrt(2)/sigma_w)
      S_wy = mu**2/(2*rho*np.sqrt(np.pi)*np.sqrt(sigma_w**2/2 + sigma_I**2))
      S_hh = mu**2/rho*(sigma_h*cls.erf(sigma_h/2/sigma_I) -
            2*sigma_I/np.sqrt(np.pi)*(1-np.exp(- sigma_h**2/4/sigma_I**2))
          ) + bar_lambda_i*sigma_h
      Q = S_ww*S_hy - S_hw*S_wy
      R = S_ww*S_hh - S_hw*S_hw
      bias = Q / R
      return bias, Q, R


  @classmethod
  def bias_theoretical_boundary_correction(
      cls,
      par):
    """Analytical bias."""
    if (par['loss'] is not None and par['filter_type'] == 'square' and
        par['append_nuisance'][1] == 'gaussian_kernel' and
        par['background_window'] == 'gaussian'):
      rho = par['rho']
      mu = par['mu']
      trial_length = par['trial_length']
      baseline = par['baseline']
      bar_lambda_i = mu + baseline
      sigma_h = par['filter_length']
      sigma_w = par['kernel_width']
      sigma_I = par['background_window_sigma']

      S_ww = mu**2/(2*rho*np.sqrt(np.pi)*np.sqrt(sigma_w**2 + sigma_I**2)) + \
          bar_lambda_i/(2*np.sqrt(np.pi)*sigma_w)
      S_hy = mu**2/(2*rho)*cls.erf(sigma_h/2/sigma_I)
      S_hw = mu**2/(2*rho)*cls.erf(sigma_h/2/np.sqrt(sigma_w**2/2+sigma_I**2)) + \
          bar_lambda_i/(2)*cls.erf(sigma_h/np.sqrt(2)/sigma_w)
      S_wy = mu**2/(2*rho*np.sqrt(np.pi)*np.sqrt(sigma_w**2/2 + sigma_I**2))
      S_hh = mu**2/rho*(sigma_h*cls.erf(sigma_h/2/sigma_I) -
            2*sigma_I/np.sqrt(np.pi)*(1-np.exp(- sigma_h**2/4/sigma_I**2))
          ) + bar_lambda_i*sigma_h

      # Boundary correction.
      S_ww = S_ww * (trial_length)
      S_hy = S_hy * (trial_length)
      S_hw = S_hw * (trial_length)
      S_wy = S_wy * (trial_length)
      S_hh = S_hh * (trial_length)

      Q = S_ww*S_hy - S_hw*S_wy
      R = S_ww*S_hh - S_hw*S_hw
      bias = Q / R
      return bias, Q, R


  @classmethod
  def var_theoretical(
      cls,
      par):
    """Analytical bias."""
    if (par['loss'] is not None and par['filter_type'] == 'square' and
        par['append_nuisance'][1] == 'gaussian_kernel' and
        par['background_window'] == 'gaussian'):
      rho = par['rho']
      mu = par['mu']
      baseline = par['baseline']
      trial_length = par['trial_length']
      num_trials = par['num_trials']
      sigma_h = par['filter_length']
      sigma_w = par['kernel_width']
      sigma_I = par['background_window_sigma']
      alpha_h = par['alpha'][0][1]
      beta_h = par['beta'][0][1]

      T = trial_length * num_trials
      bar_lambda_i = mu + baseline
      bar_lambda_j = bar_lambda_i * (1 + alpha_h * sigma_h)

      S_ww = mu**2/(2*rho*np.sqrt(np.pi)*np.sqrt(sigma_w**2 + sigma_I**2)) + \
          bar_lambda_i/(2*np.sqrt(np.pi)*sigma_w)
      S_hy = mu**2/(2*rho)*cls.erf(sigma_h/2/sigma_I)
      S_hw = mu**2/(2*rho)*cls.erf(sigma_h/2/np.sqrt(sigma_w**2/2+sigma_I**2)) + \
          bar_lambda_i/(2)*cls.erf(sigma_h/np.sqrt(2)/sigma_w)
      S_wy = mu**2/(2*rho*np.sqrt(np.pi)*np.sqrt(sigma_w**2/2 + sigma_I**2))
      S_hh = mu**2/rho*(sigma_h*cls.erf(sigma_h/2/sigma_I) -
            2*sigma_I/np.sqrt(np.pi)*(1-np.exp(- sigma_h**2/4/sigma_I**2))
          ) + bar_lambda_i*sigma_h
      Q = bar_lambda_j*S_ww
      R = S_ww*S_hh - S_hw*S_hw
      var = Q / R / T
      return var


  @classmethod
  def bias_theoretical_extreme_cases(
      cls,
      par):
    """Calculate the extreme cases for verification."""
    if (par['loss'] is not None and par['filter_type'] == 'square' and
        par['append_nuisance'][1] == 'gaussian_kernel' and
        par['background_window'] == 'gaussian'):
      rho = par['rho']
      mu = par['mu']
      baseline = par['baseline']
      trial_length = par['trial_length']
      num_trials = par['num_trials']
      sigma_h = par['filter_length']
      sigma_I = par['background_window_sigma']

      T = trial_length * num_trials
      bar_lambda_i = mu + baseline
      bar_lambda_j = bar_lambda_i * (1 + sigma_h)

      S_hh = mu**2/rho*(sigma_h*cls.erf(sigma_h/2/sigma_I) -
            2*sigma_I/np.sqrt(np.pi)*(1-np.exp(- sigma_h**2/4/sigma_I**2))
          ) + bar_lambda_i*sigma_h
      S_hlam = mu**2/(2*rho)*cls.erf(sigma_h/2/sigma_I)

      # As sigma_w --> 0 or infty, the limits are the same.
      # bias = mu**2/(rho*sigma_I) / (mu**2*sigma_h/(rho*sigma_I) +
      #                               2*np.sqrt(np.pi)*bar_lambda_i)
      bias = S_hlam / S_hh
      return bias


  @classmethod
  def var_theoretical_extreme_cases(
      cls,
      par):
    """Calculate the extreme cases for verification."""
    if (par['loss'] is not None and par['filter_type'] == 'square' and
        par['append_nuisance'][1] == 'gaussian_kernel' and
        par['background_window'] == 'gaussian'):
      rho = par['rho']
      mu = par['mu']
      num_trials = par['num_trials']
      trial_length = par['trial_length']
      baseline = par['baseline']
      sigma_h = par['filter_length']
      sigma_w = par['kernel_width']
      sigma_I = par['background_window_sigma']

      T = trial_length * num_trials
      bar_lambda_i = mu + baseline
      bar_lambda_j = bar_lambda_i * (1 + sigma_h)

      S_ww = mu**2/(2*rho*np.sqrt(np.pi)*np.sqrt(sigma_w**2 + sigma_I**2)) + \
          bar_lambda_i/(2*np.sqrt(np.pi)*sigma_w)
      S_hy = mu**2/(2*rho)*cls.erf(sigma_h/2/sigma_I)
      S_hw = mu**2/(2*rho)*cls.erf(sigma_h/2/np.sqrt(sigma_w**2/2+sigma_I**2)) + \
          bar_lambda_i/(2)*cls.erf(sigma_h/np.sqrt(2)/sigma_w)
      S_wy = mu**2/(2*rho*np.sqrt(np.pi)*np.sqrt(sigma_w**2/2 + sigma_I**2))
      S_hh = mu**2/rho*(sigma_h*cls.erf(sigma_h/2/sigma_I) -
            2*sigma_I/np.sqrt(np.pi)*(1-np.exp(- sigma_h**2/4/sigma_I**2))
          ) + bar_lambda_i*sigma_h

      # As sigma_w --> 0 or infty, the limits are the same.
      # var = bar_lambda_i / (mu**2*sigma_h**2/(rho*sigma_I*2*np.sqrt(np.pi))
      #     + bar_lambda_i*sigma_h)
      var = bar_lambda_i / S_hh
      var = var / T

      return var


  @classmethod
  def log_likelihood_theoretical(
      cls,
      par):
    """Analytical bias."""
    if (par['loss'] is not None and par['filter_type'] == 'square' and
        par['append_nuisance'][1] == 'gaussian_kernel' and
        par['background_window'] == 'gaussian'):
      rho = par['rho']
      mu = par['mu']
      baseline = par['baseline']
      trial_length = par['trial_length']
      num_trials = par['num_trials']
      sigma_h = par['filter_length']
      sigma_w = par['kernel_width']
      sigma_I = par['background_window_sigma']
      alpha_h = par['alpha'][0][1]
      beta_h = par['beta'][0][1]

      T = trial_length * num_trials
      bar_lambda_i = mu + baseline
      bar_lambda_j = bar_lambda_i * (1 + alpha_h * sigma_h)

      S_ww = mu**2/(2*rho*np.sqrt(np.pi)*np.sqrt(sigma_w**2 + sigma_I**2)) + \
          bar_lambda_i/(2*np.sqrt(np.pi)*sigma_w)
      S_hw = mu**2/(2*rho)*cls.erf(sigma_h/2/np.sqrt(sigma_w**2/2+sigma_I**2)) + \
          bar_lambda_i/(2)*cls.erf(sigma_h/np.sqrt(2)/sigma_w)
      S_hh = mu**2/rho*(sigma_h*cls.erf(sigma_h/2/sigma_I) -
            2*sigma_I/np.sqrt(np.pi)*(1-np.exp(- sigma_h**2/4/sigma_I**2))
          ) + bar_lambda_i*sigma_h
      S_wy = mu**2/(2*rho*np.sqrt(np.pi)*np.sqrt(sigma_w**2/2 + sigma_I**2))
      S_hy = mu**2/(2*rho)*cls.erf(sigma_h/2/sigma_I)

      S_ww = S_ww * T
      S_hw = S_hw * T
      S_hh = S_hh * T
      S_wy = S_wy * T
      S_hy = S_hy * T

      H = np.array([[S_ww, S_hw], [S_hw, S_hh]])
      b = np.array([[S_wy + alpha_h * S_hw],
                    [S_hy + alpha_h * S_hh]])
      log_likeli = 0.5 / bar_lambda_j * b.T @ np.linalg.inv(H) @ b

      return log_likeli


  @classmethod
  def beta_distribution_theoretical(
      cls,
      par):
    """Analytical distribution."""
    if (par['loss'] is not None and par['filter_type'] == 'square' and
        par['append_nuisance'][1] == 'gaussian_kernel' and
        par['background_window'] == 'gaussian'):
      num_trials = par['num_trials']
      trial_length = par['trial_length']
      alpha_h = par['alpha'][0][1]
      filter_length = par['beta'][0][1]
      rho = par['rho']
      mu = par['mu']
      baseline = par['baseline']
      sigma_h = par['filter_length']
      sigma_w = par['kernel_width']
      sigma_I = par['background_window_sigma']
      T = num_trials * trial_length
      bar_lambda_i = mu + baseline
      bar_lambda_j = bar_lambda_i * (1 + alpha_h * sigma_h)

      S_ww = mu**2/(2*rho*np.sqrt(np.pi)*np.sqrt(sigma_w**2 + sigma_I**2)) + \
          bar_lambda_i/(2*np.sqrt(np.pi)*sigma_w)
      S_hw = mu**2/(2*rho)*cls.erf(sigma_h/2/np.sqrt(sigma_w**2/2+sigma_I**2)) + \
          bar_lambda_i/(2)*cls.erf(sigma_h/np.sqrt(2)/sigma_w)
      S_hh = mu**2/rho*(sigma_h*cls.erf(sigma_h/2/sigma_I) -
            2*sigma_I/np.sqrt(np.pi)*(1-np.exp(- sigma_h**2/4/sigma_I**2))
          ) + bar_lambda_i*sigma_h
      S_wlam = mu**2/(2*rho*np.sqrt(np.pi)*np.sqrt(sigma_w**2/2 + sigma_I**2))
      S_hlam = mu**2/(2*rho)*cls.erf(sigma_h/2/sigma_I)

      Q = S_ww*S_hlam - S_hw*S_wlam
      R = S_ww*S_hh - S_hw*S_hw
      bias = Q / R

      var = bar_lambda_j*S_ww / (S_ww*S_hh - S_hw*S_hw) / T
      std = np.sqrt(var)
      mean = alpha_h + bias

      return mean, std


  @classmethod
  def plot_likelihood_theoretical_vs_numerical_derivation(
      cls,
      par,
      filter_true_val,
      simulation_files,
      experiment_name=None,
      output_dir =None):
    """Scan the kernel window width."""
    min_kernel_width, max_kernel_width = par['kernel_width']
    log_kernel_widths = np.linspace(np.log10(min_kernel_width),
                                    np.log10(max_kernel_width), 1000)
    kernel_widths = np.power(10, log_kernel_widths)
    CI_scale = scipy.stats.norm.ppf(0.975)  # 95% CI.
    rho = par['rho']
    mu = par['mu']
    baseline = par['baseline']
    trial_length = par['trial_length']
    num_trials = par['num_trials']
    sigma_h = par['filter_length']
    sigma_w = par['kernel_width']
    sigma_I = par['background_window_sigma']
    alpha_h = par['alpha'][0][1]
    beta_h = par['beta'][0][1]
    T = trial_length * num_trials
    bar_lambda_i = mu + baseline
    bar_lambda_j = bar_lambda_i * (1 + alpha_h * sigma_h)

    # Theoretical.
    log_likelihood_list = np.zeros_like(kernel_widths)
    for i, kernel_width in enumerate(kernel_widths):
      par_input = par.copy()
      par_input['kernel_width'] = kernel_width
      log_likelihood_list[i] = cls.log_likelihood_theoretical(par_input)
    log_likelihood_list = log_likelihood_list - np.mean(log_likelihood_list)

    # Numerical
    kernel_widths_sim, S_ww_sim = cls.extract_Sxx_numerical(
        1, 1, simulation_files)
    kernel_widths_sim, S_hw_sim = cls.extract_Sxx_numerical(
        1, 2, simulation_files)
    kernel_widths_sim, S_hh_sim = cls.extract_Sxx_numerical(
        2, 2, simulation_files)
    kernel_widths_sim, S_ws_sim = cls.extract_Sxx_numerical(
        1, 3, simulation_files)
    kernel_widths_sim, S_hs_sim = cls.extract_Sxx_numerical(
        2, 3, simulation_files)
    kernel_widths_sim, S_1s_sim = cls.extract_Sxx_numerical(
        0, 3, simulation_files)
    S_wlam_sim = S_ws_sim - alpha_h * S_hw_sim
    S_hlam_sim = S_hs_sim - alpha_h * S_hh_sim

    num_scenarios = len(simulation_files)
    all_loglikeli = np.zeros(num_scenarios)

    # Likelihood.
    for i, kernel_width in enumerate(kernel_widths_sim):
      H = np.array([[S_ww_sim[i], S_hw_sim[i]], [S_hw_sim[i], S_hh_sim[i]]])
      b = np.array([[S_ws_sim[i]], [S_hs_sim[i]]])
      bar_lambda_j = S_1s_sim[i] / T
      all_loglikeli[i] = 0.5 / bar_lambda_j * b.T @ np.linalg.inv(H) @ b

    all_loglikeli -= np.max(all_loglikeli)

    gs_kw = dict(width_ratios=[11], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(6, 3), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    ax = fig.add_subplot(axs)
    plt.plot(np.log10(kernel_widths_sim)+3, all_loglikeli, c='tab:blue',
        label='numerical mean')
    # plt.legend(loc=[1.01, 0.6])
    plt.xlim(-0.2, 3.7)
    plt.title('log-likelihood')

    # 
    scale = np.zeros(num_scenarios)
    bias = np.zeros(num_scenarios)
    var = np.zeros(num_scenarios)
    vals2 = np.zeros(num_scenarios)
    key = np.zeros(num_scenarios)

    for i, kernel_width in enumerate(kernel_widths_sim):
      scale[i] = S_ww_sim[i]*S_hh_sim[i] - S_hw_sim[i]*S_hw_sim[i]
      var[i] = bar_lambda_j*S_ww_sim[i] / scale[i]
      vals2[i] = S_wlam_sim[i]**2 / scale[i]
      bias[i] = (S_ww_sim[i]*S_hlam_sim[i]-S_hw_sim[i]*S_wlam_sim[i]) / scale[i]
      key[i] = S_wlam_sim[i]**2/2/bar_lambda_j/S_ww_sim[i]
    bias2 = np.square(bias)
    likeli = (bias2 + vals2) / 2/ var

    gs_kw = dict(width_ratios=[1, 1], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(10, 2), gridspec_kw=gs_kw,
        nrows=1, ncols=2)
    ax = fig.add_subplot(axs[0])
    plt.plot(np.log10(kernel_widths_sim)+3, bias, label='var')
    plt.xlim(-0.2, 3.7)
    plt.axhline(0, color='lightgrey', lw=1)
    ax = fig.add_subplot(axs[1])
    plt.plot(np.log10(kernel_widths_sim)+3, np.sqrt(var), label='var')
    plt.xlim(-0.2, 3.7)
    plt.ylim(0)
    plt.show()

    gs_kw = dict(width_ratios=[11], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(6, 3), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    ax = fig.add_subplot(axs)
    plt.plot(np.log10(kernel_widths_sim)+3, likeli, label='likeli')
    plt.plot(np.log10(kernel_widths_sim)+3, key, label='key')
    plt.plot(np.log10(kernel_widths_sim)+3, bias2 /2/var, label='bias2/2/var')
    plt.legend(loc=[1.01, 0.6])
    plt.xlim(-0.2, 3.7)
    # plt.ylim(0)
    plt.axhline(0, color='lightgrey', lw=1)

    if output_dir is not None:
      file_path = (output_dir + experiment_name +
                   'log_likelihood_numerical_derivation.pdf')
      plt.savefig(file_path)
      print('save figure:', file_path)
    plt.show()


  @classmethod
  def extract_Sxx_numerical(
      cls,
      basis_i,basis_j,
      simulation_files):
    """Scan the kernel window width."""
    num_scenarios = len(simulation_files)
    kernel_widths_sim = np.zeros(num_scenarios)
    Sxx_mean_sim = np.zeros(num_scenarios)
    Sxx_std_sim = np.zeros(num_scenarios)

    for i, f in enumerate(simulation_files):
      model_par_list = util.load_variable(f)
      S_xx = cls.extract_Sxx(model_par_list, verbose=False)
      S_xx = np.stack(S_xx, axis=0)
      if i == 0:
        dt = model_par_list[i]['dt']
        num_sims = len(model_par_list)

      if 'kernel_width' in model_par_list[0]:
        kernel_widths_sim[i] = model_par_list[0]['kernel_width']
        if ((basis_i == 0 and basis_j == 3) or
            (basis_i == 1 and basis_j == 3) or
            (basis_i == 2 and basis_j == 3)):
          Sxx_mean_sim[i] = np.mean(S_xx[:,basis_i,basis_j])
        else:
          Sxx_mean_sim[i] = np.mean(S_xx[:,basis_i,basis_j]) * dt
      else:
        kernel_widths_sim[i] = np.nan

    return kernel_widths_sim, Sxx_mean_sim


  @classmethod
  def plot_rmse_likelihood_theoretical(
      cls,
      par,
      sigma_Is=[0.08,0.1,0.12],
      sigma_hs=[0.02,0.03,0.04],
      alpha_hs=[-2,0,2],
      experiment_name=None,
      output_dir=None):
    """Scan the kernel window width."""
    min_kernel_width, max_kernel_width = par['kernel_width']
    log_kernel_widths = np.linspace(np.log10(min_kernel_width),
                                    np.log10(max_kernel_width), 200)
    kernel_widths = np.power(10, log_kernel_widths)
    xlim = (-0.2, 4.2)
    xticks_label = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    xticks = np.log10(np.array(xticks_label))
    xticks = list(xticks)
    style = ['k', 'b', 'g']

    # ------------------ sigma_I ------------------
    rmse_list = np.zeros([len(sigma_Is), len(kernel_widths)])
    log_likelihood_list = np.zeros([len(sigma_Is), len(kernel_widths)])
    min_rmse_kernel_width = np.zeros(len(sigma_Is))
    max_ll_kernel_width = np.zeros(len(sigma_Is))

    par_input = par.copy()
    for ind, sigma_I in enumerate(sigma_Is):
      par_input['background_window_sigma'] = sigma_I
      bias_list = np.zeros_like(kernel_widths)
      var_list = np.zeros_like(kernel_widths)
      ll_list = np.zeros_like(kernel_widths)
      for i, kernel_width in enumerate(kernel_widths):
        par_input['kernel_width'] = kernel_width
        bias_list[i], _, _ = cls.bias_theoretical(par_input)
        var_list[i] = cls.var_theoretical(par_input)
        ll_list[i] = cls.log_likelihood_theoretical(par_input)
      rmse_list[ind] = np.sqrt(var_list + np.square(bias_list))
      log_likelihood_list[ind] = ll_list - np.max(ll_list)
      nadir_ind = np.argmin(rmse_list[ind])
      min_rmse_kernel_width[ind] = kernel_widths[nadir_ind]
      max_ind = np.argmax(log_likelihood_list[ind])
      max_ll_kernel_width[ind] = kernel_widths[max_ind]

    gs_kw = dict(width_ratios=[1], height_ratios=[1,1])
    fig, axs = plt.subplots(figsize=(6, 4.5), gridspec_kw=gs_kw,
        nrows=2, ncols=1)
    plt.subplots_adjust(left=None, right=None, hspace=0.2, wspace=0)
    ax = fig.add_subplot(axs[0])
    plt.axhline(0, lw=2, color='lightgrey')
    for i, val in enumerate(sigma_Is):
      plt.plot(np.log10(kernel_widths)+3, rmse_list[i], style[i], lw=1.2,
          label=fr'$\sigma_I=${val*1000:.0f} ms')
    for v in min_rmse_kernel_width:
      plt.axvline(np.log10(v)+3, color='lightgrey', lw=1)
    plt.ylabel('RMSE [spikes/sec]')
    # plt.xlabel(r'$\sigma_w$ [ms]')
    plt.xticks(xticks, xticks_label)
    plt.ylim(0)
    plt.xlim(xlim)
    plt.legend(fontsize=8)

    ax = fig.add_subplot(axs[1])
    for i, val in enumerate(sigma_Is):
      plt.plot(np.log10(kernel_widths)+3, log_likelihood_list[i],
          style[i], lw=1.2)
    for v in max_ll_kernel_width:
      plt.axvline(np.log10(v)+3, color='lightgrey', lw=1)
    plt.ylabel('log-likelihood')
    plt.xlabel(r'$\sigma_w$ [ms]')
    plt.xticks(xticks, xticks_label)
    plt.xlim(xlim)

    if output_dir is not None:
      file_path = output_dir + experiment_name + 'tune_sigma_I.pdf'
      plt.savefig(file_path)
      print('save figure:', file_path)
    plt.show()

    # ------------------ sigma_h ------------------
    rmse_list = np.zeros([len(sigma_Is), len(kernel_widths)])
    log_likelihood_list = np.zeros([len(sigma_Is), len(kernel_widths)])
    min_rmse_kernel_width = np.zeros(len(sigma_Is))
    max_ll_kernel_width = np.zeros(len(sigma_Is))

    par_input = par.copy()
    for ind, sigma_h in enumerate(sigma_hs):
      par_input['filter_length'] = sigma_h
      bias_list = np.zeros_like(kernel_widths)
      var_list = np.zeros_like(kernel_widths)
      ll_list = np.zeros_like(kernel_widths)
      for i, kernel_width in enumerate(kernel_widths):
        par_input['kernel_width'] = kernel_width
        bias_list[i], _, _ = cls.bias_theoretical(par_input)
        var_list[i] = cls.var_theoretical(par_input)
        ll_list[i] = cls.log_likelihood_theoretical(par_input)
      rmse_list[ind] = np.sqrt(var_list + np.square(bias_list))
      log_likelihood_list[ind] = ll_list - np.max(ll_list)
      nadir_ind = np.argmin(rmse_list[ind])
      min_rmse_kernel_width[ind] = kernel_widths[nadir_ind]
      max_ind = np.argmax(log_likelihood_list[ind])
      max_ll_kernel_width[ind] = kernel_widths[max_ind]

    gs_kw = dict(width_ratios=[1], height_ratios=[1,1])
    fig, axs = plt.subplots(figsize=(6, 4.5), gridspec_kw=gs_kw,
        nrows=2, ncols=1)
    plt.subplots_adjust(left=None, right=None, hspace=0.2, wspace=0)
    ax = fig.add_subplot(axs[0])
    plt.axhline(0, lw=2, color='lightgrey')
    for i, val in enumerate(sigma_hs):
      plt.plot(np.log10(kernel_widths)+3, rmse_list[i], style[i], lw=1.2,
          label=fr'$\sigma_h=${val*1000:.0f} ms')
    for v in min_rmse_kernel_width:
      plt.axvline(np.log10(v)+3, color='lightgrey', lw=1)
    plt.ylabel('RMSE [spikes/sec]')
    # plt.xlabel(r'$\sigma_w$ [ms]')
    plt.xticks(xticks, xticks_label)
    plt.ylim(0)
    plt.xlim(xlim)
    plt.legend(fontsize=8)

    ax = fig.add_subplot(axs[1])
    for i, val in enumerate(sigma_hs):
      plt.plot(np.log10(kernel_widths)+3, log_likelihood_list[i],
          style[i], lw=1.2)
    for v in max_ll_kernel_width:
      plt.axvline(np.log10(v)+3, color='lightgrey', lw=1)
    plt.ylabel('log-likelihood')
    plt.xlabel(r'$\sigma_w$ [ms]')
    plt.xticks(xticks, xticks_label)
    plt.xlim(xlim)

    if output_dir is not None:
      file_path = output_dir + experiment_name + 'tune_sigma_h.pdf'
      plt.savefig(file_path)
      print('save figure:', file_path)
    plt.show()

    # ------------------ alpha_h ------------------
    rmse_list = np.zeros([len(sigma_Is), len(kernel_widths)])
    log_likelihood_list = np.zeros([len(sigma_Is), len(kernel_widths)])
    min_rmse_kernel_width = np.zeros(len(sigma_Is))
    max_ll_kernel_width = np.zeros(len(sigma_Is))

    par_input = par.copy()
    for ind, alpha_h in enumerate(alpha_hs):
      par_input['alpha'][0][1] = alpha_h
      bias_list = np.zeros_like(kernel_widths)
      var_list = np.zeros_like(kernel_widths)
      ll_list = np.zeros_like(kernel_widths)
      for i, kernel_width in enumerate(kernel_widths):
        par_input['kernel_width'] = kernel_width
        bias_list[i], _, _ = cls.bias_theoretical(par_input)
        var_list[i] = cls.var_theoretical(par_input)
        ll_list[i] = cls.log_likelihood_theoretical(par_input)
      rmse_list[ind] = np.sqrt(var_list + np.square(bias_list))
      log_likelihood_list[ind] = ll_list - np.max(ll_list)
      nadir_ind = np.argmin(rmse_list[ind])
      min_rmse_kernel_width[ind] = kernel_widths[nadir_ind]
      max_ind = np.argmax(log_likelihood_list[ind])
      max_ll_kernel_width[ind] = kernel_widths[max_ind]

    gs_kw = dict(width_ratios=[1], height_ratios=[1,1])
    fig, axs = plt.subplots(figsize=(6, 4.5), gridspec_kw=gs_kw,
        nrows=2, ncols=1)
    plt.subplots_adjust(left=None, right=None, hspace=0.2, wspace=0)
    ax = fig.add_subplot(axs[0])
    plt.axhline(0, lw=2, color='lightgrey')
    for i, val in enumerate(alpha_hs):
      plt.plot(np.log10(kernel_widths)+3, rmse_list[i], style[i], lw=1.2,
          label=fr'$\alpha_h=${val:.1f} spikes/sec')
    for v in min_rmse_kernel_width:
      plt.axvline(np.log10(v)+3, color='lightgrey', lw=1)
    plt.ylabel('RMSE [spikes/sec]')
    # plt.xlabel(r'$\sigma_w$ [ms]')
    plt.xticks(xticks, xticks_label)
    plt.ylim(0)
    plt.xlim(xlim)
    plt.legend(fontsize=8)

    ax = fig.add_subplot(axs[1])
    for i, val in enumerate(alpha_hs):
      plt.plot(np.log10(kernel_widths)+3, log_likelihood_list[i],
          style[i], lw=1.2)
    for v in max_ll_kernel_width:
      plt.axvline(np.log10(v)+3, color='lightgrey', lw=1)
    plt.ylabel('log-likelihood')
    plt.xlabel(r'$\sigma_w$ [ms]')
    plt.xticks(xticks, xticks_label)
    plt.xlim(xlim)

    if output_dir is not None:
      file_path = output_dir + experiment_name + 'tune_alpha_h.pdf'
      plt.savefig(file_path)
      print('save figure:', file_path)
    plt.show()


  @classmethod
  def plot_bias_rmse_likelihood_theoretical(
      cls,
      par,
      sigma_Is=[0.08,0.1,0.12],
      sigma_hs=[0.02,0.03,0.04],
      alpha_hs=[-2,0,2],
      experiment_name=None,
      output_dir=None):
    """Scan the kernel window width."""
    min_kernel_width, max_kernel_width = par['kernel_width']
    log_kernel_widths = np.linspace(np.log10(min_kernel_width),
                                    np.log10(max_kernel_width), 200)
    kernel_widths = np.power(10, log_kernel_widths)
    xlim = (-0.2, 3.2)
    xticks_label = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    xticks = np.log10(np.array(xticks_label))
    xticks = list(xticks)
    style = ['k', 'b', 'g']

    # ------------------ sigma_I ------------------
    rmse_list = np.zeros([len(sigma_Is), len(kernel_widths)])
    all_bias_list = np.zeros([len(sigma_Is), len(kernel_widths)])
    log_likelihood_list = np.zeros([len(sigma_Is), len(kernel_widths)])
    min_rmse_kernel_width = np.zeros(len(sigma_Is))
    max_ll_kernel_width = np.zeros(len(sigma_Is))

    par_input = par.copy()
    for ind, sigma_I in enumerate(sigma_Is):
      par_input['background_window_sigma'] = sigma_I
      bias_list = np.zeros_like(kernel_widths)
      var_list = np.zeros_like(kernel_widths)
      ll_list = np.zeros_like(kernel_widths)
      for i, kernel_width in enumerate(kernel_widths):
        par_input['kernel_width'] = kernel_width
        bias_list[i], _, _ = cls.bias_theoretical(par_input)
        var_list[i] = cls.var_theoretical(par_input)
        ll_list[i] = cls.log_likelihood_theoretical(par_input)
      rmse_list[ind] = np.sqrt(var_list + np.square(bias_list))
      all_bias_list[ind] = bias_list
      log_likelihood_list[ind] = ll_list - np.max(ll_list)
      nadir_ind = np.argmin(rmse_list[ind])
      min_rmse_kernel_width[ind] = kernel_widths[nadir_ind]
      max_ind = np.argmax(log_likelihood_list[ind])
      max_ll_kernel_width[ind] = kernel_widths[max_ind]

    gs_kw = dict(width_ratios=[1], height_ratios=[1,1,1])
    fig, axs = plt.subplots(figsize=(5, 6), gridspec_kw=gs_kw,
        nrows=3, ncols=1)
    plt.subplots_adjust(left=None, right=None, hspace=0.2, wspace=0)
    ax = fig.add_subplot(axs[0])
    plt.axhline(0, lw=2, color='lightgrey')
    for i, val in enumerate(sigma_Is):
      plt.plot(np.log10(kernel_widths)+3, all_bias_list[i], style[i], lw=1.2,
          label=fr'$\sigma_I=${val*1000:.0f} ms')
    for v in min_rmse_kernel_width:
      plt.axvline(np.log10(v)+3, color='lightgrey', lw=1)
    plt.ylabel('Bias [spikes/sec]')
    # plt.xlabel(r'$\sigma_w$ [ms]')
    plt.xticks(xticks, xticks_label)
    # plt.ylim(-1, 2.5)
    plt.xlim(xlim)
    # plt.legend(loc=[0.7,0], fontsize=8)
    plt.legend(loc=[0.0,0], fontsize=8)

    ax = fig.add_subplot(axs[1])
    plt.axhline(0, lw=2, color='lightgrey')
    for i, val in enumerate(sigma_Is):
      plt.plot(np.log10(kernel_widths)+3, rmse_list[i], style[i], lw=1.2,
          label=fr'$\sigma_I=${val*1000:.0f} ms')
    for v in min_rmse_kernel_width:
      plt.axvline(np.log10(v)+3, color='lightgrey', lw=1)
    plt.ylabel('RMSE [spikes/sec]')
    # plt.xlabel(r'$\sigma_w$ [ms]')
    plt.xticks(xticks, xticks_label)
    plt.ylim(0)
    plt.xlim(xlim)
    # plt.legend(fontsize=8)

    ax = fig.add_subplot(axs[2])
    for i, val in enumerate(sigma_Is):
      plt.plot(np.log10(kernel_widths)+3, log_likelihood_list[i],
          style[i], lw=1.2)
    for v in max_ll_kernel_width:
      plt.axvline(np.log10(v)+3, color='lightgrey', lw=1)
    plt.ylabel('log-likelihood')
    plt.xlabel(r'$\sigma_w$ [ms]')
    plt.xticks(xticks, xticks_label)
    plt.xlim(xlim)

    if output_dir is not None:
      file_path = output_dir + experiment_name + 'bias_rmse_likeli_tune_sigma_I.pdf'
      plt.savefig(file_path)
      print('save figure:', file_path)
    plt.show()

    # ------------------ sigma_h ------------------
    rmse_list = np.zeros([len(sigma_Is), len(kernel_widths)])
    all_bias_list = np.zeros([len(sigma_Is), len(kernel_widths)])
    log_likelihood_list = np.zeros([len(sigma_Is), len(kernel_widths)])
    min_rmse_kernel_width = np.zeros(len(sigma_Is))
    max_ll_kernel_width = np.zeros(len(sigma_Is))

    par_input = par.copy()
    for ind, sigma_h in enumerate(sigma_hs):
      par_input['filter_length'] = sigma_h
      bias_list = np.zeros_like(kernel_widths)
      var_list = np.zeros_like(kernel_widths)
      ll_list = np.zeros_like(kernel_widths)
      for i, kernel_width in enumerate(kernel_widths):
        par_input['kernel_width'] = kernel_width
        bias_list[i], _, _ = cls.bias_theoretical(par_input)
        var_list[i] = cls.var_theoretical(par_input)
        ll_list[i] = cls.log_likelihood_theoretical(par_input)
      rmse_list[ind] = np.sqrt(var_list + np.square(bias_list))
      all_bias_list[ind] = bias_list
      log_likelihood_list[ind] = ll_list - np.max(ll_list)
      nadir_ind = np.argmin(rmse_list[ind])
      min_rmse_kernel_width[ind] = kernel_widths[nadir_ind]
      max_ind = np.argmax(log_likelihood_list[ind])
      max_ll_kernel_width[ind] = kernel_widths[max_ind]

    gs_kw = dict(width_ratios=[1], height_ratios=[1,1,1])
    fig, axs = plt.subplots(figsize=(5, 6), gridspec_kw=gs_kw,
        nrows=3, ncols=1)
    plt.subplots_adjust(left=None, right=None, hspace=0.2, wspace=0)

    ax = fig.add_subplot(axs[0])
    plt.axhline(0, lw=2, color='lightgrey')
    for i, val in enumerate(sigma_hs):
      plt.plot(np.log10(kernel_widths)+3, all_bias_list[i], style[i], lw=1.2,
          label=fr'$\sigma_h=${val*1000:.0f} ms')
    for v in min_rmse_kernel_width:
      plt.axvline(np.log10(v)+3, color='lightgrey', lw=1)
    # plt.ylabel('RMSE [spikes/sec]')
    # plt.xlabel(r'$\sigma_w$ [ms]')
    plt.xticks(xticks, xticks_label)
    # plt.ylim(-1, 2.5)
    plt.xlim(xlim)
    # plt.legend(loc=[0.75,0], fontsize=8)
    plt.legend(loc=[0.0,0], fontsize=8)

    ax = fig.add_subplot(axs[1])
    plt.axhline(0, lw=2, color='lightgrey')
    for i, val in enumerate(sigma_hs):
      plt.plot(np.log10(kernel_widths)+3, rmse_list[i], style[i], lw=1.2,
          label=fr'$\sigma_h=${val*1000:.0f} ms')
    for v in min_rmse_kernel_width:
      plt.axvline(np.log10(v)+3, color='lightgrey', lw=1)
    # plt.ylabel('RMSE [spikes/sec]')
    # plt.xlabel(r'$\sigma_w$ [ms]')
    plt.xticks(xticks, xticks_label)
    plt.ylim(0)
    plt.xlim(xlim)
    # plt.legend(fontsize=8)

    ax = fig.add_subplot(axs[2])
    for i, val in enumerate(sigma_hs):
      plt.plot(np.log10(kernel_widths)+3, log_likelihood_list[i],
          style[i], lw=1.2)
    for v in max_ll_kernel_width:
      plt.axvline(np.log10(v)+3, color='lightgrey', lw=1)
    # plt.ylabel('log-likelihood')
    plt.xlabel(r'$\sigma_w$ [ms]')
    plt.xticks(xticks, xticks_label)
    plt.xlim(xlim)

    if output_dir is not None:
      file_path = output_dir + experiment_name + 'bias_rmse_likeli_tune_sigma_h.pdf'
      plt.savefig(file_path)
      print('save figure:', file_path)
    plt.show()

    # ------------------ alpha_{i\to j} ------------------
    rmse_list = np.zeros([len(sigma_Is), len(kernel_widths)])
    all_bias_list = np.zeros([len(sigma_Is), len(kernel_widths)])
    log_likelihood_list = np.zeros([len(sigma_Is), len(kernel_widths)])
    min_rmse_kernel_width = np.zeros(len(sigma_Is))
    max_ll_kernel_width = np.zeros(len(sigma_Is))

    par_input = par.copy()
    for ind, alpha_h in enumerate(alpha_hs):
      par_input['alpha'][0][1] = alpha_h
      bias_list = np.zeros_like(kernel_widths)
      var_list = np.zeros_like(kernel_widths)
      ll_list = np.zeros_like(kernel_widths)
      for i, kernel_width in enumerate(kernel_widths):
        par_input['kernel_width'] = kernel_width
        bias_list[i], _, _ = cls.bias_theoretical(par_input)
        var_list[i] = cls.var_theoretical(par_input)
        ll_list[i] = cls.log_likelihood_theoretical(par_input)
      rmse_list[ind] = np.sqrt(var_list + np.square(bias_list))
      all_bias_list[ind] = bias_list
      log_likelihood_list[ind] = ll_list - np.max(ll_list)
      nadir_ind = np.argmin(rmse_list[ind])
      min_rmse_kernel_width[ind] = kernel_widths[nadir_ind]
      max_ind = np.argmax(log_likelihood_list[ind])
      max_ll_kernel_width[ind] = kernel_widths[max_ind]

    gs_kw = dict(width_ratios=[1], height_ratios=[1,1,1])
    fig, axs = plt.subplots(figsize=(5, 6), gridspec_kw=gs_kw,
        nrows=3, ncols=1)
    plt.subplots_adjust(left=None, right=None, hspace=0.2, wspace=0)

    ax = fig.add_subplot(axs[0])
    plt.axhline(0, lw=2, color='lightgrey')
    for i, val in enumerate(alpha_hs):
      plt.plot(np.log10(kernel_widths)+3, all_bias_list[i], style[i], lw=1.2,
          label=r'$\alpha_{i\to j}=$' + f'{val:.1f} spikes/sec')
    for v in min_rmse_kernel_width:
      plt.axvline(np.log10(v)+3, color='lightgrey', lw=1)
    # plt.ylabel('RMSE [spikes/sec]')
    # plt.xlabel(r'$\sigma_w$ [ms]')
    plt.xticks(xticks, xticks_label)
    # plt.ylim(-1, 2.5)
    plt.xlim(xlim)
    # plt.legend(loc=[0.59,0], fontsize=8)
    plt.legend(loc=[0.0,0], fontsize=8)

    ax = fig.add_subplot(axs[1])
    plt.axhline(0, lw=2, color='lightgrey')
    for i, val in enumerate(alpha_hs):
      plt.plot(np.log10(kernel_widths)+3, rmse_list[i], style[i], lw=1.2,
          label=r'$\alpha_{i\to j}=$' + f'{val:.1f} spikes/sec')
    for v in min_rmse_kernel_width:
      plt.axvline(np.log10(v)+3, color='lightgrey', lw=1)
    # plt.ylabel('RMSE [spikes/sec]')
    # plt.xlabel(r'$\sigma_w$ [ms]')
    plt.xticks(xticks, xticks_label)
    plt.ylim(0)
    plt.xlim(xlim)
    # plt.legend(fontsize=8)

    ax = fig.add_subplot(axs[2])
    for i, val in enumerate(alpha_hs):
      plt.plot(np.log10(kernel_widths)+3, log_likelihood_list[i],
          style[i], lw=1.2)
    for v in max_ll_kernel_width:
      plt.axvline(np.log10(v)+3, color='lightgrey', lw=1)
    # plt.ylabel('log-likelihood')
    plt.xlabel(r'$\sigma_w$ [ms]')
    plt.xticks(xticks, xticks_label)
    plt.xlim(xlim)

    if output_dir is not None:
      file_path = output_dir + experiment_name + 'bias_rmse_likeli_tune_alpha_h.pdf'
      plt.savefig(file_path)
      print('save figure:', file_path)
    plt.show()


  @classmethod
  def plot_multiple_bias_theoretical(
      cls,
      par,
      sigma_Is=[0.08,0.1,0.12],
      sigma_hs=[0.02,0.03,0.04],
      alpha_hs=[-2,0,2],
      experiment_name=None,
      output_dir=None):
    """Scan the kernel window width."""
    min_kernel_width, max_kernel_width = par['kernel_width']
    log_kernel_widths = np.linspace(np.log10(min_kernel_width),
                                    np.log10(max_kernel_width), 1000)
    kernel_widths = np.power(10, log_kernel_widths)
    xlim = (-0.3, 3.4)
    xticks_label = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    xticks = np.log10(np.array(xticks_label))
    xticks = list(xticks)
    style = ['k', 'b', 'g']

    # ------------------ sigma_I ------------------
    rmse_list = np.zeros([len(sigma_Is), len(kernel_widths)])
    all_bias_list = np.zeros([len(sigma_Is), len(kernel_widths)])
    bias_lim_list = np.zeros(len(sigma_Is))
    log_likelihood_list = np.zeros([len(sigma_Is), len(kernel_widths)])
    min_rmse_kernel_width = np.zeros(len(sigma_Is))
    max_ll_kernel_width = np.zeros(len(sigma_Is))

    par_input = par.copy()
    for ind, sigma_I in enumerate(sigma_Is):
      par_input['background_window_sigma'] = sigma_I
      bias_list = np.zeros_like(kernel_widths)
      var_list = np.zeros_like(kernel_widths)
      ll_list = np.zeros_like(kernel_widths)
      for i, kernel_width in enumerate(kernel_widths):
        par_input['kernel_width'] = kernel_width
        bias_list[i], _, _ = cls.bias_theoretical(par_input)
        var_list[i] = cls.var_theoretical(par_input)
        ll_list[i] = cls.log_likelihood_theoretical(par_input)
      bias_lim_list[ind] = cls.bias_theoretical_extreme_cases(par_input)
      rmse_list[ind] = np.sqrt(var_list + np.square(bias_list))
      all_bias_list[ind] = np.array(bias_list)
      log_likelihood_list[ind] = ll_list - np.max(ll_list)
      nadir_ind = np.argmin(rmse_list[ind])
      min_rmse_kernel_width[ind] = kernel_widths[nadir_ind]
      max_ind = np.argmax(log_likelihood_list[ind])
      max_ll_kernel_width[ind] = kernel_widths[max_ind]

    gs_kw = dict(width_ratios=[1], height_ratios=[1,])
    fig, axs = plt.subplots(figsize=(6, 2), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    plt.subplots_adjust(left=None, right=None, hspace=0.2, wspace=0)
    ax = fig.add_subplot(axs)
    ax.tick_params(labelbottom=False)
    plt.axhline(0, lw=2, color='lightgrey')
    for i, val in enumerate(sigma_Is):
      plt.plot(np.log10(kernel_widths)+3, all_bias_list[i], style[i], lw=1.2,
          label=fr'$\sigma_I=${val*1000:.0f} ms')
      # plt.plot([3.4], [bias_lim_list[i]], '+', c=style[i], ms=11)

    # for v in min_rmse_kernel_width:
    #   plt.axvline(np.log10(v)+3, color='lightgrey', lw=1)
    plt.ylabel('Bias [spikes/sec]', fontsize=12)
    # plt.xlabel(r'$\sigma_w$ [ms]', fontsize=12)
    plt.xticks(xticks, xticks_label, rotation=-45, fontsize=12)
    plt.ylim(-1, 2.5)
    plt.xlim(xlim)
    # plt.legend(loc=[-0.2,1.05], ncol=3, fontsize=9)
    plt.legend(loc=[0,.01], ncol=1, fontsize=9)

    # ax = fig.add_subplot(axs[1])
    # for i, val in enumerate(sigma_Is):
    #   plt.plot(np.log10(kernel_widths)+3, log_likelihood_list[i],
    #       style[i], lw=1.2)
    # for v in max_ll_kernel_width:
    #   plt.axvline(np.log10(v)+3, color='lightgrey', lw=1)
    # plt.ylabel('log-likelihood')
    # plt.xlabel(r'$\sigma_w$ [ms]')
    # plt.xticks(xticks, xticks_label)
    # plt.xlim(xlim)

    if output_dir is not None:
      file_path = output_dir + experiment_name + 'bias_tune_sigma_I.pdf'
      plt.savefig(file_path)
      print('save figure:', file_path)
    plt.show()

    # ------------------ sigma_h ------------------
    rmse_list = np.zeros([len(sigma_Is), len(kernel_widths)])
    all_bias_list = np.zeros([len(sigma_Is), len(kernel_widths)])
    bias_lim_list = np.zeros(len(sigma_Is))
    log_likelihood_list = np.zeros([len(sigma_Is), len(kernel_widths)])
    min_rmse_kernel_width = np.zeros(len(sigma_Is))
    max_ll_kernel_width = np.zeros(len(sigma_Is))

    par_input = par.copy()
    for ind, sigma_h in enumerate(sigma_hs):
      par_input['filter_length'] = sigma_h
      bias_list = np.zeros_like(kernel_widths)
      var_list = np.zeros_like(kernel_widths)
      ll_list = np.zeros_like(kernel_widths)
      for i, kernel_width in enumerate(kernel_widths):
        par_input['kernel_width'] = kernel_width
        bias_list[i], _, _ = cls.bias_theoretical(par_input)
        var_list[i] = cls.var_theoretical(par_input)
        ll_list[i] = cls.log_likelihood_theoretical(par_input)
      bias_lim_list[ind] = cls.bias_theoretical_extreme_cases(par_input)
      rmse_list[ind] = np.sqrt(var_list + np.square(bias_list))
      all_bias_list[ind] = np.array(bias_list)
      log_likelihood_list[ind] = ll_list - np.max(ll_list)
      nadir_ind = np.argmin(rmse_list[ind])
      min_rmse_kernel_width[ind] = kernel_widths[nadir_ind]
      max_ind = np.argmax(log_likelihood_list[ind])
      max_ll_kernel_width[ind] = kernel_widths[max_ind]

    gs_kw = dict(width_ratios=[1], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(6, 2), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    plt.subplots_adjust(left=None, right=None, hspace=0.2, wspace=0)
    ax = fig.add_subplot(axs)
    plt.axhline(0, lw=2, color='lightgrey')
    for i, val in enumerate(sigma_hs):
      plt.plot(np.log10(kernel_widths)+3, all_bias_list[i], style[i], lw=1.2,
          label=fr'$\sigma_h=${val*1000:.0f} ms')
    # for v in min_rmse_kernel_width:
    #   plt.axvline(np.log10(v)+3, color='lightgrey', lw=1)
    plt.ylabel('Bias [spikes/sec]')
    plt.xlabel(r'$\sigma_w$ [ms]', fontsize=12)
    plt.xticks(xticks, xticks_label, rotation=-45, fontsize=12)
    plt.ylim(-1, 2.5)
    plt.xlim(xlim)
    # plt.legend(loc=[-0.2,1.05], ncol=3, fontsize=9)
    plt.legend(loc=[0,.01], ncol=1, fontsize=9)

    # ax = fig.add_subplot(axs[1])
    # for i, val in enumerate(sigma_hs):
    #   plt.plot(np.log10(kernel_widths)+3, log_likelihood_list[i],
    #       style[i], lw=1.2)
    # for v in max_ll_kernel_width:
    #   plt.axvline(np.log10(v)+3, color='lightgrey', lw=1)
    # plt.ylabel('log-likelihood')
    # plt.xlabel(r'$\sigma_w$ [ms]')
    # plt.xticks(xticks, xticks_label)
    # plt.xlim(xlim)

    if output_dir is not None:
      file_path = output_dir + experiment_name + 'bias_tune_sigma_h.pdf'
      plt.savefig(file_path)
      print('save figure:', file_path)
    plt.show()

    return

    # ------------------ alpha_h ------------------
    rmse_list = np.zeros([len(sigma_Is), len(kernel_widths)])
    log_likelihood_list = np.zeros([len(sigma_Is), len(kernel_widths)])
    min_rmse_kernel_width = np.zeros(len(sigma_Is))
    max_ll_kernel_width = np.zeros(len(sigma_Is))

    par_input = par.copy()
    for ind, alpha_h in enumerate(alpha_hs):
      par_input['alpha'][0][1] = alpha_h
      bias_list = np.zeros_like(kernel_widths)
      var_list = np.zeros_like(kernel_widths)
      ll_list = np.zeros_like(kernel_widths)
      for i, kernel_width in enumerate(kernel_widths):
        par_input['kernel_width'] = kernel_width
        bias_list[i], _, _ = cls.bias_theoretical(par_input)
        var_list[i] = cls.var_theoretical(par_input)
        ll_list[i] = cls.log_likelihood_theoretical(par_input)
      rmse_list[ind] = np.sqrt(var_list + np.square(bias_list))
      log_likelihood_list[ind] = ll_list - np.max(ll_list)
      nadir_ind = np.argmin(rmse_list[ind])
      min_rmse_kernel_width[ind] = kernel_widths[nadir_ind]
      max_ind = np.argmax(log_likelihood_list[ind])
      max_ll_kernel_width[ind] = kernel_widths[max_ind]

    gs_kw = dict(width_ratios=[1], height_ratios=[1,1])
    fig, axs = plt.subplots(figsize=(6, 4.5), gridspec_kw=gs_kw,
        nrows=2, ncols=1)
    plt.subplots_adjust(left=None, right=None, hspace=0.2, wspace=0)
    ax = fig.add_subplot(axs[0])
    plt.axhline(0, lw=2, color='lightgrey')
    for i, val in enumerate(alpha_hs):
      plt.plot(np.log10(kernel_widths)+3, rmse_list[i], style[i], lw=1.2,
          label=fr'$\alpha_h=${val:.1f} spikes/sec')
    for v in min_rmse_kernel_width:
      plt.axvline(np.log10(v)+3, color='lightgrey', lw=1)
    plt.ylabel('RMSE [spikes/sec]')
    # plt.xlabel(r'$\sigma_w$ [ms]')
    plt.xticks(xticks, xticks_label)
    plt.ylim(0)
    plt.xlim(xlim)
    plt.legend(fontsize=8)

    ax = fig.add_subplot(axs[1])
    for i, val in enumerate(alpha_hs):
      plt.plot(np.log10(kernel_widths)+3, log_likelihood_list[i],
          style[i], lw=1.2)
    for v in max_ll_kernel_width:
      plt.axvline(np.log10(v)+3, color='lightgrey', lw=1)
    plt.ylabel('log-likelihood')
    plt.xlabel(r'$\sigma_w$ [ms]')
    plt.xticks(xticks, xticks_label)
    plt.xlim(xlim)

    if output_dir is not None:
      file_path = output_dir + experiment_name + 'tune_alpha_h.pdf'
      plt.savefig(file_path)
      print('save figure:', file_path)
    plt.show()


  @classmethod
  def plot_likelihood_theoretical(
      cls,
      par,
      file_path=None):
    """Scan the kernel window width."""
    min_kernel_width, max_kernel_width = par['kernel_width']
    log_kernel_widths = np.linspace(np.log10(min_kernel_width),
                                    np.log10(max_kernel_width), 1000)
    kernel_widths = np.power(10, log_kernel_widths)
    CI_scale = scipy.stats.norm.ppf(0.975)  # 95% CI.
    rho = par['rho']
    mu = par['mu']
    baseline = par['baseline']
    trial_length = par['trial_length']
    num_trials = par['num_trials']
    sigma_h = par['filter_length']
    sigma_w = par['kernel_width']
    sigma_I = par['background_window_sigma']
    alpha_h = par['alpha'][0][1]
    beta_h = par['beta'][0][1]
    T = trial_length * num_trials
    bar_lambda_i = mu + baseline
    bar_lambda_j = bar_lambda_i * (1 + alpha_h * sigma_h)

    # Theoretical.
    log_likelihood_list = np.zeros_like(kernel_widths)
    for i, kernel_width in enumerate(kernel_widths):
      par_input = par.copy()
      par_input['kernel_width'] = kernel_width
      log_likelihood_list[i] = cls.log_likelihood_theoretical(par_input)
    log_likelihood_list = log_likelihood_list - np.mean(log_likelihood_list)


  @classmethod
  def plot_bias_theoretical(
      cls,
      par,
      file_path=None):
    """Scan the kernel window width."""
    min_kernel_width, max_kernel_width = par['kernel_width']
    log_kernel_widths = np.linspace(np.log10(min_kernel_width),
                                    np.log10(max_kernel_width), 200)
    kernel_widths = np.power(10, log_kernel_widths)

    bias_list = np.zeros_like(kernel_widths)
    Q_list = np.zeros_like(kernel_widths)
    R_list = np.zeros_like(kernel_widths)
    for i, kernel_width in enumerate(kernel_widths):
      par['kernel_width'] = kernel_width
      bias_list[i], Q_list[i], R_list[i] = cls.bias_theoretical(par)

    root_ind = np.where(np.diff(np.sign(bias_list)) != 0)
    roots = kernel_widths[root_ind]
    nadir_ind = np.argmin(bias_list)
    nadir = kernel_widths[nadir_ind]
    bias_extreme = cls.bias_theoretical_extreme_cases(par)
    print('roots:', roots, '\tnadir:', nadir)

    gs_kw = dict(width_ratios=[11], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(6, 2), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    # plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
    ax = fig.add_subplot(axs)
    plt.axhline(0, lw=2, color='lightgrey')
    plt.plot(np.log10(kernel_widths), bias_list, 'k', lw=2.5, label='theoretical')
    # plt.axhline(bias_extreme, ls='-.', color='k')
    plt.plot([0.25], [bias_extreme], 'k+', ms=11, label='theoretical limit')
    plt.plot([-3], [bias_extreme], 'k+', ms=11)
    # ax.set_xscale('log')
    plt.ylabel('Bias')
    plt.xlabel(r'$\log \sigma_w$')
    xticks = [-3,-2.5, -2,-1.5,-1,-0.5,0,0.25]
    xticks_label = [r'$-\infty$',-2.5, -2,-1.5,-1,-0.5,0,r'$\infty$']
    # plt.xlim(0.001, max_kernel_width)
    # plt.grid(True, which='both', ls='-')
    plt.title('bias')

    return

    gs_kw = dict(width_ratios=[1, 1], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(16, 3), gridspec_kw=gs_kw,
        nrows=1, ncols=2)
    ax = fig.add_subplot(axs[0])
    plt.axhline(0, lw=2, color='lightgrey')
    plt.plot(kernel_widths, Q_list, 'b')
    # plt.plot(kernel_widths, R_list, 'g')
    # plt.ylim(-10000, 1000000)
    ax.set_xscale('log')
    plt.title('Q')

    ax = fig.add_subplot(axs[1])
    plt.axhline(0, lw=2, color='lightgrey')
    # plt.plot(kernel_widths, Q_list, 'b')
    plt.plot(kernel_widths, R_list, 'g')
    # plt.ylim(-10000, 1000000)
    ax.set_xscale('log')
    plt.title('R')
    plt.show()


  @classmethod
  def bootstrap_bias_ci(
      cls,
      x,
      x0,
      num_resamples=1000,
      ci_alpha=0.05):
    """Get the CI of bias, SE and MSE using bootstrap.

    Args:
      x: data.
      x0: true value.
    """
    bootstrap_vals = np.zeros(num_resamples)
    for itr in range(num_resamples):
      x_cnd = np.random.choice(x, size=len(x))
      bootstrap_vals[itr] = np.mean(x_cnd-x0)

    CI_left = np.quantile(bootstrap_vals, ci_alpha/2)
    CI_right = np.quantile(bootstrap_vals, 1-ci_alpha/2)

    return np.array([CI_left, CI_right])


  @classmethod
  def bootstrap_se_ci(
      cls,
      x,
      x0,
      num_resamples=1000,
      ci_alpha=0.05):
    """Get the CI of bias, SE and MSE using bootstrap.

    Args:
      x: data.
      x0: true value.
    """
    bootstrap_vals = np.zeros(num_resamples)
    for itr in range(num_resamples):
      x_cnd = np.random.choice(x, size=len(x))
      bootstrap_vals[itr] = np.std(x_cnd)

    CI_left = np.quantile(bootstrap_vals, ci_alpha/2)
    CI_right = np.quantile(bootstrap_vals, 1-ci_alpha/2)

    return np.array([CI_left, CI_right])


  @classmethod
  def bootstrap_rmse_ci(
      cls,
      x,
      x0,
      num_resamples=1000,
      ci_alpha=0.05):
    """Get the CI of bias, SE and MSE using bootstrap.

    Args:
      x: data.
      x0: true value.
    """
    bootstrap_vals = np.zeros(num_resamples)
    for itr in range(num_resamples):
      x_cnd = np.random.choice(x, size=len(x))
      bootstrap_vals[itr] = np.sqrt(np.mean(np.square(x_cnd-x0)))

    CI_left = np.quantile(bootstrap_vals, ci_alpha/2)
    CI_right = np.quantile(bootstrap_vals, 1-ci_alpha/2)

    return np.array([CI_left, CI_right])


  @classmethod
  def bootstrap_rmise_ci(
      cls,
      fs,
      f0,
      num_resamples=1000,
      ci_alpha=0.05):
    """Get the RMISE using bootstrap.

    Args:
      fs: function data.
      f0: true function.
    """
    bootstrap_vals = np.zeros(num_resamples)
    for itr in range(num_resamples):
      cnd_ids = np.random.choice(fs.shape[0], size=fs.shape[0])
      fs_cnd = fs[cnd_ids]
      rmises = np.sqrt(np.mean(np.square(fs_cnd - f0), axis=1))
      bootstrap_vals[itr] = np.mean(rmises)

    CI_left = np.quantile(bootstrap_vals, ci_alpha/2)
    CI_right = np.quantile(bootstrap_vals, 1-ci_alpha/2)

    return np.array([CI_left, CI_right])


  @classmethod
  def plot_likelihood_theoretical_vs_numerical(
      cls,
      par,
      filter_true_val,
      simulation_files,
      show_theoretical=True,
      aligned_by='max',
      experiment_name=None,
      par2=None,
      output_dir=None,
      verbose=False):
    """Scan the kernel window width.

    Args:
      aligned_by: max, min, max_min
    """
    min_kernel_width, max_kernel_width = par['kernel_width']
    log_kernel_widths = np.linspace(np.log10(min_kernel_width),
                                    np.log10(max_kernel_width), 1000)
    kernel_widths = np.power(10, log_kernel_widths)
    CI_scale = scipy.stats.norm.ppf(0.975)  # 95% CI.
    xlim = (-0.2, 4)

    # Theoretical.
    log_likelihood_list = np.zeros_like(kernel_widths)
    log_likelihood_list2 = np.zeros_like(kernel_widths)
    for i, kernel_width in enumerate(kernel_widths):
      par_input = par.copy()
      par_input['kernel_width'] = kernel_width
      log_likelihood_list[i] = cls.log_likelihood_theoretical(par_input)
      if par2 is not None:
        par_input = par2.copy()
        par_input['kernel_width'] = kernel_width
        log_likelihood_list2[i] = cls.log_likelihood_theoretical(par_input)

    if aligned_by == 'max':
      log_likelihood_list = log_likelihood_list - np.max(log_likelihood_list)
      log_likelihood_list2 = log_likelihood_list2 - np.max(log_likelihood_list2)
    elif aligned_by == 'min':
      log_likelihood_list = log_likelihood_list - np.min(log_likelihood_list)
      log_likelihood_list2 = log_likelihood_list2 - np.min(log_likelihood_list2)
    elif aligned_by == 'max_min':
      gap = np.max(log_likelihood_list) - np.min(log_likelihood_list)
      log_likelihood_list = log_likelihood_list - np.max(log_likelihood_list)
      log_likelihood_list = log_likelihood_list / gap
      gap = np.max(log_likelihood_list2) - np.min(log_likelihood_list2)
      log_likelihood_list2 = log_likelihood_list2 - np.max(log_likelihood_list2)
      log_likelihood_list2 = log_likelihood_list2 / gap

    # Numerical
    num_scenarios = len(simulation_files)
    kernel_widths_sim = np.zeros(num_scenarios)
    log_likelihood_mean_sim = np.zeros(num_scenarios)
    log_likelihood_std_sim = np.zeros(num_scenarios)
    bootstrap_CI = np.zeros([num_scenarios,2])
    all_loglikeli = [0]*num_scenarios

    for i, f in enumerate(simulation_files):
      try:
        model_par_list = util.load_variable(f)
      except:
        print('No file', f)
      if i == 0:
        num_sims = len(model_par_list)
        print(f'num simulation models {num_sims}')
      loglikelis = -cls.extract_nll(model_par_list, verbose=False)
      all_loglikeli[i] = loglikelis
      # print('len(loglikelis)', len(loglikelis))
      has_lim = False
      if 'kernel_width' in model_par_list[0]:
        kernel_widths_sim[i] = model_par_list[0]['kernel_width']
      else:
        kernel_widths_sim[i] = np.nan
        has_lim = True

    ll_stacked = np.stack(all_loglikeli, axis=0)
    if aligned_by == 'max':
      all_loglikeli = ll_stacked - np.max(ll_stacked, axis=0)
    elif aligned_by == 'min':
      all_loglikeli = ll_stacked - np.min(ll_stacked, axis=0)
    elif aligned_by == 'max_min':
      gap = np.max(ll_stacked, axis=0) - np.min(ll_stacked, axis=0)
      all_loglikeli = (ll_stacked - np.max(ll_stacked, axis=0)) / gap

    if verbose:
      # log-likelihoods need to be aligned.
      plt.figure(figsize=[8, 2])
      plt.subplot(121)
      plt.plot(np.log10(kernel_widths_sim)+3, ll_stacked[:,::5],
          c='k', lw=1)
      plt.subplot(122)
      plt.plot(np.log10(kernel_widths_sim)+3, all_loglikeli[:,::5], c='k')
      plt.xlim(-0.2, 4.2)
    log_likelihood_mean_sim = np.mean(all_loglikeli, axis=1)
    log_likelihood_std_sim = np.std(all_loglikeli, axis=1)
    x_ninfty = 0
    x_infty = 3.4
    x_nojitter = 3.8
    if show_theoretical:
      nadir_ind = np.argmax(log_likelihood_list)
      max_ll_kernel_width = kernel_widths[nadir_ind]
    else:
      # max of the mean curve.
      # nadir_ind = np.argmax(log_likelihood_mean_sim)
      # max_ll_kernel_width = kernel_widths_sim[nadir_ind]

      # mean of the maximums.
      nadir_ind = np.argmax(all_loglikeli, axis=0)
      max_ll_kernel_width = kernel_widths_sim[nadir_ind].mean()

    print('max_ll_kernel_width:', max_ll_kernel_width)

    gs_kw = dict(width_ratios=[11], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(6, 2), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    ax = fig.add_subplot(axs)
    if show_theoretical:
      plt.plot(np.log10(kernel_widths)+3, log_likelihood_list, 'k', lw=2.5,
          label='theoretical')
      if par2 is not None:
        plt.plot(np.log10(kernel_widths)+3, log_likelihood_list2, 'g', lw=2.5)

    # plt.ylabel('log-likelihood (offset aligned)')
    plt.ylabel('log-likelihood')
    plt.xlabel(r'$\sigma_w$ [ms]')
    # xticks = list(np.arange(0.5, 3.5, 0.5))
    # xticks_label = list(np.arange(0.5, 3.5, 0.5))
    xticks_label = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    xticks = np.log10(np.array(xticks_label))
    xticks = list(xticks)
    # xticks = [x_ninfty] + xticks + [x_infty]
    # xticks_label = [r'0'] + xticks_label + [r'$\infty$']

    plt.axvline(np.log10(max_ll_kernel_width)+3, c='lightgrey', lw=1)
    # CI = CI_scale * log_likelihood_std_sim / np.sqrt(num_sims)
    CI = CI_scale * log_likelihood_std_sim
    CI = np.vstack((log_likelihood_mean_sim - CI,
                    log_likelihood_mean_sim + CI)).T
    plt.fill_between(np.log10(kernel_widths_sim)+3, CI[:,0], CI[:,1],
        facecolor='lightblue', alpha=0.3, label='numerical 95% CI')
    plt.plot(np.log10(kernel_widths_sim)+3, log_likelihood_mean_sim, '.-',
        c='tab:blue', label='numerical mean')
    # plt.legend(loc=[1.01, 0.6])
    plt.xlim(xlim)
    plt.xticks(xticks, xticks_label)
    # plt.ylim(-60, 10)

    if output_dir is not None:
      file_path = output_dir + experiment_name + 'log_likelihood.pdf'
      plt.savefig(file_path)
      print('save figure:', file_path)
    plt.show()


  @classmethod
  def plot_square_filter_beta_theoretical_vs_numerical(
      cls,
      par,
      filter_true_val,
      simulation_files,
      show_theoretical=True,
      experiment_name=None,
      par2=None,
      output_dir=None,
      verbose=False):
    """Scan the kernel window width."""
    min_kernel_width, max_kernel_width = par['kernel_width']
    log_kernel_widths = np.linspace(np.log10(min_kernel_width),
                                    np.log10(max_kernel_width), 2000)
    kernel_widths = np.power(10, log_kernel_widths)
    CI_scale = scipy.stats.norm.ppf(0.975)  # 95% CI.
    num_bootstrap_resamples = 200
    xlim = (-0.2, 4)

    #-------------- bias --------------
    bias_list = np.zeros_like(kernel_widths)
    bias_list2 = np.zeros_like(kernel_widths)
    Q_list = np.zeros_like(kernel_widths)
    R_list = np.zeros_like(kernel_widths)

    for i, kernel_width in enumerate(kernel_widths):
      par_input = par.copy()
      par_input['kernel_width'] = kernel_width
      bias_list[i], Q_list[i], R_list[i] = cls.bias_theoretical(par_input)
      if par2 is not None:
        par_input = par2.copy()
        par_input['kernel_width'] = kernel_width
        bias_list2[i], _,_ = cls.bias_theoretical(par_input)

    root_ind = np.where(np.diff(np.sign(bias_list)) != 0)
    roots = kernel_widths[root_ind]
    nadir_ind = np.argmin(bias_list)
    nadir = kernel_widths[nadir_ind]
    bias_extreme = cls.bias_theoretical_extreme_cases(par)
    if par2 is not None:
      bias_extreme2 = cls.bias_theoretical_extreme_cases(par2)
    print('theoretical roots:', roots, '\tnadir:', nadir)

    num_scenarios = len(simulation_files)
    kernel_widths_sim = np.zeros(num_scenarios)
    bias_sim = np.zeros(num_scenarios)
    bias_sim_all = [0] * num_scenarios
    se_sim = np.zeros(num_scenarios)
    rmse_sim = np.zeros(num_scenarios)
    bias_bootstrap_CI = np.zeros([num_scenarios,2])
    se_bootstrap_CI = np.zeros([num_scenarios,2])
    rmse_bootstrap_CI = np.zeros([num_scenarios,2])

    for i, f in enumerate(simulation_files):
      try:
        model_par_list = util.load_variable(f)
      except:
        print('No file', f)
      if i == 0:
        num_sims = len(model_par_list)
        print(f'num simulation models {num_sims}')

      h_vals = cls.extract_filters(model_par_list, verbose=False)
      has_lim = False
      if 'kernel_width' in model_par_list[0]:
        kernel_widths_sim[i] = model_par_list[0]['kernel_width']
        bias_sim[i] = np.mean(h_vals) - filter_true_val
        bias_sim_all[i] = np.array(h_vals) - filter_true_val
        se_sim[i] = np.std(h_vals)
        rmse_sim[i] = np.sqrt(np.mean(np.square(h_vals - filter_true_val)))
        bias_bootstrap_CI[i] = cls.bootstrap_bias_ci(
            h_vals, filter_true_val, ci_alpha=0.05,
            num_resamples=num_bootstrap_resamples)
        se_bootstrap_CI[i] = cls.bootstrap_se_ci(
            h_vals, filter_true_val, ci_alpha=0.05,
            num_resamples=num_bootstrap_resamples)
        rmse_bootstrap_CI[i] = cls.bootstrap_rmse_ci(
            h_vals, filter_true_val, ci_alpha=0.05,
            num_resamples=num_bootstrap_resamples)

      else:
        kernel_widths_sim[i] = np.nan
        has_lim = True
        bias_lim_sim = np.mean(h_vals) - filter_true_val
        bias_lim_sim_all = np.array(h_vals) - filter_true_val
        se_lim_sim = np.std(h_vals)
        rmse_lim_sim = np.sqrt(np.mean(np.square(h_vals - filter_true_val)))
        bias_lim_bootstrap_CI = cls.bootstrap_bias_ci(
            h_vals, filter_true_val, ci_alpha=0.05,
            num_resamples=num_bootstrap_resamples)
        se_lim_bootstrap_CI = cls.bootstrap_se_ci(
            h_vals, filter_true_val, ci_alpha=0.05,
            num_resamples=num_bootstrap_resamples)
        rmse_lim_bootstrap_CI = cls.bootstrap_rmse_ci(
            h_vals, filter_true_val, ci_alpha=0.05,
            num_resamples=num_bootstrap_resamples)

    if verbose:
      bias_CI = CI_scale * np.array(se_sim) / np.sqrt(num_sims)
      bias_CI = np.vstack((bias_sim - bias_CI, bias_sim + bias_CI)).T

      plt.figure(figsize=[10,2])
      plt.subplot(121)
      plt.plot(np.log10(kernel_widths_sim)+3, bias_CI[:,0] - bias_sim)
      plt.plot(np.log10(kernel_widths_sim)+3, bias_bootstrap_CI[:,0] - bias_sim)
      plt.title('upper CI comparison')
      plt.subplot(122)
      plt.plot(np.log10(kernel_widths_sim)+3, bias_CI[:,1] - bias_sim,
          label='CI_scale * std')
      plt.plot(np.log10(kernel_widths_sim)+3, bias_bootstrap_CI[:,1] - bias_sim,
          label='bootstrap')
      plt.title('lower CI comparison')
      plt.legend(loc=[1.01,0.5])

    x_ninfty = 0
    x_infty = 3.4
    x_nojitter = 3.8
    # xticks = list(np.arange(0.5, 3.5, 0.5))
    # xticks_label = list(np.arange(0.5, 3.5, 0.5))
    xticks_label = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    xticks = np.log10(np.array(xticks_label))
    xticks = list(xticks)

    gs_kw = dict(width_ratios=[11], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(6, 2), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    ax = fig.add_subplot(axs)
    plt.axhline(0, lw=2, color='lightgrey')
    if show_theoretical:
      plt.plot(np.log10(kernel_widths)+3, bias_list, 'k', lw=2.5,
          label='theoretical')
      plt.plot([0], [bias_extreme], 'k+', ms=11, label='theoretical limit')
      plt.plot([3.4], [bias_extreme], 'k+', ms=11)
      if par2 is not None:
        plt.plot(np.log10(kernel_widths)+3, bias_list2, 'g', lw=2.5,
            label='theoretical')
        plt.plot([0], [bias_extreme2], 'g+', ms=11, label='theoretical limit')
        plt.plot([3.4], [bias_extreme2], 'g+', ms=11)

      xticks = [x_ninfty] + xticks + [x_infty]
      xticks_label = [r'0'] + xticks_label + [r'$\infty$']
    plt.ylabel('Bias [spikes/sec]')
    plt.xlabel(r'$\sigma_w$ [ms]')

    if simulation_files is not None:
      plt.fill_between(np.log10(kernel_widths_sim)+3,
          bias_bootstrap_CI[:,0], bias_bootstrap_CI[:,1],
          facecolor='lightblue', alpha=0.3, label='numerical 95% CI')
      plt.plot(np.log10(kernel_widths_sim)+3, bias_sim, '.-',
          c='tab:blue', label='numerical')
      if has_lim:
        # plt.plot([x_nojitter], [bias_lim_sim], 'o', c='tab:blue')
        # plt.errorbar([x_nojitter], [bias_lim_sim], yerr=[lim_std_sim],
        #     fmt='o', c='tab:blue')
        yerr = np.abs(bias_lim_bootstrap_CI-bias_lim_sim).reshape(-1,1)
        plt.errorbar([x_nojitter], [bias_lim_sim], yerr=yerr, fmt='o', capsize=3,
            c='tab:blue')
        xticks.append(x_nojitter)
        # xticks_label.append('no\nnuisance')
        xticks_label.append('Hawkes')
    plt.xticks(xticks, xticks_label)
    # plt.legend(loc=[1.01, 0.6])
    plt.xlim(xlim)
    # plt.ylim(-2.2, 2.2)
    # plt.grid()

    if output_dir is not None:
      file_path = output_dir + experiment_name + 'bias.pdf'
      plt.savefig(file_path)
      print('save figure:', file_path)
      plt.show()
    else:
      plt.close()

    #---------------- variance ---------------
    var_list = np.zeros_like(kernel_widths)
    var_list2 = np.zeros_like(kernel_widths)

    for i, kernel_width in enumerate(kernel_widths):
      par_input = par.copy()
      par_input['kernel_width'] = kernel_width
      var_list[i] = cls.var_theoretical(par_input)
      if par2 is not None:
        par_input2 = par2.copy()
        par_input2['kernel_width'] = kernel_width
        var_list2[i] = cls.var_theoretical(par_input)
    var_extreme = cls.var_theoretical_extreme_cases(par_input)
    if par2 is not None:
      var_extreme2 = cls.var_theoretical_extreme_cases(par_input2)

    gs_kw = dict(width_ratios=[11], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(6, 2), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    ax = fig.add_subplot(axs)
    if show_theoretical:
      plt.plot(np.log10(kernel_widths)+3, np.sqrt(var_list), 'k', lw=2.5,
          label='theoretical')
      plt.plot([0], [np.sqrt(var_extreme)], 'k+', ms=11, label='theoretical limit')
      plt.plot([3.4], [np.sqrt(var_extreme)], 'k+', ms=11)
      if par2 is not None:
        plt.plot(np.log10(kernel_widths)+3, np.sqrt(var_list2), 'g', lw=2.5,
            label='theoretical')
        plt.plot([0], [np.sqrt(var_extreme2)], 'g+', ms=11, label='theoretical limit')
        plt.plot([3.4], [np.sqrt(var_extreme2)], 'g+', ms=11)

    if simulation_files is not None:
      plt.fill_between(np.log10(kernel_widths_sim)+3,
            se_bootstrap_CI[:,0], se_bootstrap_CI[:,1],
            facecolor='lightblue', alpha=0.3, label='numerical 95% CI')
      plt.plot(np.log10(kernel_widths_sim)+3, se_sim, '.-', c='tab:blue',
          label='numerical')
      if has_lim:
        # plt.plot([x_nojitter], [se_lim_sim], 'o', c='tab:blue')
        yerr = np.abs(se_lim_bootstrap_CI-se_lim_sim).reshape(-1,1)
        plt.errorbar([x_nojitter], [se_lim_sim], yerr=yerr, fmt='o', capsize=3,
            c='tab:blue')
        xticks.append(x_nojitter)
        # xticks_label.append('no\nnuisance')
        xticks_label.append('Hawkes')

    plt.ylabel('SE [spikes/sec]')
    plt.xlabel(r'$\sigma_w$ [ms]')
    plt.xticks(xticks, xticks_label)
    # plt.legend(loc=[1.01, 0.6])
    # plt.ylim(0, np.max(np.sqrt(var_list)) * 1.3)
    plt.ylim(0, 0.4)
    plt.xlim(xlim)

    if output_dir is not None:
      file_path = output_dir + experiment_name + 'var.pdf'
      plt.savefig(file_path)
      print('save figure:', file_path)
      plt.show()
    else:
      plt.close()

    #---------------- RMSE ---------------
    rmse_list = np.sqrt(var_list + np.square(bias_list))
    rmse_extreme = np.sqrt(var_extreme + np.square(bias_extreme))
    nadir_ind = np.argmin(rmse_list)
    min_rmse_kernel_width = kernel_widths[nadir_ind]
    print('min_rmse_kernel_width:', min_rmse_kernel_width)
    if par2 is not None:
      rmse_list2 = np.sqrt(var_list2 + np.square(bias_list2))
      rmse_extreme2 = np.sqrt(var_extreme2 + np.square(bias_extreme2))

    gs_kw = dict(width_ratios=[11], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(6, 2), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    ax = fig.add_subplot(axs)
    if show_theoretical:
      plt.plot(np.log10(kernel_widths)+3, rmse_list, 'k', lw=2.5,
          label='theoretical')
      plt.plot([0], [rmse_extreme], 'k+', ms=11, label='theoretical limit')
      plt.plot([3.4], [rmse_extreme], 'k+', ms=11)
      plt.axvline(np.log10(min_rmse_kernel_width)+3, c='lightgrey', lw=1)
      if par2 is not None:
        plt.plot(np.log10(kernel_widths)+3, rmse_list2, 'g', lw=2.5)
        plt.plot([0], [rmse_extreme2], 'g+', ms=11)
        plt.plot([3.4], [rmse_extreme2], 'g+', ms=11)

    if simulation_files is not None:
      if len(rmse_sim) > 2:
        nadir_ind = np.argmin(rmse_sim[:-2])
      else:
        nadir_ind = np.argmin(rmse_sim)
      min_rmse_kernel_width_sim = kernel_widths_sim[nadir_ind]
      print('sim RMSE optimal sigma_w', min_rmse_kernel_width_sim)
      plt.fill_between(np.log10(kernel_widths_sim)+3,
            rmse_bootstrap_CI[:,0], rmse_bootstrap_CI[:,1],
            facecolor='lightblue', alpha=0.3, label='numerical 95% CI')
      plt.plot(np.log10(kernel_widths_sim)+3, rmse_sim, '.-', c='tab:blue',
          label='numerical')
      if not show_theoretical:
        print('empirical max is not shown')
      #   plt.axvline(np.log10(min_rmse_kernel_width_sim)+3, c='lightgrey', lw=1)
      # plt.axvline(np.log10(min_rmse_kernel_width_sim)+3, c='lightgrey', lw=1)

      if has_lim:
        # plt.plot([x_nojitter], [rmse_lim_sim], 'o', c='tab:blue')
        yerr = np.abs(rmse_lim_bootstrap_CI-rmse_lim_sim).reshape(-1,1)
        plt.errorbar([x_nojitter], [rmse_lim_sim], yerr=yerr, fmt='o', capsize=3,
            c='tab:blue')
        xticks.append(x_nojitter)
        # xticks_label.append('no\nnuisance')
        xticks_label.append('Hawkes')

    plt.ylabel('RMSE [spikes/sec]')
    plt.xticks(xticks, xticks_label)
    # ax.tick_params(bottom=True, labelbottom=False)
    # plt.xlabel(r'$\sigma_w$ [ms]')
    plt.legend(loc=[1.01, 0.6])
    plt.xlim(xlim)
    # plt.ylim(0, 1)

    if output_dir is not None:
      file_path = output_dir + experiment_name + 'rmse.pdf'
      plt.savefig(file_path)
      print('save figure:', file_path)
      plt.show()
    else:
      plt.close()

    return (kernel_widths_sim, bias_sim, bias_sim_all, bias_lim_sim, bias_lim_sim_all,
        se_sim, se_lim_sim, rmse_sim, rmse_lim_sim)


  @classmethod
  def plot_bspline_filter_beta_theoretical_vs_numerical(
      cls,
      true_par,
      simulation_files,
      show_theoretical=True,
      experiment_name=None,
      output_dir=None,
      verbose=False):
    """Scan the kernel window width."""
    if true_par['filter_type'] == 'square':
        filter_true_val = true_par['alpha'][0][1]
        filter_length = true_par['beta'][0][1]
        def true_filter(t):
          if np.isscalar(t):
            return filter_true_val if t <= filter_length else 0
          else:
            vals = np.zeros_like(t)
            vals[t <= filter_length] = filter_true_val
            return vals
    min_kernel_width, max_kernel_width = true_par['kernel_width']
    log_kernel_widths = np.linspace(np.log10(min_kernel_width),
                                    np.log10(max_kernel_width), 2000)
    kernel_widths = np.power(10, log_kernel_widths)
    CI_scale = scipy.stats.norm.ppf(0.975)  # 95% CI.
    num_bootstrap_resamples = 400
    xlim = (-0.2, 4)

    #-------------- RMISE --------------
    dt=0.0002
    num_scenarios = len(simulation_files)
    kernel_widths_sim = np.zeros(num_scenarios)
    rmise_sim = np.zeros(num_scenarios)
    rmise_bootstrap_CI = np.zeros([num_scenarios,2])

    for i, f in enumerate(simulation_files):
      model_par_list = util.load_variable(f)
      if i == 0:
        num_sims = len(model_par_list)
        print(f'num simulation models {num_sims}')
      t, h_vals = cls.extract_filters(model_par_list, dt=dt, verbose=False)
      h_true = true_filter(t)

      if 'kernel_width' in model_par_list[0]:
        kernel_widths_sim[i] = model_par_list[0]['kernel_width']
        has_lim = False
        rmises = np.sqrt(np.mean(np.square(h_vals - h_true), axis=1))
        rmise_sim[i] = np.mean(rmises)
        rmise_bootstrap_CI[i] = cls.bootstrap_rmise_ci(
            h_vals, h_true, ci_alpha=0.05, num_resamples=num_bootstrap_resamples)
      else:
        kernel_widths_sim[i] = np.nan
        has_lim = True
        rmises = np.sqrt(np.mean(np.square(h_vals - h_true), axis=1))
        rmise_lim_sim = np.mean(rmises)
        rmise_bootstrap_lim_CI = cls.bootstrap_rmise_ci(
            h_vals, h_true, ci_alpha=0.05, num_resamples=num_bootstrap_resamples)
      if verbose:
        plt.figure(figsize=[4, 2])
        plt.plot(t, h_vals.T, c='lightgrey', lw=0.1)
        plt.plot(t, h_vals.mean(axis=0), c='k', lw=1.5)
        plt.plot(t, h_true, color='green')
        plt.title(f'RMISE {rmise_sim[i]:.3f} [spikes/sec]')
        plt.show()

    nadir_ind = np.argmin(rmise_sim[:-1])
    min_rmise_kernel_width = kernel_widths_sim[nadir_ind]
    print('min_rmise_kernel_width:', min_rmise_kernel_width)

    x_ninfty = 0
    x_infty = 3.4
    x_nojitter = 3.8
    # xticks = list(np.arange(0.5, 3.5, 0.5))
    # xticks_label = list(np.arange(0.5, 3.5, 0.5))
    xticks_label = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    xticks = np.log10(np.array(xticks_label))
    xticks = list(xticks)

    gs_kw = dict(width_ratios=[1], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(6, 2), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    ax = fig.add_subplot(axs)
    # plt.axhline(0, lw=2, color='lightgrey')
    plt.fill_between(np.log10(kernel_widths_sim)+3,
        rmise_bootstrap_CI[:,0], rmise_bootstrap_CI[:,1],
        facecolor='lightblue', alpha=0.3, label='numerical 95% CI')
    plt.plot(np.log10(kernel_widths_sim)+3, rmise_sim, '.-',
        c='tab:blue', label='numerical')
    if has_lim:
      # plt.plot([x_nojitter], [bias_lim_sim], 'o', c='tab:blue')
      # plt.errorbar([x_nojitter], [bias_lim_sim], yerr=[lim_std_sim],
      #     fmt='o', c='tab:blue')
      yerr = np.abs(rmise_bootstrap_lim_CI- rmise_lim_sim).reshape(-1,1)
      plt.errorbar([x_nojitter], [rmise_lim_sim], yerr=yerr, fmt='o', capsize=3,
          c='tab:blue')
      xticks.append(x_nojitter)
      xticks_label.append('no\nnuisance')
    plt.axvline(np.log10(min_rmise_kernel_width)+3, c='lightgrey', lw=1)
    plt.xticks(xticks, xticks_label)
    # plt.legend(loc=[1.01, 0.6])
    plt.xlim(xlim)
    plt.ylabel('RMISE [spikes/sec]')
    plt.xlabel(r'$\sigma_w$ [ms]')

    if output_dir is not None:
      file_path = output_dir + experiment_name + 'rmise.pdf'
      plt.savefig(file_path)
      print('save figure:', file_path)
    plt.show()

    dt=0.0002
    sample_t1 = int(5/1000/dt)
    sample_t2 = int(15/1000/dt)
    sample_t3 = int(25/1000/dt)
    #-------------- Bias --------------
    num_scenarios = len(simulation_files)
    kernel_widths_sim = np.zeros(num_scenarios)
    bias_sim1 = np.zeros(num_scenarios)
    bias_sim2 = np.zeros(num_scenarios)
    bias_sim3 = np.zeros(num_scenarios)
    bias_bootstrap_CI = np.zeros([num_scenarios,2])

    for i, f in enumerate(simulation_files):
      model_par_list = util.load_variable(f)
      if i == 0:
        num_sims = len(model_par_list)
        print(f'num simulation models {num_sims}')
      t, h_vals = cls.extract_filters(model_par_list, dt=dt, verbose=False)
      h_true = true_filter(t)

      if 'kernel_width' in model_par_list[0]:
        kernel_widths_sim[i] = model_par_list[0]['kernel_width']
        has_lim = False
        bias_sim1[i] = np.mean(h_vals[:,sample_t1] - h_true[sample_t1])
        bias_sim2[i] = np.mean(h_vals[:,sample_t2] - h_true[sample_t2])
        bias_sim3[i] = np.mean(h_vals[:,sample_t3] - h_true[sample_t3])
        # rmise_bootstrap_CI[i] = cls.bootstrap_rmise_ci(
        #     h_vals, h_true, ci_alpha=0.05, num_resamples=num_bootstrap_resamples)
      else:
        kernel_widths_sim[i] = np.nan
        has_lim = True
        bias_lim_sim1 = np.mean(h_vals[:,sample_t1] - h_true[sample_t1])
        bias_lim_sim2 = np.mean(h_vals[:,sample_t2] - h_true[sample_t2])
        bias_lim_sim3 = np.mean(h_vals[:,sample_t3] - h_true[sample_t3])
        # rmise_bootstrap_lim_CI = cls.bootstrap_rmise_ci(
        #     h_vals, h_true, ci_alpha=0.05, num_resamples=num_bootstrap_resamples)
      if verbose:
        plt.figure(figsize=[4, 2])
        plt.plot(t, h_vals.T, c='lightgrey', lw=0.1)
        plt.plot(t, h_vals.mean(axis=0), c='k', lw=1.5)
        plt.plot(t, h_true, color='green')
        plt.title(f'Bias {bias_sim1[i]:.3f} [spikes/sec]')
        plt.show()

    x_ninfty = 0
    x_infty = 3.4
    x_nojitter = 3.8
    # xticks = list(np.arange(0.5, 3.5, 0.5))
    # xticks_label = list(np.arange(0.5, 3.5, 0.5))
    xticks_label = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    xticks = np.log10(np.array(xticks_label))
    xticks = list(xticks)

    gs_kw = dict(width_ratios=[1], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(6, 2), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    ax = fig.add_subplot(axs)
    plt.axhline(0, lw=2, color='lightgrey')
    # plt.fill_between(np.log10(kernel_widths_sim)+3,
    #     rmise_bootstrap_CI[:,0], rmise_bootstrap_CI[:,1],
    #     facecolor='lightblue', alpha=0.3, label='numerical 95% CI')
    plt.plot(np.log10(kernel_widths_sim)+3, bias_sim1, '.-',
        c='tab:blue', ms=3, label='lag=5 ms')
    plt.plot(np.log10(kernel_widths_sim)+3, bias_sim2, '.-',
        c='k', ms=3, label='lag=15 ms')
    plt.plot(np.log10(kernel_widths_sim)+3, bias_sim3, '.-',
        c='tab:green', ms=3, label='lag=25 ms')
    if has_lim:
      plt.plot([x_nojitter], [bias_lim_sim1], 'o', c='tab:blue')
      plt.plot([x_nojitter], [bias_lim_sim2], 'o', c='k')
      plt.plot([x_nojitter], [bias_lim_sim3], 'o', c='tab:green')
      # plt.errorbar([x_nojitter], [bias_lim_sim], yerr=[lim_std_sim],
      #     fmt='o', c='tab:blue')
      # yerr = np.abs(rmise_bootstrap_lim_CI- rmise_lim_sim).reshape(-1,1)
      # plt.errorbar([x_nojitter], [rmise_lim_sim], yerr=yerr, fmt='o', capsize=3,
      #     c='tab:blue')
      xticks.append(x_nojitter)
      xticks_label.append('no\nnuisance')
    plt.xticks(xticks, xticks_label)
    plt.legend(loc=[0.7, 0.05])
    plt.xlim(xlim)
    plt.ylabel('Bias [spikes/sec]')
    plt.xlabel(r'$\sigma_w$ [ms]')

    if output_dir is not None:
      file_path = output_dir + experiment_name + 'bias.pdf'
      plt.savefig(file_path)
      print('save figure:', file_path)
    plt.show()

    #-------------- SE --------------
    num_scenarios = len(simulation_files)
    kernel_widths_sim = np.zeros(num_scenarios)
    se_sim1 = np.zeros(num_scenarios)
    se_sim2 = np.zeros(num_scenarios)
    se_sim3 = np.zeros(num_scenarios)
    se_bootstrap_CI = np.zeros([num_scenarios,2])

    for i, f in enumerate(simulation_files):
      model_par_list = util.load_variable(f)
      if i == 0:
        num_sims = len(model_par_list)
        print(f'num simulation models {num_sims}')
      t, h_vals = cls.extract_filters(model_par_list, dt=dt, verbose=False)
      h_true = true_filter(t)

      if 'kernel_width' in model_par_list[0]:
        kernel_widths_sim[i] = model_par_list[0]['kernel_width']
        has_lim = False
        se_sim1[i] = np.std(h_vals[:,sample_t1] - h_true[sample_t1])
        se_sim2[i] = np.std(h_vals[:,sample_t2] - h_true[sample_t2])
        se_sim3[i] = np.std(h_vals[:,sample_t3] - h_true[sample_t3])
        # se_bootstrap_CI[i] = cls.bootstrap_rmise_ci(
        #     h_vals, h_true, ci_alpha=0.05, num_resamples=num_bootstrap_resamples)
      else:
        kernel_widths_sim[i] = np.nan
        has_lim = True
        se_lim_sim1 = np.std(h_vals[:,sample_t1] - h_true[sample_t1])
        se_lim_sim2 = np.std(h_vals[:,sample_t2] - h_true[sample_t2])
        se_lim_sim3 = np.std(h_vals[:,sample_t3] - h_true[sample_t3])
        # rmise_bootstrap_lim_CI = cls.bootstrap_rmise_ci(
        #     h_vals, h_true, ci_alpha=0.05, num_resamples=num_bootstrap_resamples)
      if verbose:
        plt.figure(figsize=[4, 2])
        plt.plot(t, h_vals.T, c='lightgrey', lw=0.1)
        plt.plot(t, h_vals.mean(axis=0), c='k', lw=1.5)
        plt.plot(t, h_true, color='green')
        plt.title(f'Bias {bias_sim1[i]:.3f} [spikes/sec]')
        plt.show()

    x_ninfty = 0
    x_infty = 3.4
    x_nojitter = 3.8
    # xticks = list(np.arange(0.5, 3.5, 0.5))
    # xticks_label = list(np.arange(0.5, 3.5, 0.5))
    xticks_label = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    xticks = np.log10(np.array(xticks_label))
    xticks = list(xticks)

    gs_kw = dict(width_ratios=[1], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(6, 2), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    ax = fig.add_subplot(axs)
    plt.axhline(0, lw=2, color='lightgrey')
    # plt.fill_between(np.log10(kernel_widths_sim)+3,
    #     rmise_bootstrap_CI[:,0], rmise_bootstrap_CI[:,1],
    #     facecolor='lightblue', alpha=0.3, label='numerical 95% CI')
    plt.plot(np.log10(kernel_widths_sim)+3, se_sim1, '.-',
        c='tab:blue', ms=3, label='lag=5 ms')
    plt.plot(np.log10(kernel_widths_sim)+3, se_sim2, '.-',
        c='k', ms=3, label='lag=15 ms')
    plt.plot(np.log10(kernel_widths_sim)+3, se_sim3, '.-',
        c='tab:green', ms=3, label='lag=25 ms')
    if has_lim:
      plt.plot([x_nojitter], [se_lim_sim1], 'o', c='tab:blue')
      plt.plot([x_nojitter], [se_lim_sim2], 'o', c='k')
      plt.plot([x_nojitter], [se_lim_sim3], 'o', c='tab:green')
      # plt.errorbar([x_nojitter], [bias_lim_sim], yerr=[lim_std_sim],
      #     fmt='o', c='tab:blue')
      # yerr = np.abs(rmise_bootstrap_lim_CI- rmise_lim_sim).reshape(-1,1)
      # plt.errorbar([x_nojitter], [rmise_lim_sim], yerr=yerr, fmt='o', capsize=3,
      #     c='tab:blue')
      xticks.append(x_nojitter)
      xticks_label.append('no\nnuisance')
    plt.xticks(xticks, xticks_label)
    # plt.legend(loc=[0.7, 0.05])
    plt.xlim(xlim)
    plt.ylabel('SE [spikes/sec]')
    plt.xlabel(r'$\sigma_w$ [ms]')

    if output_dir is not None:
      file_path = output_dir + experiment_name + 'SE.pdf'
      plt.savefig(file_path)
      print('save figure:', file_path)
    plt.show()


  @classmethod
  def plot_bspline_filter_numerical(
      cls,
      true_par,
      simulation_files,
      experiment_name=None,
      output_dir=None,
      verbose=False):
    """Scan the kernel window width."""
    if true_par['filter_type'] == 'square':
        filter_true_val = true_par['alpha'][0][1]
        filter_length = true_par['beta'][0][1]
        def true_filter(t):
          if np.isscalar(t):
            return filter_true_val if t <= filter_length else 0
          else:
            vals = np.zeros_like(t)
            vals[t <= filter_length] = filter_true_val
            return vals
    min_kernel_width, max_kernel_width = true_par['kernel_width']
    log_kernel_widths = np.linspace(np.log10(min_kernel_width),
                                    np.log10(max_kernel_width), 2000)
    kernel_widths = np.power(10, log_kernel_widths)
    CI_scale = scipy.stats.norm.ppf(0.975)  # 95% CI.
    num_bootstrap_resamples = 400
    xlim = (-0.2, 4)

    #-------------- IRMSE --------------
    dt=0.0002
    num_scenarios = len(simulation_files)
    kernel_widths_sim = np.zeros(num_scenarios)
    rmise_sim = np.zeros(num_scenarios)
    rmise_bootstrap_CI = np.zeros([num_scenarios,2])

    for i, f in enumerate(simulation_files):
      model_par_list = util.load_variable(f)
      if i == 0:
        num_sims = len(model_par_list)
        print(f'num simulation models {num_sims}')
      t, h_vals = cls.extract_filters(model_par_list, dt=dt, verbose=False)
      h_true = true_filter(t)

      if 'kernel_width' in model_par_list[0]:
        kernel_widths_sim[i] = model_par_list[0]['kernel_width']
        has_lim = False
        rmises = np.sqrt(np.mean(np.square(h_vals - h_true), axis=1))
        rmise_sim[i] = np.mean(rmises)
        rmise_bootstrap_CI[i] = cls.bootstrap_rmise_ci(
            h_vals, h_true, ci_alpha=0.05, num_resamples=num_bootstrap_resamples)
      else:
        kernel_widths_sim[i] = np.nan
        has_lim = True
        rmises = np.sqrt(np.mean(np.square(h_vals - h_true), axis=1))
        rmise_lim_sim = np.mean(rmises)
        rmise_bootstrap_lim_CI = cls.bootstrap_rmise_ci(
            h_vals, h_true, ci_alpha=0.05, num_resamples=num_bootstrap_resamples)

      t_plot = t * 1000
      h_mean = h_vals.mean(axis=0)
      h_CI = h_vals.std() / np.sqrt(num_sims) * CI_scale

      gs_kw = dict(width_ratios=[1], height_ratios=[1])
      fig, axs = plt.subplots(figsize=(4, 2), gridspec_kw=gs_kw,
          nrows=1, ncols=1)
      ax = fig.add_subplot(axs)
      plt.fill_between(t_plot, h_mean-h_CI, h_mean+h_CI,
          facecolor='lightblue', alpha=0.3, label='95% CI')
      # plt.plot(t_plot, h_vals.T, c='lightgrey', lw=0.1)
      plt.plot(t_plot, h_vals.mean(axis=0), c='tab:blue', lw=1.5)
      plt.plot(t_plot, h_true, color='k')
      plt.text(0.7, 0.85, fr'$\sigma_w$ = {kernel_widths_sim[i]*1000:.0f} ms',
          transform=ax.transAxes)
      plt.ylim(-1, 3.5)
      plt.xlabel('Time [ms]')
      plt.ylabel('Firing rate [spikes/s]')
      if output_dir is not None:
        file_path = (output_dir + experiment_name +
            f'kernel{kernel_widths_sim[i]*1000:.0f}ms_filter_comp.pdf')
        plt.savefig(file_path)
        print('save figure:', file_path)
      plt.show()


  @classmethod
  def plot_bias_numerical_comparison(
      cls,
      par,
      filter_true_val,
      simulation_files,
      simulation_files2,
      file_path=None):
    """Scan the kernel window width."""
    min_kernel_width, max_kernel_width = par['kernel_width']
    log_kernel_widths = np.linspace(np.log10(min_kernel_width),
                                    np.log10(max_kernel_width), 200)
    kernel_widths = np.power(10, log_kernel_widths)

    bias_list = np.zeros_like(kernel_widths)
    Q_list = np.zeros_like(kernel_widths)
    R_list = np.zeros_like(kernel_widths)

    for i, kernel_width in enumerate(kernel_widths):
      par_input = par.copy()
      par_input['kernel_width'] = kernel_width
      bias_list[i], Q_list[i], R_list[i] = cls.bias_theoretical(par_input)

    root_ind = np.where(np.diff(np.sign(bias_list)) != 0)
    roots = kernel_widths[root_ind]
    nadir_ind = np.argmin(bias_list)
    nadir = kernel_widths[nadir_ind]
    bias_extreme = cls.bias_theoretical_extreme_cases(par)
    print('roots:', roots, '\tnadir:', nadir)

    kernel_widths_sim = []
    bias_sim = []
    bias_std_sim = []
    for i, f in enumerate(simulation_files):
      model_par_list = util.load_variable(f)
      h_vals = cls.extract_filters(model_par_list, verbose=False)
      has_lim = False
      if 'kernel_width' in model_par_list[0]:
        kernel_widths_sim.append(model_par_list[0]['kernel_width'])
        bias_sim.append(np.mean(h_vals) - filter_true_val)
        bias_std_sim.append(np.std(h_vals))
      else:
        has_lim = True
        bias_lim_sim = np.mean(h_vals) - filter_true_val
        bias_lim_std_sim = np.std(h_vals)

    kernel_widths_sim2 = []
    bias_sim2 = []
    bias_std_sim2 = []
    for i, f in enumerate(simulation_files2):
      model_par_list = util.load_variable(f)
      h_vals = cls.extract_filters(model_par_list, verbose=False)
      has_lim2 = False
      if 'kernel_width' in model_par_list[0]:
        kernel_widths_sim2.append(model_par_list[0]['kernel_width'])
        bias_sim2.append(np.mean(h_vals) - filter_true_val)
        bias_std_sim2.append(np.std(h_vals))
      else:
        has_lim2 = True
        bias_lim_sim2 = np.mean(h_vals) - filter_true_val
        bias_lim_std_sim2 = np.std(h_vals)

    gs_kw = dict(width_ratios=[11], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(7, 3), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    ax = fig.add_subplot(axs)
    plt.axhline(0, lw=2, color='lightgrey')
    # plt.plot(np.log10(kernel_widths * 1000), bias_list, 'k', lw=2.5,
    #     label='theoretical')
    # plt.plot([0], [bias_extreme], 'k+', ms=11, label='theoretical limit')
    # plt.plot([3.4], [bias_extreme], 'k+', ms=11)
    plt.ylabel('Bias')
    plt.xlabel(r'$\log \sigma_w$ [log ms]')
    xticks = list(np.arange(0.5, 3.5, 0.5))
    xticks_label = list(np.arange(0.5, 3.5, 0.5))
    x_ninfty = 0
    x_infty = 3.4
    x_nojitter = 3.5
    # xticks = [x_ninfty] + xticks + [x_infty]
    # xticks_label = [r'$-\infty$'] + xticks_label + [r'$\infty$']
    # plt.title('bias')

    plt.errorbar(np.log10(kernel_widths_sim)+3, bias_sim,
        yerr=bias_std_sim, fmt='-o', c='tab:blue', label='squared error loss model')
    if has_lim:
      plt.errorbar([x_nojitter], [bias_lim_sim], yerr=[bias_lim_std_sim],
          fmt='o', c='tab:blue')
      xticks.append(x_nojitter)
      xticks_label.append('no jitter')

    plt.errorbar(np.log10(kernel_widths_sim2)+3, bias_sim2,
        yerr=bias_std_sim2, fmt='-o', c='tab:green', label='likelihood-based model')
    if has_lim:
      plt.errorbar([x_nojitter+0.5], [bias_lim_sim2], yerr=[bias_lim_std_sim2],
          fmt='o', c='tab:green')
      xticks.append(x_nojitter+0.5)
      xticks_label.append('no jitter')

    plt.xticks(xticks, xticks_label)
    plt.legend()  # loc=[1.01, 0.7]

    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      print('save figure:', file_path)
    plt.show()


  @classmethod
  def plot_beta_qq_plot(
      cls,
      par,
      simulation_file,
      correct_center=True,
      experiment_name=None,
      output_dir=None,
      verbose=False):
    """QQ plot to check Normality.

    Args:
      correct_center: If true, shift the theoretical center to  the data center.
    """
    min_kernel_width, max_kernel_width = par['kernel_width']
    log_kernel_widths = np.linspace(np.log10(min_kernel_width),
                                    np.log10(max_kernel_width), 2000)
    kernel_widths = np.power(10, log_kernel_widths)
    CI_scale = scipy.stats.norm.ppf(0.975)  # 95% CI.

    model_par_list = util.load_variable(simulation_file)
    model_par_list = model_par_list[:100]
    num_sims = len(model_par_list)
    kernel_width_sim = model_par_list[0]['kernel_width']
    h_vals_sim = cls.extract_filters(model_par_list, verbose=False)
    print(f'num simulation models {num_sims}\tkernel_width {kernel_width_sim}')

    par_input = par.copy()
    par_input['kernel_width'] = kernel_width_sim
    mean, std = cls.beta_distribution_theoretical(par_input)
    mean_data = np.mean(h_vals_sim)

    x = np.linspace(mean-4*std, mean+4*std, 200)
    pdf = scipy.stats.norm.pdf(x, loc=mean, scale=std)
    pdf_center_corrected = scipy.stats.norm.pdf(x, loc=mean_data, scale=std)
    quantile_thresholds = x
    ecdf = np.zeros_like(quantile_thresholds)
    mcdf = np.zeros_like(quantile_thresholds)
    mcdf_center_corrected = np.zeros_like(quantile_thresholds)
    for i, q in enumerate(quantile_thresholds):
      ecdf[i] = np.sum(h_vals_sim <= q) / num_sims
      mcdf[i] = scipy.stats.norm.cdf(q, loc=mean, scale=std)
      mcdf_center_corrected[i] = scipy.stats.norm.cdf(q, loc=mean_data, scale=std)

    gs_kw = dict(width_ratios=[1, 1], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(6, 1.5), gridspec_kw=gs_kw, nrows=1, ncols=2)
    ax = fig.add_subplot(axs[0])
    seaborn.distplot(h_vals_sim, bins=15, norm_hist=True)
    plt.plot(x, pdf)
    plt.plot(x, pdf_center_corrected)
    plt.axvline(mean)
    plt.axvline(np.mean(h_vals_sim))

    ax = fig.add_subplot(axs[1])
    plt.plot(quantile_thresholds, ecdf)
    plt.plot(quantile_thresholds, mcdf)
    plt.plot(quantile_thresholds, mcdf_center_corrected)
    plt.show()

    # Q-Q plot.
    test_size = 0.05
    c_alpha = np.sqrt(-np.log(test_size / 2) / 2)
    CI_up = mcdf + c_alpha/np.sqrt(num_sims)
    CI_dn = mcdf - c_alpha/np.sqrt(num_sims)

    gs_kw = dict(width_ratios=[1], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(3, 2.8), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    ax = fig.add_subplot(axs)
    plt.plot(mcdf, ecdf, 'k', lw=2)
    if correct_center:
      plt.plot(mcdf_center_corrected, ecdf, 'k--', lw=2)

    # plt.plot([0,1], [0,1], color='lightgrey', lw=1)
    plt.xticks([0,1], [0,1])
    plt.yticks([0,1], [0,1])
    plt.plot(mcdf, CI_up, '--', c='lightgrey', lw=1, label='95% CI')
    plt.plot(mcdf, CI_dn, '--', c='lightgrey', lw=1)
    plt.axis([0,1,0,1])
    plt.xlabel('Theoretical quantile')
    plt.ylabel('Numerical quantile')
    # plt.legend()
    plt.text(0.03, 0.9, fr'$\sigma_w$ = {kernel_width_sim*1000:.0f} ms',
        fontsize=12, transform=ax.transAxes)

    if output_dir is not None:
      file_path = (output_dir + experiment_name +
          f'kernel{kernel_width_sim*1000:.0f}ms_QQ_plot.pdf')
      plt.savefig(file_path, bbox_inches='tight')
      print('save figure:', file_path)
    plt.show()


  @classmethod
  def extract_nll(
      cls,
      model_par_list,
      center=0,
      verbose=False):
    """Inference for the regression."""
    num_models = len(model_par_list)
    vals = np.zeros(num_models)

    for m in range(num_models):
      vals[m] = model_par_list[m]['nll']

    return vals


  @classmethod
  def estimate_optimal_jitter_window_width_regression(
      cls,
      spike_times_x, spike_times_y,
      model_par,
      kernel_width_grid=None,
      output_dir=None):
    """Calculate Sxx from raw data."""
    trial_window = model_par['trial_window']
    num_trials = len(spike_times_y)
    num_spikes_x = [len(spikes) for spikes in spike_times_x]
    num_spikes_x = np.sum(num_spikes_x)
    num_spikes_y = [len(spikes) for spikes in spike_times_y]
    num_spikes_y = np.sum(num_spikes_y)
    trial_length = trial_window[1]
    model_par['num_trials'] = num_trials
    model_par['trial_length'] = trial_length
    bar_lambda_i = num_spikes_x / num_trials / trial_length
    bar_lambda_j = num_spikes_y / num_trials / trial_length
    # Coupling filter. The estimator is not sensitive to this.
    if 'filter_type' not in model_par:
      model_par['filter_type'] = 'square'
      model_par['filter_length'] = 0.03
    if 'learning_rate' not in model_par:
      model_par['learning_rate'] = 0.5
    if 'max_num_itrs' not in model_par:
      model_par['max_num_itrs'] = 40
    if 'epsilon' not in model_par:
      model_par['epsilon'] = 1e-6
    if 'const_offset' not in model_par:
      model_par['const_offset'] = 0

    # Numerical. # The columns are: const, W, h, y.
    # kernel_widths_sim = np.array([35]) / 1000
    if model_par['append_nuisance'][1] == 'gaussian_kernel':
      kernel_widths_sim = np.array([
          2,3,5,10,20,30,40,45,50,55,60,65,70,75,80,85,90,95,
          100,105,110,120,125,130,135,140,145,150,155,160,200,400]) / 1000
    elif model_par['append_nuisance'][1] == 'triangle_kernel':
      kernel_widths_sim = np.array([
          40,80,100,120,130,150,160,180,
          200,250,400,420,430,450, 480,500,550,600,620,650,800,1000]) / 1000
    else:
      kernel_widths_sim = np.array([
          10,20,30,40,45,50,55,60,65,70,80,90,95,
          100,105,110,120,125,130,135,140,145,150,155,160,180,
          200,250,300,350,400,420,430,450,480,
          500,550,600,620,650,800,1000]) / 1000

    if kernel_width_grid is not None:
      kernel_widths_sim = kernel_width_grid

    num_scenarios = len(kernel_widths_sim)
    Sxx_hat = np.zeros([num_scenarios,5,5])
    log_likeli_hat = np.zeros(num_scenarios)

    for k, kernel_width in tqdm(enumerate(kernel_widths_sim), file=sys.stdout,
        ncols=100, total=len(kernel_widths_sim)):
      par_input = model_par.copy()
      par_input['kernel_width'] = kernel_width
      model_par_hat = cls.bivariate_continuous_time_coupling_filter_regression(
          spike_times_x, spike_times_y, trial_window, par_input, verbose=False)
      log_likeli_hat[k] = -model_par_hat['nll']

    # Align the peak of the log-likelihood to the peak.
    log_likeli_hat = log_likeli_hat - np.max(log_likeli_hat)
    xlim = (0, 3.2)
    xticks_label = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    xticks = np.log10(np.array(xticks_label))
    xticks = list(xticks)
    max_ind = np.argmax(log_likeli_hat)
    max_ll_kernel_width = kernel_widths_sim[max_ind]
    print(f'optimal kernel width  {max_ll_kernel_width * 1000}')

    gs_kw = dict(width_ratios=[1], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(6, 2), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    ax = fig.add_subplot(axs)
    plt.axvline(np.log10(max_ll_kernel_width)+3, color='lightgrey', lw=1)
    plt.plot(np.log10(kernel_widths_sim)+3, log_likeli_hat,
        '.-', c='tab:blue', label='numerical')
    plt.xlim(xlim)
    plt.xticks(xticks, xticks_label)
    plt.ylabel('log-likelihood')
    plt.xlabel(r'$\sigma_w$ [ms]')

    if output_dir is not None:
      file_path = output_dir + experiment_name + f'Sxx_{basis_i}_{basis_j}.pdf'
      plt.savefig(file_path)
      print('save figure:', file_path)
    plt.show()

    return kernel_widths_sim, log_likeli_hat, max_ll_kernel_width


  @classmethod
  def estimate_optimal_jitter_window_width_full_regression(
      cls,
      spike_times_x, spike_times_y,
      model_par,
      kernel_width_grid=None,
      output_dir=None):
    """Calculate Sxx from raw data."""
    trial_window = model_par['trial_window']
    num_trials = len(spike_times_y)
    num_spikes_x = [len(spikes) for spikes in spike_times_x]
    num_spikes_x = np.sum(num_spikes_x)
    num_spikes_y = [len(spikes) for spikes in spike_times_y]
    num_spikes_y = np.sum(num_spikes_y)
    trial_length = trial_window[1]
    model_par['num_trials'] = num_trials
    model_par['trial_length'] = trial_length
    bar_lambda_i = num_spikes_x / num_trials / trial_length
    bar_lambda_j = num_spikes_y / num_trials / trial_length
    # Coupling filter. The estimator is not sensitive to this.
    if 'filter_type' not in model_par:
      model_par['filter_type'] = 'square'
      model_par['filter_length'] = 0.03
    if 'learning_rate' not in model_par:
      model_par['learning_rate'] = 0.5
    if 'max_num_itrs' not in model_par:
      model_par['max_num_itrs'] = 40
    if 'epsilon' not in model_par:
      model_par['epsilon'] = 1e-6
    if 'const_offset' not in model_par:
      model_par['const_offset'] = 0

    # Numerical. # The columns are: const, W, h, y.
    # kernel_widths_sim = np.array([35]) / 1000
    if model_par['append_nuisance'][1] == 'gaussian_kernel':
      kernel_widths_sim = np.array([
          2,3,5,10,20,30,40,45,50,55,60,65,70,75,80,85,90,95,
          100,105,110,120,125,130,135,140,145,150,155,160,200,400]) / 1000
    elif model_par['append_nuisance'][1] == 'triangle_kernel':
      kernel_widths_sim = np.array([
          40,80,100,120,130,150,160,180,
          200,250,400,420,430,450, 480,500,550,600,620,650,800,1000]) / 1000
    else:
      kernel_widths_sim = np.array([
          10,20,30,40,45,50,55,60,65,70,80,90,95,
          100,105,110,120,125,130,135,140,145,150,155,160,180,
          200,250,300,350,400,420,430,450,480,
          500,550,600,620,650,800,1000]) / 1000

    if kernel_width_grid is not None:
      kernel_widths_sim = kernel_width_grid

    num_scenarios = len(kernel_widths_sim)
    Sxx_hat = np.zeros([num_scenarios,5,5])
    log_likeli_hat = np.zeros(num_scenarios)

    for k, kernel_width in tqdm(enumerate(kernel_widths_sim), file=sys.stdout,
        ncols=100, total=len(kernel_widths_sim)):
      par_input = model_par.copy()
      par_input['kernel_width'] = kernel_width
      model_par_hat = cls.bivariate_continuous_time_coupling_filter_full_regression(
          spike_times_x, spike_times_y, trial_window, par_input, verbose=False)
      log_likeli_hat[k] = -model_par_hat['nll']

    # Align the peak of the log-likelihood to the peak.
    log_likeli_hat = log_likeli_hat - np.max(log_likeli_hat)
    xlim = (0, 3.2)
    xticks_label = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    xticks = np.log10(np.array(xticks_label))
    xticks = list(xticks)
    max_ind = np.argmax(log_likeli_hat)
    max_ll_kernel_width = kernel_widths_sim[max_ind]
    print(f'optimal kernel width  {max_ll_kernel_width * 1000}')

    gs_kw = dict(width_ratios=[1], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(6, 2), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    ax = fig.add_subplot(axs)
    plt.axvline(np.log10(max_ll_kernel_width)+3, color='lightgrey', lw=1)
    plt.plot(np.log10(kernel_widths_sim)+3, log_likeli_hat,
        '.-', c='tab:blue', label='numerical')
    plt.xlim(xlim)
    plt.xticks(xticks, xticks_label)
    plt.ylabel('log-likelihood')
    plt.xlabel(r'$\sigma_w$ [ms]')

    if output_dir is not None:
      file_path = output_dir + experiment_name + f'Sxx_{basis_i}_{basis_j}.pdf'
      plt.savefig(file_path)
      print('save figure:', file_path)
    plt.show()

    return kernel_widths_sim, log_likeli_hat, max_ll_kernel_width


  @classmethod
  def estimate_optimal_jitter_window_width(
      cls,
      spike_times_x, spike_times_y,
      model_par,
      kernel_width_grid=None,
      output_dir=None):
    """Calculate Sxx from raw data."""
    dt = model_par['dt']
    trial_window = model_par['trial_window']
    num_trials = len(spike_times_y)
    num_spikes_x = [len(spikes) for spikes in spike_times_x]
    num_spikes_x = np.sum(num_spikes_x)
    num_spikes_y = [len(spikes) for spikes in spike_times_y]
    num_spikes_y = np.sum(num_spikes_y)
    trial_length = trial_window[1]
    model_par['num_trials'] = num_trials
    model_par['trial_length'] = trial_length
    bar_lambda_i = num_spikes_x / num_trials / trial_length
    bar_lambda_j = num_spikes_y / num_trials / trial_length
    # Coupling filter. The estimator is not sensitive to this.
    alpha_h = 0
    if 'filter_type' not in model_par:
      model_par['filter_type'] = 'square'
      model_par['filter_length'] = 0.03

    # Numerical. # The columns are: const, W, h, y.
    # kernel_widths_sim = np.array([35]) / 1000
    if model_par['append_nuisance'][1] == 'gaussian_kernel':
      kernel_widths_sim = np.array([
          2,3,5,10,20,30,40,45,50,55,60,65,70,75,80,85,90,95,
          100,105,110,120,125,130,135,140,145,150,155,160,200,400]) / 1000
    elif model_par['append_nuisance'][1] == 'triangle_kernel':
      kernel_widths_sim = np.array([
          40,80,100,120,130,150,160,180,
          200,250,400,420,430,450, 480,500,550,600,620,650,800,1000]) / 1000
    else:
      kernel_widths_sim = np.array([
          10,20,30,40,45,50,55,60,65,70,80,90,95,
          100,105,110,120,125,130,135,140,145,150,155,160,180,
          200,250,300,350,400,420,430,450,480,
          500,550,600,620,650,800,1000]) / 1000

    if kernel_width_grid is not None:
      kernel_widths_sim = kernel_width_grid

    num_scenarios = len(kernel_widths_sim)
    Sxx_hat = np.zeros([num_scenarios,5,5])
    log_likeli_hat = np.zeros(num_scenarios)

    for k, kernel_width in tqdm(enumerate(kernel_widths_sim), file=sys.stdout,
        ncols=100, total=len(kernel_widths_sim)):
      par_input = model_par.copy()
      par_input['kernel_width'] = kernel_width
      spike_hist_stacked_x, _ = cls.bin_spike_times(spike_times_x, dt, trial_length)
      (num_basis, num_samples, X, basis_integral, offset, offset_integral
          ) = cls.bivariate_discrete_time_coupling_filter_build_regressors(
          spike_hist_stacked_x, trial_window, par_input, mean_norm=True)
      spike_hist_stacked, _ = cls.bin_spike_times(spike_times_y, dt, trial_length)
      spike_hist = spike_hist_stacked.reshape(-1, 1)
      XY = np.hstack((X, spike_hist))
      S_xx = XY.T @ XY
      for (i,j) in [(1,1),(1,2),(2,2),(1,3),(2,3)]:
        if (i == 1 and j == 3) or (i == 2 and j == 3):
          Sxx_hat[k,i,j] = S_xx[i,3]
        elif (i == 1 and j == 4) or (i == 2 and j == 4):
          Sxx_hat[k,i,j] = S_xx[i,3] - alpha_h * S_xx[i,2] * dt
        else:
          Sxx_hat[k,i,j] = S_xx[i,j] * dt
      S_ww = Sxx_hat[k,1,1]
      S_hw = Sxx_hat[k,1,2]
      S_hh = Sxx_hat[k,2,2]
      S_wy = Sxx_hat[k,1,3]
      S_hy = Sxx_hat[k,2,3]
      b = np.array([[S_wy],[S_hy]])
      H = np.array([[S_ww, S_hw],[S_hw, S_hh]])
      log_likeli_hat[k] = 1/2/bar_lambda_j * b.T @ np.linalg.inv(H) @ b

    # Align the peak of the log-likelihood to the peak.
    log_likeli_hat = log_likeli_hat - np.max(log_likeli_hat)
    xlim = (0, 3.2)
    xticks_label = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    xticks = np.log10(np.array(xticks_label))
    xticks = list(xticks)
    max_ind = np.argmax(log_likeli_hat)
    max_ll_kernel_width = kernel_widths_sim[max_ind]
    print(f'optimal kernel width  {max_ll_kernel_width * 1000}')

    gs_kw = dict(width_ratios=[1], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(6, 2), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    ax = fig.add_subplot(axs)
    plt.axvline(np.log10(max_ll_kernel_width)+3, color='lightgrey', lw=1)
    plt.plot(np.log10(kernel_widths_sim)+3, log_likeli_hat,
        '.-', c='tab:blue', label='numerical')
    plt.xlim(xlim)
    plt.xticks(xticks, xticks_label)
    plt.ylabel('log-likelihood')
    plt.xlabel(r'$\sigma_w$ [ms]')

    if output_dir is not None:
      file_path = output_dir + experiment_name + f'Sxx_{basis_i}_{basis_j}.pdf'
      plt.savefig(file_path)
      print('save figure:', file_path)
    plt.show()

    return kernel_widths_sim, log_likeli_hat, max_ll_kernel_width


  @classmethod
  def estimate_optimal_jitter_window_width_simple(
      cls,
      spike_times_x, spike_times_y,
      model_par,
      output_dir=None):
    """Calculate Sxx from raw data."""
    dt = model_par['dt']
    trial_window = model_par['trial_window']
    num_trials = len(spike_times_y)
    num_spikes_x = [len(spikes) for spikes in spike_times_x]
    num_spikes_x = np.sum(num_spikes_x)
    num_spikes_y = [len(spikes) for spikes in spike_times_y]
    num_spikes_y = np.sum(num_spikes_y)
    trial_length = trial_window[1]
    model_par['num_trials'] = num_trials
    model_par['trial_length'] = trial_length
    bar_lambda_i = num_spikes_x / num_trials / trial_length
    bar_lambda_j = num_spikes_y / num_trials / trial_length
    # Coupling filter. The estimator is not sensitive to this.
    alpha_h = 0
    if 'filter_type' not in model_par:
      model_par['filter_type'] = 'square'
      model_par['filter_length'] = 0.03

    # Numerical. # The columns are: const, W, h, y.
    # kernel_widths_sim = np.array([35]) / 1000
    if model_par['append_nuisance'][1] == 'gaussian_kernel':
      kernel_widths_sim = np.array([
          2,3,5,10,20,30,40,45,50,55,60,65,70,75,80,90,95,
          100,105,110,120,125,130,135,140,145,150,155,160,200,400]) / 1000
    elif model_par['append_nuisance'][1] == 'triangle_kernel':
      kernel_widths_sim = np.array([
          40,80,100,120,130,150,160,180,
          200,250,400,420,430,450, 480,500,550,600,620,650,800,1000]) / 1000
    else:
      kernel_widths_sim = np.array([
          10,20,30,40,45,50,55,60,65,70,80,90,95,
          100,105,110,120,125,130,135,140,145,150,155,160,180,
          200,250,300,350,400,420,430,450,480,
          500,550,600,620,650,800,1000]) / 1000

    num_scenarios = len(kernel_widths_sim)
    Sxx_hat = np.zeros([num_scenarios,5,5])
    log_likeli_hat = np.zeros(num_scenarios)

    for k, kernel_width in enumerate(kernel_widths_sim):
      par_input = model_par.copy()
      par_input['kernel_width'] = kernel_width
      spike_hist_stacked_x, _ = cls.bin_spike_times(spike_times_x, dt, trial_length)
      (num_basis, num_samples, X, basis_integral, offset, offset_integral
          ) = cls.bivariate_discrete_time_coupling_filter_build_regressors(
          spike_hist_stacked_x, trial_window, par_input, mean_norm=True)
      spike_hist_stacked, _ = cls.bin_spike_times(spike_times_y, dt, trial_length)
      spike_hist = spike_hist_stacked.reshape(-1, 1)
      XY = np.hstack((X, spike_hist))
      S_xx = XY.T @ XY
      for (i,j) in [(1,1),(1,2),(2,2),(1,3),(2,3)]:
        if (i == 1 and j == 3) or (i == 2 and j == 3):
          Sxx_hat[k,i,j] = S_xx[i,3]
        elif (i == 1 and j == 4) or (i == 2 and j == 4):
          Sxx_hat[k,i,j] = S_xx[i,3] - alpha_h * S_xx[i,2] * dt
        else:
          Sxx_hat[k,i,j] = S_xx[i,j] * dt
      S_ww = Sxx_hat[k,1,1]
      S_hw = Sxx_hat[k,1,2]
      S_hh = Sxx_hat[k,2,2]
      S_wy = Sxx_hat[k,1,3]
      S_hy = Sxx_hat[k,2,3]
      # b = np.array([[S_wy],[S_hy]])
      # H = np.array([[S_ww, S_hw],[S_hw, S_hh]])
      # log_likeli_hat[k] = 1/2/bar_lambda_j * b.T @ np.linalg.inv(H) @ b
      b = S_wy
      H = S_ww
      log_likeli_hat[k] = 1/2/bar_lambda_j * b * b / H

    # Align the peak of the log-likelihood to the peak.
    log_likeli_hat = log_likeli_hat - np.max(log_likeli_hat)
    xlim = (0, 3.2)
    xticks_label = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    xticks = np.log10(np.array(xticks_label))
    xticks = list(xticks)
    max_ind = np.argmax(log_likeli_hat)
    max_ll_kernel_width = kernel_widths_sim[max_ind]
    print(f'optimal kernel width  {max_ll_kernel_width * 1000}')

    gs_kw = dict(width_ratios=[1], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(6, 2), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    ax = fig.add_subplot(axs)
    # plt.axvline(np.log10(max_ll_kernel_width)+3, color='lightgrey', lw=1)
    plt.plot(np.log10(kernel_widths_sim)+3, log_likeli_hat,
        '.-', c='tab:blue', label='numerical')
    plt.xlim(xlim)
    plt.xticks(xticks, xticks_label)
    plt.ylabel('log-likelihood')
    plt.xlabel(r'$\sigma_w$ [ms]')

    if output_dir is not None:
      file_path = output_dir + experiment_name + f'Sxx_{basis_i}_{basis_j}.pdf'
      plt.savefig(file_path)
      print('save figure:', file_path)
    plt.show()

    return kernel_widths_sim, log_likeli_hat


  @classmethod
  def plot_multiple_plugin_estimator_likelihood_curves(
      cls,
      kernel_widths_sim_list,
      log_likeli_hat_list,
      file_path=None):
    """Plot Multiple likelihood curves together."""
    num_curves = len(kernel_widths_sim_list)
    print('num_curves', num_curves)

    gs_kw = dict(width_ratios=[1], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(6, 2), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    ax = fig.add_subplot(axs)
    # for i, kernel_widths_sim in enumerate(kernel_widths_sim_list):
    kernel_widths_sim = kernel_widths_sim_list[0]
    log_likeli_hat = log_likeli_hat_list[0]
    plt.plot(np.log10(kernel_widths_sim)+3, log_likeli_hat,
        '.-', ms=2, c='k', label='No fast background')

    kernel_widths_sim = kernel_widths_sim_list[1]
    log_likeli_hat = log_likeli_hat_list[1]
    plt.plot(np.log10(kernel_widths_sim)+3, log_likeli_hat,
        '.-', ms=2, c='tab:blue', label='With fast background')

    plt.legend()
    xlim = (0, 3.2)
    xticks_label = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    xticks = np.log10(np.array(xticks_label))
    xticks = list(xticks)
    plt.xlim(xlim)
    plt.xticks(xticks, xticks_label)
    plt.ylabel('log-likelihood')
    plt.xlabel(r'$\sigma_w$ [ms]')

    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      print('save figure:', file_path)
    plt.show()




  @classmethod
  def bivariate_continuous_time_coupling_filter_regression_jitter(
      cls,
      spike_times_x,
      spike_times_y,
      trial_window,
      model_par,
      verbose=False):
    """Bivariate continuous-time PP-GLM with jittered inputs."""
    num_jitter = model_par['num_jitter']
    jitter_window_width = model_par['jitter_window_width']

    # Raw statistic.
    model_par_raw = cls.bivariate_continuous_time_coupling_filter_regression(
        spike_times_x, spike_times_y, trial_window, model_par.copy())

    # Jittered models.
    model_par_jitter = [0] * num_jitter
    if verbose:
      trange = tqdm(range(num_jitter), ncols=100, file=sys.stdout)
    else:
      trange = range(num_jitter)
    for r in trange:
      # spike_times_x_surrogate = cls.jitter_spike_times_basic(
      #     spike_times_x, jitter_window_width, num_jitter=1, verbose=False)
      spike_times_x_surrogate = cls.jitter_spike_times_interval(
          spike_times_x, jitter_window_width, num_jitter=1, verbose=False)
      # spike_times_y_surrogate = cls.jitter_spike_times_interval(
      #     spike_times_y, jitter_window_width, num_jitter=1, verbose=False)

      model_par_hat = cls.bivariate_continuous_time_coupling_filter_regression(
          spike_times_x_surrogate, spike_times_y,
          trial_window, model_par.copy())
      model_par_jitter[r] = model_par_hat.copy()
      # For later warm start.
      # model_par['beta_init'] = model_par_hat['beta']

    return model_par_raw, model_par_jitter


  @classmethod
  def get_beta_mean(
      cls,
      model_par_list):
    """Get the mean of beta from `model_par_list`."""
    num_models = len(model_par_list)
    num_basis = model_par_list[0]['num_basis']
    # print(f'num_models {num_models}, num_basis {num_basis}')
    betas = np.zeros([num_models, num_basis])
    for m in range(num_models):
      betas[m] = model_par_list[m]['beta'].reshape(-1)
    beta_mean = betas.mean(axis=0)
    return beta_mean.reshape(num_basis, 1)


  @classmethod
  def plot_continuous_time_bivariate_regression_jitter_model_par(
      cls,
      model_par_raw,
      model_par_jitter,
      filter_par=None,
      ylim=None,
      file_path=None):
    """Visualize estimated filters."""
    num_jitter = len(model_par_jitter)
    num_basis = model_par_raw['num_basis']
    filter_length = model_par_raw['filter_length']

    if model_par_raw['filter_type'] == 'bspline':
      sub_axs, t, h_raw, h_raw_std = cls.reconstruct_bspline_basis(
          model_par_raw, verbose=False)
      num_bins = len(t)

      # Jittered estimators.
      h_jitter = np.zeros([num_jitter, num_bins])
      for r in range(num_jitter):
        _, _, h_jitter[r], _ = cls.reconstruct_bspline_basis(
              model_par_jitter[r], verbose=False)
      h_CI_up = np.quantile(h_jitter, 0.975, axis=0)
      h_CI_down = np.quantile(h_jitter, 0.025, axis=0)
      h_mean = np.mean(h_jitter, axis=0)

    if model_par_raw['filter_type'] == 'square':
      t, h_raw, h_raw_std = cls.reconstruct_basis(model_par_raw)
      num_bins = len(t)

      # Jittered estimators.
      h_jitter = np.zeros([num_jitter, num_bins])
      for r in range(num_jitter):
        _, h_jitter[r], _ = cls.reconstruct_basis(model_par_jitter[r])
      h_CI_up = np.quantile(h_jitter, 0.975, axis=0)
      h_CI_down = np.quantile(h_jitter, 0.025, axis=0)
      h_mean = np.mean(h_jitter, axis=0)

    gs_kw = dict(width_ratios=[1, 1, 1], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(16, 3), gridspec_kw=gs_kw,
        nrows=1, ncols=3)
    # plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)

    # Raw estimator.
    ax = fig.add_subplot(axs[0])
    plt.axhline(y=0, c='lightgrey')
    plt.axvline(x=0, c='lightgrey')
    if filter_par is not None and filter_par['type'] == 'triangle':
      filter_alpha = filter_par['alpha'][0][1]
      filter_beta = filter_par['beta'][0][1]
      ax.plot([0, filter_beta], [filter_alpha, 0], 'g', label='True filter')
    elif filter_par is not None and filter_par['type'] == 'square':
      filter_alpha = filter_par['alpha'][0][1]
      filter_beta = filter_par['beta'][0][1]
      ax.plot([0, filter_beta], [filter_alpha, filter_alpha], 'g',
          label='True filter')
    elif filter_par is not None and filter_par['type'] == 'exp':
      filter_alpha = filter_par['alpha'][0][1]
      filter_beta = filter_par['beta'][0][1]
      y = filter_alpha * np.exp(-filter_beta * t)
      ax.plot(t, y, 'g', label='True filter')

    plt.plot(t, h_raw, c='k', label='Filter')
    CI_scale = scipy.stats.norm.ppf(0.975)  # 95% CI.
    plt.fill_between(t, h_raw+h_raw_std*CI_scale, h_raw-h_raw_std*CI_scale,
                     facecolor='lightgrey', alpha=0.3, label='95% CI')
    xticks = np.arange(0, filter_length, 0.01)
    plt.xticks(xticks, xticks*1000)
    plt.xlabel('Time [ms]')
    plt.ylabel('Firing rate [spk/sec]')
    plt.ylim(ylim)
    plt.xlim(0)
    plt.title('Raw data estimator')
    plt.legend(ncol=1)

    ax = fig.add_subplot(axs[1])
    plt.axhline(y=0, c='lightgrey')
    plt.axvline(x=0, c='lightgrey')
    # plt.plot(t, h_jitter.T, c='grey', lw=0.8)
    plt.plot(t, h_raw, ls='-', c='k', lw=2, label='raw')
    plt.plot(t, h_mean, ls='--', c='grey', lw=1, label='jitter mean')
    # plt.plot(t, h_tmp, ls='--', c='g', lw=1)
    plt.fill_between(t, h_CI_up, h_CI_down, facecolor='lightgrey', alpha=0.3,
                     label='jitter 95% CI')
    xticks = np.arange(0, filter_length, 0.01)
    plt.xticks(xticks, xticks*1000)
    plt.xlabel('Time [ms]')
    plt.ylabel('Firing rate [spk/sec]')
    plt.ylim(ylim)
    plt.xlim(0)
    plt.legend(ncol=1)
    plt.title('Jittered estimators')

    ax = fig.add_subplot(axs[2])
    if filter_par is not None and filter_par['type'] == 'square':
      filter_alpha = filter_par['alpha'][0][1]
      filter_beta = filter_par['beta'][0][1]
      ax.plot([0, filter_beta], [filter_alpha, filter_alpha], 'g',
          label='True filter')
      # plt.ylim(-filter_alpha*0.3, filter_alpha*1.3)
    plt.axhline(y=0, c='lightgrey')
    plt.axvline(x=0, c='lightgrey')
    plt.plot(t, h_raw-h_mean, ls='-', c='k', lw=2, label='raw')
    plt.fill_between(t, h_CI_up-h_mean, h_CI_down-h_mean,
        facecolor='lightgrey', alpha=0.3, label='jitter 95% CI')
    xticks = np.arange(0, filter_length, 0.01)
    plt.xticks(xticks, xticks*1000)
    plt.xlabel('Time [ms]')
    plt.ylabel('Firing rate [spk/sec]')
    plt.ylim(ylim)
    plt.xlim(0)
    plt.legend(ncol=1)
    plt.title('Jittered estimators mean corrected')

    if file_path is not None:
      # plt.savefig(file_path, bbox_inches='tight')
      # print('save figure:', file_path)
      plt.close()
    plt.show()
    return

    gs_kw = dict(width_ratios=[1], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(5, 3), gridspec_kw=gs_kw,
        nrows=1, ncols=1)
    # plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)

    ax = fig.add_subplot(axs)
    plt.axhline(y=0, c='lightgrey')
    plt.axvline(x=0, c='lightgrey')
    plt.plot(t, h_raw-h_mean, ls='-', c='k', lw=2, label='Filter')
    plt.fill_between(t, h_CI_up-h_mean, h_CI_down-h_mean,
        facecolor='lightgrey', alpha=0.3, label='Jitter 95% CI')
    xticks = np.arange(0, filter_length, 0.01)
    plt.xticks(xticks, xticks*1000)
    plt.xlabel('Time [ms]')
    plt.ylabel('Firing rate [spk/sec]')
    plt.ylim(ylim)
    plt.xlim(0)
    plt.legend(ncol=1)
    plt.title('Jitter corrected filter')

    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      print('save figure:', file_path)
      plt.close()
    plt.show()


  @classmethod
  def bivariate_continuous_time_coupling_filter_regression_jitter_inference_single(
      cls,
      model_par,
      verbose=False):
    """Inference for the regression."""
    model_par_raw, model_par_jitter = model_par
    filter_length = model_par_raw['filter_length']
    num_nuisance = len(model_par_raw['append_nuisance'])
    num_jitter = len(model_par_jitter)

    if model_par_raw['filter_type'] == 'square':
      _, h_raw, _ = cls.reconstruct_basis(model_par_raw)
      h_raw = h_raw[0]

      # Jittered estimators.
      h_jitter = np.zeros([num_jitter])
      for r in range(num_jitter):
        _, h_filter, _ = cls.reconstruct_basis(model_par_jitter[r])
        h_jitter[r] = h_filter[0]

      cdf = (sum(h_raw > h_jitter) + 1) / (num_jitter + 1)
      one_side_p_val = min(cdf, 1 - cdf)
      two_side_p_val = one_side_p_val * 2

      if verbose:
        print(f'p-val:{two_side_p_val:.2e}')
      return two_side_p_val


  @classmethod
  def bivariate_continuous_time_coupling_filter_regression_jitter_inference(
      cls,
      model_par_list,
      verbose=False):
    """Inference for the regression."""
    num_models = len(model_par_list)
    p_vals = np.zeros(num_models)
    print(f'num_models:{num_models}')

    for m in range(num_models):
      p_vals[m] = cls.bivariate_continuous_time_coupling_filter_regression_jitter_inference_single(
          model_par_list[m], verbose=verbose)

    return p_vals


  @classmethod
  def check_ks(
      cls,
      u_list,
      test_size=0.05,
      bin_width=0.01,
      null_cdf=None,
      lmbd=None,
      verbose=False,
      figure_path=None):
    """Plot the Q-Q curve.

    Calculation of CI bandwidth:
    https://en.wikipedia.org/wiki/Kolmogorov-Smirnov_test

    Args:
      null_cdf: Used for null distribution correction.
    """
    num_samples = len(u_list)
    if num_samples == 0:
      return True, None, None, None, None

    bins = np.linspace(0, 1, int(1 / bin_width) + 1)
    c_alpha = np.sqrt(-np.log(test_size / 2) / 2)
    epdf, bin_edges = np.histogram(u_list, bins=bins, density=True)
    ecdf = np.cumsum(epdf) * bin_width

    if null_cdf is None:
      mcdf = bin_edges[1:]
    else:
      mcdf = null_cdf

    CI_up = mcdf + c_alpha/np.sqrt(num_samples)
    CI_dn = mcdf - c_alpha/np.sqrt(num_samples)
    CI_trap = ((ecdf > CI_up) | (ecdf < CI_dn)).sum()

    # if verbose and CI_trap > 0:
    if verbose:
      plt.figure(figsize=[4, 1.8])
      plt.subplot(121)
      plt.plot(mcdf, ecdf)
      plt.plot(mcdf, CI_up, 'k--')
      plt.plot(mcdf, CI_dn, 'k--')
      # plt.plot(mcdf, mcdf, 'k--')
      plt.title(f'Trap {CI_trap}  N{num_samples}')
      plt.axis([0,1,0,1])
      plt.xticks([0, 1], [0, 1])
      plt.yticks([0, 1], [0, 1])
      # plt.grid('on')
      plt.subplot(122)
      seaborn.distplot(u_list, bins=30,
          norm_hist=True, kde=False, color='grey')
      plt.plot([0, 1], [1, 1], 'k')
      # plt.ylim(0, 1.5)
      plt.xlim(0, 1)
      plt.xticks([0, 1], [0, 1])
      plt.yticks([], [])
      plt.show()

    # This part is for paper KS bias analysis.
    # if figure_path is not None:
    #   plt.figure(figsize=[3, 2.5])
    #   seaborn.distplot(u_list, bins=30,
    #       norm_hist=True, kde=False, color='grey')
    #   plt.plot([0, 1], [1, 1], 'k')
    #   plt.xlim(0, 1)
    #   plt.ylim(0, 1.5)
    #   plt.xticks([0, 1], [0, 1])
    #   plt.yticks([0, 1], [0, 1])
    #   # plt.title('FR=30 Hz   Len=2 s')
    #   # plt.tight_layout()
    #   plt.savefig(figure_path)
    #   print('Save figure to:', figure_path)
    #   plt.show()
        # This part is for paper KS bias analysis.
    if figure_path is not None:
      # plt.figure(figsize=[3, 2.7])
      plt.figure(figsize=[2.2, 2])
      plt.plot(mcdf, CI_up, '--', color='lightgrey')
      plt.plot(mcdf, CI_dn, '--', color='lightgrey')
      # plt.plot(mcdf, mcdf, 'k--')
      plt.plot(mcdf, ecdf, 'k')
      plt.axis([0,1,0,1])
      plt.xticks([0, 1], [0, 1])
      plt.yticks([0, 1], [0, 1])
      plt.xlabel('Theoretial quantile')
      plt.ylabel('Empirical quantile')
      plt.savefig(figure_path)
      print('Save figure to:', figure_path)
      plt.show()

    return CI_trap == 0, mcdf, ecdf, CI_up, CI_dn


  @classmethod
  def ks_test_interval_transform(
      cls,
      spikes,
      lmbd,
      t_end=None,
      dt=0.002,
      bins=None,
      verbose=False):
    """Unit KS test. This is based on time rescale theorem.

    Args:
      lmbd: Intensity function. Each bin represent the integral over dt.
      bins: Timeline for lmbd.
      spikes: Array of spike times.
    """
    # if len(spikes) < 2:
    #   return []

    # if (len(spikes) > 0 and spikes[0] != 0) or len(spikes) == 0:
    #   spikes = np.insert(spikes, 0, 0)
    if len(spikes) == 0:
      return []

    if (t_end is not None and append_interval_end and len(spikes) > 0 and
        spikes[-1] < t_end):
      spikes = np.append(spikes, t_end)

    if bins is not None:
      t_start = bins[0]
    else:
      t_start = 0

    u_list = []
    for spike_id in range(1, len(spikes)):
      # Not include the start bin.
      interval_left = int((spikes[spike_id-1]-t_start) // dt + 1)
      # Include the end bin. Python array index does not include the last index.
      interval_right = int((spikes[spike_id]-t_start) // dt)

      # Haslinger correction.
      r = np.random.rand()
      lmbd_last = lmbd[interval_right]
      # p_last = 1 - np.exp(-lmbd_last)
      p_last = 1 - np.exp(-np.clip(lmbd_last, a_min=-10, a_max=10))

      delta = -np.log(1 - r * p_last)
      # delta = 0  # No correction.

      tau = lmbd[interval_left:interval_right].sum() + delta
      # u_val = 1 - np.exp(-tau)
      u_val = 1 - np.exp(-np.clip(tau, a_min=-10, a_max=10))
      u_list.append(u_val)

    if verbose:
      plt.figure(figsize=(3,3))
      seaborn.distplot(u_list, bins=30, norm_hist=True, kde=False, color='grey')
      plt.plot([0, 1], [1, 1], 'k')

    return u_list


  @classmethod
  def get_ks_test_null_cdf(
      cls,
      lmbd,
      t_end=None,
      bins=None,
      num_trials=1000,
      censor_interval=None,
      verbose=False):
    """Gets the null CDF using simulation.

    Args:
      lmbd: Single trial intensity function.
    """
    import hierarchical_model_generator
    num_bins = len(lmbd)
    u_list = []
    for i in range(num_trials):
      generator = hierarchical_model_generator.HierarchicalModelGenerator
      _, spk_times, _, _ = generator.generate_spike_train(
          lmbd, dt=0.002, random_seed=None)
      if censor_interval is not None:
        spk_times = spk_times[(spk_times>=censor_interval[0]) &
                              (spk_times<=censor_interval[1])]
      u_val = cls.ks_test_interval_transform(spk_times, lmbd, bins=bins,
          dt=0.002, verbose=False)
      u_list.extend(u_val)

    bin_width=0.01
    plot_bins = np.linspace(0, 1, int(1 / bin_width) + 1)
    pdf, bin_edges = np.histogram(u_list, bins=plot_bins, density=True)
    cdf = np.cumsum(pdf) * bin_width

    if verbose:
      plt.figure(figsize=[3, 3])
      plt.plot(plot_bins[1:], cdf)

    return plot_bins[1:], cdf


  @classmethod
  def ks_test(
      cls,
      spike_times_x,
      spike_times_y,
      trial_window,
      model_par,
      dt=0.001,
      test_size=0.05,
      verbose=True):
    """Verify optimization values."""
    model_par['dt'] = dt
    num_trials = len(spike_times_y)
    num_spikes_x = [len(spikes) for spikes in spike_times_x]
    num_spikes_x = np.sum(num_spikes_x)
    num_spikes_y = [len(spikes) for spikes in spike_times_y]
    num_spikes_y = np.sum(num_spikes_y)
    T = trial_window[1]
    trial_length = trial_window[1] - trial_window[0]
    num_bins_trial = int(trial_length/dt)
    beta = model_par['beta']

    spike_hist_stacked_x, bins = cls.bin_spike_times(spike_times_x, dt,
        trial_window=trial_window)
    (num_basis, num_samples, X, basis_integral, offset, offset_integral
        ) = cls.bivariate_discrete_time_coupling_filter_build_regressors(
        spike_hist_stacked_x, trial_window, model_par, mean_norm=True)
    lambda_hat = X @ beta

    u_list = []
    start_ind = 0
    end_ind = num_bins_trial
    for r in range(num_trials):
      lambda_hat_r = lambda_hat[start_ind:end_ind]
      # if np.max(lambda_hat_r*dt) > 20:
      #   plt.figure(figsize=[4, 1.5])
      #   plt.plot(lambda_hat_r)
      #   plt.show()
      u_vals = cls.ks_test_interval_transform(spike_times_y[r],
          lambda_hat_r*dt, dt=dt, bins=bins[:-1], verbose=False)
      if len(u_vals) == 0:
        continue
      u_list.append(u_vals)
      start_ind += num_bins_trial
      end_ind += num_bins_trial

    u_list = np.vstack(u_list).reshape(-1)

    if verbose:
      CI_trap, mcdf, ecdf, CI_up, CI_dn = cls.check_ks(u_list,
          test_size, bin_width=0.02, verbose=True)

    return u_list


  @classmethod
  def multivariate_continuous_time_coupling_filter_build_regressors(
      cls,
      spike_times_x_list,
      spike_times_y,
      trial_window,
      model_par_list=None):
    """Build multivariate regressors."""
    num_x = len(spike_times_x_list)
    num_trials = len(spike_times_y)
    num_spikes_y = [len(spikes) for spikes in spike_times_y]
    num_samples = np.sum(num_spikes_y)

    num_basis_per_node = [0] * num_x
    X = [0] * num_x
    basis_integral = [0] * num_x
    offset = 0
    offset_integral = 0

    # Build regressor block for each node, then concatenate them.
    for n, spike_times_x in enumerate(spike_times_x_list):
      (num_basis_per_node[n], num_samples, X[n], basis_integral[n],
          offset_, offset_integral_
          ) = cls.bivariate_continuous_time_coupling_filter_build_regressors(
          spike_times_x, spike_times_y, trial_window, model_par_list[n])
      model_par_list[n]['num_basis'] = num_basis_per_node[n]
      offset += offset_
      offset_integral += offset_integral_

    X = np.hstack(X)
    basis_integral = np.vstack(basis_integral)
    num_basis = sum(num_basis_per_node)

    return num_basis, num_samples, X, basis_integral, offset, offset_integral


  @classmethod
  def multivariate_continuous_time_coupling_filter_regression(
      cls,
      spike_times_x_list,
      spike_times_y,
      trial_window,
      model_par_list=None,
      mute_warning=False,
      verbose=False):
    """Bivariate continuous-time PP-GLM.

    Args:
      model_par_list: corresponds to `spike_times_x_list`. model_par_list[0] is
          the main model with training parameters. Others may be used for
          nuisance regressor design.
    """
    num_x = len(spike_times_x_list)
    num_trials = len(spike_times_y)
    model_par = model_par_list[0]
    learning_rate = model_par['learning_rate']
    max_num_itrs = model_par['max_num_itrs']
    epsilon = model_par['epsilon']

    (num_basis, num_samples, X, basis_integral, offset, offset_integral
        ) = cls.multivariate_continuous_time_coupling_filter_build_regressors(
        spike_times_x_list, spike_times_y, trial_window, model_par_list)

    # Initialize parameters.
    if 'beta_init' in model_par:
      beta = model_par['beta_init'].copy()
    elif model_par['append_nuisance'] == 'const':
      beta = np.zeros([num_basis, 1]) + 1
      # Mean FR.
      beta[0] = num_samples / num_trials / (trial_window[1]-trial_window[0])
    elif 'const' in model_par['append_nuisance']:
      beta = np.zeros([num_basis, 1]) + 0.1
      beta[0] = num_samples / num_trials / (trial_window[1]-trial_window[0])
    else:
      beta = np.zeros([num_basis, 1]) + 0.1

    beta_old = beta
    if verbose:
      print(f'X.shape {X.shape}, basis_integral.shape {basis_integral.shape},' +
            f'beta.shape {beta.shape} np.shape(offset){np.shape(offset)}')

    if hasattr(tqdm, '_instances'):
      tqdm._instances.clear()
    if verbose:
      trange = tqdm(range(max_num_itrs), ncols=100, file=sys.stdout)
    else:
      trange =range(max_num_itrs)
    for itr in trange:
      lmbd = X @ beta + offset
      non_zero_ind = np.where(lmbd > 0)[0]
      lmbd_integral = basis_integral.T @ beta + offset_integral
      nll = cls.spike_times_neg_log_likelihood(lmbd[non_zero_ind], lmbd_integral)
      gradient = - X[non_zero_ind].T @ (1 / lmbd[non_zero_ind]) + basis_integral
      hessian = X[non_zero_ind].T @ (X[non_zero_ind] / np.square(lmbd[non_zero_ind]))

      try:
        delta = np.linalg.inv(hessian) @ gradient
      except np.linalg.LinAlgError:
        hessian = hessian + np.eye(hessian.shape[0])*0.01
        delta = np.linalg.inv(hessian) @ gradient
      # The threshold is set as 5 because we know the range of the optimal beta
      # is around 10.
      if any(np.abs(learning_rate*delta) > 5):
        lr = learning_rate * 0.2
        learning_rate = max(lr, 0.001)
      beta = beta - learning_rate * delta

      # Check convergence.
      beta_err = np.sum(np.abs(beta_old - beta))
      if beta_err < epsilon:
        break
      beta_old = beta

    if not mute_warning and itr == max_num_itrs-1 and max_num_itrs > 1:
      warnings.warn(f'Reach max itrs {max_num_itrs}. Last err:{beta_err:.3e}')
      model_par['warnings'] = 'itr_max'

    # Pack output.
    num_basis_per_node = [0] * (num_x)
    for n in range(num_x):
      num_basis_per_node[n] = model_par_list[n]['num_basis']
    node_basis_range = np.cumsum([0] + num_basis_per_node)
    beta_cov = np.linalg.inv(hessian)
    for n in range(num_x):
      model_par_tmp = model_par_list[n]
      l_id, r_id = node_basis_range[n], node_basis_range[n+1]
      model_par_tmp['beta'] = beta[l_id:r_id]
      model_par_tmp['beta_cov'] = beta_cov[l_id:r_id,l_id:r_id]
      model_par_tmp['num_itrs'] = itr
      model_par_tmp['nll'] = nll[0,0]

    if verbose:
      print('num itr', itr, nll, beta_err)
      print('beta', beta.reshape(-1))

    return model_par_list


  @classmethod
  def plot_continuous_time_multivariate_regression_model_par_one_target(
      cls,
      model_par_list,
      filter_par=None,
      src_nodes=None,
      tgt_node=None,
      ylim=None,
      file_path=None):
    """Visualize estimated filters.

    Args:
      model_par_list: estimated model list.
      filter_par: true model.
    """
    num_x = len(model_par_list)-1  # first one is constant basis.
    CI_scale = scipy.stats.norm.ppf(0.975)  # 95% CI.
    num_cols = num_x
    num_rows = np.ceil(num_x/num_cols).astype(int)
    gs_kw = dict(width_ratios=[1]*num_cols, height_ratios=[1]*num_rows)
    fig, axs = plt.subplots(figsize=(18, 1.8*num_rows), gridspec_kw=gs_kw,
        nrows=num_rows, ncols=num_cols)
    plt.subplots_adjust(left=None, right=None, hspace=0.15, wspace=0.15)
    axs = axs.reshape(-1)
    for n in range(num_x):
      model_par_tmp = model_par_list[n+1]
      if ('const' in model_par_tmp['append_nuisance'] and
          len(model_par_tmp['append_nuisance']) == 1):
        continue
      filter_length = model_par_tmp['filter_length']
      t, h, h_std = cls.reconstruct_basis(model_par_tmp)

      ax = fig.add_subplot(axs[n])
      # ax.tick_params(labelleft=False, labelbottom=False)
      plt.axhline(y=0, c='lightgrey')
      plt.axvline(x=0, c='lightgrey')
      plt.plot(t, h, 'k', label='Estimator')
      plt.fill_between(t, h+h_std*CI_scale, h-h_std*CI_scale,
                       facecolor='lightgrey', alpha=0.3, label='95% CI')
      # Add true filter if `filter_par` is provided..
      if filter_par is not None and filter_par['type'] == 'triangle':
        filter_alpha = filter_par['alpha'][0][1]
        filter_beta = filter_par['beta'][0][1]
        ax = axs[0]
        ax.plot([0, filter_beta], [filter_alpha, 0], 'g', label='True filter')
        plt.ylim(ylim)
      if filter_par is not None and filter_par['type'] == 'square':
        filter_alpha = filter_par['alpha'][tgt_node][src_nodes[n]]
        filter_beta = filter_par['beta'][tgt_node][src_nodes[n]]
        plt.plot([0, filter_beta], [filter_alpha, filter_alpha], 'g',
            label='True filter')
        plt.ylim(ylim)
        plt.xlim(0)
      if n == (num_rows-1) * num_cols:
        ax.tick_params(labelleft=True, labelbottom=True)
      if n == num_x-1:
        ax.tick_params(labelbottom=True)
      # plt.legend(loc=(1.05, 0), ncol=1)
      # xticks = np.arange(0, filter_length, 0.01)
      # plt.xticks(xticks, xticks*1000)
      # plt.xlabel('Time [ms]')
      # plt.ylabel('Firing rate [spk/sec]')
      plt.text(0.8,0.9, fr'{src_nodes[n]} $\to$ {tgt_node}', transform=ax.transAxes)
      plt.ylim(ylim)
      plt.xlim(0)

    if file_path is not None:
      plt.savefig(file_path)
      print('save figure:', file_path)
      plt.close()
    plt.show()


