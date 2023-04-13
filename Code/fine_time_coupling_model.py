import os
import sys

import collections
from collections import defaultdict
from glob import glob
import io
import itertools
import numpy as np
import matplotlib
import matplotlib.pylab
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import networkx as nx
import nxviz
import pandas as pd
import random
import seaborn
import scipy
from scipy.ndimage import gaussian_filter1d
import sklearn
import sklearn.cluster
import statsmodels
from statsmodels.stats.multitest import multipletests
import time
from tqdm import tqdm
import warnings

import data_model
import smoothing_spline
import util
import jitter


XorrPair = collections.namedtuple(
    'XorrPair', ['unit_from', 'unit_to', 'raw', 'mean', 'ci_alpha',
    'CI_down', 'CI_up', 'p_val', 'p_val_rnd'])


class FineTimeCouplingModel(data_model.AllenInstituteDataModel):

  def __init__(self, session=None, empty_samples=False):
    super().__init__(session)
    # Output setup.
    if hasattr(self.session, 'ecephys_session_id'):
      self.session_id = self.session.ecephys_session_id
    else:
      self.session_id = 0


  def initial_step(
      self,
      spike_trains,
      spike_times,
      selected_units,
      trials_groups,
      trial_window,
      probes,
      num_areas,
      verbose=True):
    """Initializes the parameters."""
    self.spike_trains = spike_trains
    self.spike_times = spike_times
    self.selected_units = selected_units
    self.probes = probes
    self.trials_groups = trials_groups
    self.trial_window = trial_window
    # TODO: make this compatible with non-zero start.
    # self.trial_length = trial_window[1] - trial_window[0]

    self.num_areas = num_areas
    self.num_trials = trials_groups.size().max()  # Max trials per condition.
    self.total_num_trials = trials_groups.size().sum()
    self.num_conditions = len(trials_groups)

    self.map_c_to_cid = {}
    self.spike_counts = spike_times.applymap(lambda x: len(x))
    self.spike_counts_c = [pd.DataFrame()] * self.num_conditions

    for c, (stimulus_condition_id, trials_table) in enumerate(trials_groups):
      trials_indices = trials_groups.get_group(stimulus_condition_id).index.values
      self.map_c_to_cid[c] = stimulus_condition_id
      self.spike_counts_c[c] = self.spike_counts.loc[:,trials_indices]
    condition_list = [(i,item[0]) for i,item in enumerate(trials_groups)]
    print('conditions:', condition_list)

    # Fit using smoothing splines.
    self.fit_model = smoothing_spline.SmoothingSpline()
    self.jittertool = jitter.JitterTool()

    # Cache for model fitting.
    self.membership_fit_cache = {}


  @classmethod
  def print_conditions(
      cls,
      trials_groups):
    """Displays information of trials groups."""
    is_drifing_gratings = True
    trial_counter = 0

    for c, (condition_id, trials_df) in enumerate(trials_groups):
      if ('orientation' not in trials_df or
          'temporal_frequency' not in trials_df or
          'contrast' not in trials_df):
        is_drifing_gratings = False
        break
      orientation = trials_df['orientation'].unique()
      temporal_frequency = trials_df['temporal_frequency'].unique()
      contrast = trials_df['contrast'].unique()
      print(f'{c}  {condition_id} ' + 
            f'temp freq {temporal_frequency} ' +
            f'orient {orientation} ' +
            f'contrast {contrast} ' +
            f'{trials_df.index.values}')
      trial_counter += len(trials_df)

    if not is_drifing_gratings:
      for key in trials_groups.groups:
        trial_ids = trials_groups.groups[key].values
        print('stimulus_id', key, ' trial_id', trial_ids)
        trial_counter += len(trial_ids)

    print('total num trials:', trial_counter)


  def get_active_units(
      self,
      active_firing_rate_quantile_threshold=0.7,
      group_type='probe',
      verbose=False):
    """Get active units using mean firing rate.

    Args:
      group_type:
          'all': threshold all neurons together.
          'probe': threshold neurons within each probe.
    """
    trial_length = self.trial_window[1] - self.trial_window[0]
    unit_mean_firing_rate = self.spike_counts.mean(axis=1) / trial_length

    if group_type == 'all':
      mean_firing_rate_threshold = np.quantile(
          unit_mean_firing_rate, active_firing_rate_quantile_threshold)
      active_units_ids = unit_mean_firing_rate[unit_mean_firing_rate >
                                               mean_firing_rate_threshold]
      active_units_ids = active_units_ids.index.values
      active_units = self.selected_units.loc[active_units_ids]

      if verbose:
        plt.figure(figsize=[8,2])
        seaborn.distplot(unit_mean_firing_rate,
                         bins=100, color='tab:gray', kde=False)
        plt.axvline(x=mean_firing_rate_threshold, lw=0.3, ls='--', c='red')
        plt.title(f'num active units: {len(active_units)}')
        plt.show()

    elif group_type == 'probe':
      active_units_ids = []
      for a, probe in enumerate(self.probes):
        area_units = self.selected_units[
            self.selected_units['probe_description'] == probe].index.values
        area_unit_mean_firing_rate = unit_mean_firing_rate.loc[area_units]
        mean_firing_rate_threshold = np.quantile(
            area_unit_mean_firing_rate, active_firing_rate_quantile_threshold)
        area_active_units_ids = area_unit_mean_firing_rate[
            area_unit_mean_firing_rate > mean_firing_rate_threshold]
        active_units_ids.extend(area_active_units_ids.index.values.tolist())
      active_units = self.selected_units.loc[active_units_ids]

      if verbose:
        plt.figure(figsize=[15,2])
        for a, probe in enumerate(self.probes):
          area_units = self.selected_units[
              self.selected_units['probe_description'] == probe].index.values
          area_unit_mean_firing_rate = unit_mean_firing_rate.loc[area_units]
          mean_firing_rate_threshold = np.quantile(
              area_unit_mean_firing_rate, active_firing_rate_quantile_threshold)
          area_active_units_ids = area_unit_mean_firing_rate[
              area_unit_mean_firing_rate > mean_firing_rate_threshold]

          plt.subplot(1, self.num_areas, a+1)
          seaborn.distplot(area_unit_mean_firing_rate,
                           bins=100, color='tab:gray', kde=False)
          plt.axvline(x=mean_firing_rate_threshold, lw=0.3, ls='--', c='red')
          plt.title(f'num active units: {len(area_active_units_ids)}')
        plt.show()

    return active_units


  def check_empty_trials(
      self,
      spike_times,
      active_units,
      verbose=False):
    """Check empty trials of each neuron."""
    trial_ids = spike_times.columns.values

    for a, probe in enumerate(self.probes):
      area_units = active_units[active_units['probe_description'] == probe].index.values
      # print(area_units)
      for n, unit in enumerate(area_units):
        spike_times_y = spike_times.loc[unit,trial_ids].tolist()
        num_spikes_y = np.array([len(spikes) for spikes in spike_times_y])
        # num_spikes_y = np.sum(num_spikes_y)
        num_empty_trials = np.sum(num_spikes_y == 0)
        print(unit, num_empty_trials)
      print()


  def construct_unit_pairs(
      self,
      active_units,
      pair_type='between_probe',
      probes=None,
      probe_pairs=None,
      verbose=False):
    """Construct pairs of units.

    Args:
      pair_type:
          all: pairs between all units.
          between_probe: pairs between probes.
          within_probe: pairs within probes.
    """
    unit_pairs = []
    area_pairs = []

    if pair_type == 'all':
      unit_ids = active_units.index.values
      for unit_x, unit_y in itertools.combinations(unit_ids, 2):
        unit_pairs.append((unit_x, unit_y))

    elif pair_type == 'between_probe':
      if probe_pairs is None:
        probe_pairs = itertools.combinations(self.probes, 2)
      for probe_x, probe_y in probe_pairs:
        area_pairs.append((probe_x, probe_y))
        units_x = active_units[active_units['probe_description']==probe_x].index.values
        units_y = active_units[active_units['probe_description']==probe_y].index.values
        for unit_x in units_x:
          for unit_y in units_y:
            unit_pairs.append((unit_x, unit_y))

    elif pair_type == 'within_probe':
      if probes is None:
        probes = self.probes
      for probe in probes:
        units = active_units[active_units['probe_description'] == probe].index.values
        for unit_x, unit_y in itertools.combinations(units, 2):
          unit_pairs.append((unit_x, unit_y))

    if verbose:
      print('area paris', area_pairs)
      print(f'num_pairs: {len(unit_pairs)}')
    return unit_pairs


  def explore_all_pairs_cross_correlation_jitter(
      self,
      unit_pairs,
      spk_bin_width=0.002,
      lag_range=[-0.08, 0.08],
      jitter_window_width=0.04,
      distribution_type='poisson',
      num_jitter=300,
      ci_alpha=0.01,
      verbose=False):
    """Explore all pairs of cross-correlation."""
    trial_length = self.trial_length
    num_pairs = len(unit_pairs)
    xorr_pairs = {}

    for p, unit_pair in enumerate(tqdm(unit_pairs)):
      unit_x, unit_y = unit_pair
      spike_times_x = self.spike_times.loc[unit_x].tolist()
      spike_times_y = self.spike_times.loc[unit_y].tolist()

      (lags, xorr_raw, xorr_mean, xorr_CI_down, xorr_CI_up, p_val, p_val_rnd
          ) = self.jittertool.cross_correlation_jitter(
          spike_times_x, spike_times_y,
          spk_bin_width=spk_bin_width, trial_length=trial_length,
          lag_range=lag_range, jitter_window_width=jitter_window_width,
          distribution_type=distribution_type, num_jitter=num_jitter,
          ci_alpha=ci_alpha, verbose=False)

      xorr_pairs[(unit_x, unit_y)] = XorrPair(unit_from=unit_x, unit_to=unit_y,
          raw=xorr_raw, mean=xorr_mean, ci_alpha=ci_alpha, CI_down=xorr_CI_down,
          CI_up=xorr_CI_up, p_val=p_val, p_val_rnd=p_val_rnd)
      if 'lags' not in xorr_pairs:
        xorr_pairs['lags'] = lags
      if 'jitter_window_width' not in xorr_pairs:
        xorr_pairs['jitter_window_width'] = jitter_window_width
      if 'spk_bin_width' not in xorr_pairs:
        xorr_pairs['spk_bin_width'] = spk_bin_width

      if verbose:
        probe_x = self.selected_units.loc[unit_x, 'probe_description']
        probe_y = self.selected_units.loc[unit_y, 'probe_description']
        plot_label = (f'{probe_x} {unit_x} ==> {probe_y} {unit_y}    ' +
            f'jit-win={jitter_window_width * 1000} ms,  ' +
            f'spk-bin={spk_bin_width  * 1000} ms')
        self.plot_cross_correlation_jitter(lags, xorr_raw,
            xorr_mean, xorr_CI_up, xorr_CI_down,
            plot_label=plot_label, ci_alpha=ci_alpha)

    return xorr_pairs


  # Deprecated.
  def explore_all_pairs_cross_correlation_jitter_mc(
      self,
      spk_bin_width=0.002,
      lag_range=[-0.08, 0.08],
      jitter_window_width=0.04,
      num_jitter=300,
      ci_alpha=0.01,
      verbose=False):
    """Explore all pairs of cross-correlation.

    I break down the `jittertool.cross_correlation_jitter` because for each
    unit, I only need to jitter once, no need to jitter for every pair.
    This has been replaced by approximation method such as binomial or poisson
    jitter solution.
    """
    active_units = self.get_active_units(0.98, verbose=True)
    trial_length = self.trial_length

    for probe_x, probe_y in itertools.combinations(self.probes, 2):
      units_x = active_units[active_units['probe_description'] == probe_x].index.values
      units_y = active_units[active_units['probe_description'] == probe_y].index.values
      for unit_x in units_x:
        # Generate jitter data.
        spike_times_x = self.spike_times.loc[unit_x].tolist()
        spike_hist_x, bins = self.jittertool.bin_spike_times(
            spike_times_x, spk_bin_width, trial_length)
        num_bins = len(bins)
        num_trains = len(spike_times_x)
        spike_hist_surrogate_x = np.zeros([num_jitter, num_trains, num_bins])

        for r in range(num_jitter):
          spike_times_surrogate_x = self.jittertool.jitter_spike_times_interval(
              spike_times_x, jitter_window_width, num_jitter=1, verbose=False)
          spike_hist_surrogate_x[r], _ = self.jittertool.bin_spike_times(
              spike_times_surrogate_x, spk_bin_width, trial_length)

        for unit_y in tqdm(units_y):
        # for unit_y in units_y:
          spike_times_y = self.spike_times.loc[unit_y].tolist()
          spike_hist_y, _    = self.jittertool.bin_spike_times(
              spike_times_y, spk_bin_width, trial_length)
          xorr_raw, lags = self.jittertool.cross_correlation(
              bins, spike_hist_x, spike_hist_y, lag_range, verbose=False)

          # Jitter surrogate.
          num_trials = len(spike_times_x)
          num_lags = len(lags)
          xorr_jitter = np.zeros([num_jitter, num_lags])

          for r in range(num_jitter):
            xorr_jitter[r], lags = self.jittertool.cross_correlation(
                bins, spike_hist_surrogate_x[r], spike_hist_y, lag_range,
                verbose=False)

          xorr_CI_up = np.quantile(xorr_jitter, 1-ci_alpha/2, axis=0)
          xorr_CI_down = np.quantile(xorr_jitter, ci_alpha/2, axis=0)
          xorr_mean = np.mean(xorr_jitter, axis=0)

          if verbose:
            plot_label = (f'{probe_x} {unit_x} \u27F6 {probe_y} {unit_y}    ' +
                f'jit-win={jitter_window_width * 1000} ms,  ' +
                f'spk-bin={spk_bin_width  * 1000} ms')
            self.plot_cross_correlation_jitter(lags, xorr_raw,
                xorr_mean, xorr_CI_down, xorr_CI_up,
                plot_label=plot_label, ci_alpha=ci_alpha)


  def group_xorr_pairs(
      self,
      xorr_pairs,
      verbose=False):
    """Group the xorr pairs.

    Args:
      xorr_pairs: Outcomes from `explore_all_pairs_cross_correlation_jitter`
          or `explore_all_pairs_cross_correlation_jitter`.
    """
    lags = xorr_pairs['lags']
    jitter_window_width = xorr_pairs['jitter_window_width']
    spk_bin_width = xorr_pairs['spk_bin_width']

    # Set up the groups in the same format as `xorr_pairs`.
    num_probes = len(self.probes)
    num_groups = int(num_probes * (num_probes-1) / 2)
    grouped_pairs = collections.defaultdict(dict)
    for probe_x, probe_y in itertools.combinations(self.probes, 2):
      grouped_pairs[(probe_x, probe_y)]['lags'] = lags
      grouped_pairs[(probe_x, probe_y)]['jitter_window_width'] = jitter_window_width
      grouped_pairs[(probe_x, probe_y)]['spk_bin_width'] = spk_bin_width

    for key in xorr_pairs:
      if not isinstance(key, tuple):
        continue
      unit_x, unit_y = key
      probe_x = self.selected_units.loc[unit_x, 'probe_description']
      probe_y = self.selected_units.loc[unit_y, 'probe_description']
      grouped_pairs[(probe_x, probe_y)][key] = xorr_pairs[key]

    if verbose:
      for probe_x, probe_y in itertools.combinations(self.probes, 2):
        print(probe_x, probe_y, len(grouped_pairs[(probe_x, probe_y)]))

    return grouped_pairs


  @classmethod
  def plot_cross_correlation_jitter(
      cls,
      lags,
      xorr_raw,
      xorr_mean,
      xorr_CI_down,
      xorr_CI_up,
      plot_label,
      ci_alpha,
      ylim=None):
    """Plot the cross-correlation with jitter bands."""
    gs_kw = dict(width_ratios=[1, 1], height_ratios=[1])
    fig, axs = plt.subplots(figsize=(12, 3), gridspec_kw=gs_kw,
        nrows=1, ncols=2)
    # plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)

    ax = fig.add_subplot(axs[0])
    plt.fill_between(lags * 1000, xorr_CI_down, xorr_CI_up,
                     facecolor='lightgrey', alpha=0.8,
                     label=f'{(1-ci_alpha) * 100}% CI')
    plt.plot(lags * 1000, xorr_mean, c='grey', ls='--')
    plt.plot(lags * 1000, xorr_raw, c='k')
    plt.axvline(x=0, lw=0.4, c='darkgrey')
    plt.xlabel('lag [ms]')
    plt.ylabel('xorr-correlogram')
    plt.text(0.1, 1.07, plot_label, transform=ax.transAxes)
    plt.legend()

    ax = fig.add_subplot(axs[1])
    plt.fill_between(lags * 1000, xorr_CI_down - xorr_mean, 
                     xorr_CI_up - xorr_mean,
                     facecolor='lightgrey', alpha=0.8)
    plt.plot(lags * 1000, xorr_raw - xorr_mean, c='k')
    plt.axvline(x=0, lw=0.4, c='darkgrey')
    plt.axhline(y=0, lw=0.4, c='darkgrey')
    plt.xlabel('lag [ms]')
    plt.text(0.7, 0.9, 'mean corrected', transform=ax.transAxes)
    plt.ylim(ylim)
    plt.show()


  def detect_between_probe_unit_pairs(
      self,
      active_units,
      unit_pairs,
      xorr_pairs,
      detect_threshold=0.02,
      verbose=False):
    """Plot pairs of units."""
    probes = active_units['probe_description'].unique()
    pair_cnt = {}
    significant_pair_cnt = {}
    for probe_x, probe_y in itertools.combinations(probes, 2):
      pair_cnt[(probe_x, probe_y)] = 0
      significant_pair_cnt[(probe_x, probe_y)] = 0

    for probe_x, probe_y in itertools.combinations(probes, 2):
      units_x = active_units[active_units['probe_description'] == probe_x].index.values
      units_y = active_units[active_units['probe_description'] == probe_y].index.values
      significant_pair_cnt[(probe_x, probe_y)] = 0

      for unit_x in units_x:
        for unit_y in units_y:
          if (unit_x, unit_y) not in unit_pairs:
            continue
          pair_cnt[(probe_x, probe_y)] += 1

          # Detect significance.
          xorr_pair = xorr_pairs[(unit_x, unit_y)]
          lags = xorr_pairs['lags']
          xorr_raw = xorr_pair.raw
          xorr_mean = xorr_pair.mean
          xorr_CI_down = xorr_pair.CI_down
          xorr_CI_up = xorr_pair.CI_up
          (up_significant, down_significant
              ) = self.jittertool.verify_xorr_significance(
              lags, xorr_raw, xorr_mean, xorr_CI_down, xorr_CI_up,
              detect_threshold=detect_threshold, verbose=False)
          if not up_significant and not down_significant:
            continue
          significant_pair_cnt[(probe_x, probe_y)] += 1

          # Plot the sieved pair.
          if verbose:
            jit_win = xorr_pairs['jitter_window_width']
            spk_bin = xorr_pairs['spk_bin_width']
            plot_label = (f'{probe_x} {unit_x} ==> {probe_y} {unit_y}    ' +
                f'jit-win={jit_win*1000} ms,  spk-bin={spk_bin*1000} ms')
            self.plot_cross_correlation_jitter(lags, xorr_raw,
                xorr_pair.mean, xorr_pair.CI_down, xorr_pair.CI_up,
                plot_label=plot_label, ci_alpha=xorr_pair.ci_alpha)

    print(f'Total units: {len(active_units)}')
    print(f'Total pairs: {len(unit_pairs)}')

    total_significant_cnt = 0
    total_cnt = 0
    for probe_x, probe_y in itertools.combinations(probes, 2):
      total_significant_cnt += significant_pair_cnt[(probe_x, probe_y)]
      total_cnt += pair_cnt[(probe_x, probe_y)]
      ratio = significant_pair_cnt[(probe_x, probe_y)] / pair_cnt[(probe_x, probe_y)]
      print(f'{probe_x} -- {probe_y}')
      print(f'num significant pairs: {significant_pair_cnt[(probe_x, probe_y)]}')
      print(f'significant pairs portion: {np.around(ratio*100, 1)}%')

    print(f'Total cnt: {total_significant_cnt}')
    print('Overall significant ratio: '+
        f'{np.around(total_significant_cnt/total_cnt*100, 2)}%')


  def detect_within_probe_unit_pairs(
      self,
      active_units,
      unit_pairs,
      xorr_pairs,
      detect_threshold=0.02,
      verbose=False):
    """Plot pairs of units."""
    probes = active_units['probe_description'].unique()
    pair_cnt = {}
    significant_pair_cnt = {}
    for probe in probes:
      pair_cnt[probe] = 0
      significant_pair_cnt[probe] = 0

    for probe in probes:
      units = active_units[active_units['probe_description'] == probe].index.values
      significant_pair_cnt[probe] = 0

      for unit_x, unit_y in itertools.combinations(units, 2):
        if (unit_x, unit_y) not in unit_pairs:
          continue
        pair_cnt[probe] += 1

        # Detect significance.
        xorr_pair = xorr_pairs[(unit_x, unit_y)]
        lags = xorr_pairs['lags']
        xorr_raw = xorr_pair.raw
        xorr_mean = xorr_pair.mean
        xorr_CI_down = xorr_pair.CI_down
        xorr_CI_up = xorr_pair.CI_up
        (up_significant, down_significant
            ) = self.jittertool.verify_xorr_significance(
            lags, xorr_raw, xorr_mean, xorr_CI_down, xorr_CI_up,
            detect_threshold=detect_threshold, verbose=False)
        if not up_significant and not down_significant:
          continue
        significant_pair_cnt[probe] += 1

        # Plot the sieved pair.
        if verbose:
          jit_win = xorr_pairs['jitter_window_width']
          spk_bin = xorr_pairs['spk_bin_width']
          plot_label = (f'{probe} {unit_x} ==> {unit_y}    ' +
              f'jit-win={jit_win*1000} ms,  spk-bin={spk_bin*1000} ms')
          self.plot_cross_correlation_jitter(lags, xorr_raw,
              xorr_pair.mean, xorr_pair.CI_down, xorr_pair.CI_up,
              plot_label=plot_label, ci_alpha=xorr_pair.ci_alpha)

    print(f'Total units: {len(active_units)}')
    print(f'Total pairs: {len(unit_pairs)}')

    total_significant_cnt = 0
    total_cnt = 0
    for probe in probes:
      total_significant_cnt += significant_pair_cnt[probe]
      total_cnt += pair_cnt[probe]
      ratio = significant_pair_cnt[probe] / pair_cnt[probe]
      print(f'{probe}')
      print(f'num significant pairs: {significant_pair_cnt[probe]}')
      print(f'significant pairs portion: {np.around(ratio*100, 1)}%')

    print(f'Total cnt: {total_significant_cnt}')
    print('Overall significant ratio: '+
        f'{np.around(total_significant_cnt/total_cnt*100, 2)}%')

  def select_plot_edges(
      self,
      df_inference,
      probe_pairs=None,
      select_ratio=1.0):
    """df_inference has columns below.
    source target pval  h source_probe  target_probe  source_area target_area
    951092303 951103361 1.598066e-01  0.628878  probeA  probeC  VISam VISp
    """
    df_inference = df_inference.copy()

    if probe_pairs is not None:
      df_inference['source_target_tuple'] = list(zip(
          df_inference.source_probe, df_inference.target_probe))
      df_inference = df_inference[df_inference.source_target_tuple.isin(probe_pairs)]
      del df_inference['source_target_tuple']

    if select_ratio < 1.0:
      hide_edges = df_inference[df_inference.significant].sample(
          frac=(1-select_ratio), random_state=3)
      df_inference.loc[hide_edges.index] = False

    print('edge stats:\n', df_inference.significant.value_counts())
    print(df_inference[df_inference.significant].h.value_counts(bins=[-1e10,0,1e10]))

    return df_inference


  def build_graph_probe_connection(
      self,
      df_probe,
      layout_probe,
      graph_template=None,
      figure_path=None,
      verbose=False):
    """Plot pairs of units.

    Args:
      layout_probe: should be in order.
    """
    # graph = nx.Graph(name='probe connections')
    graph = nx.DiGraph(name='probe connections')

    # Add nodes layout.
    for n, p in enumerate(layout_probe):
      graph.add_node(p, probe=p)

    # Add edges.
    for key in df_probe.index:
      (unit_x, unit_y) = key
      edge_weight = df_probe.loc[key].significant
      graph.add_weighted_edges_from([(unit_x, unit_y, edge_weight)])

    # Add information after building the graph.
    for v in graph:
      # graph.nodes[v]['degree'] = graph.degree(v)
      graph.nodes[v]['degree'] = 1.0

    # Plot chord graph.
    # self.plot_chord_diagram(graph, graph_template=graph_template, figure_path=figure_path)


    probe_color = {'probeA': 'tab:blue', 'probeB': 'tab:orange',
        'probeC': 'tab:green', 'probeD': 'tab:red', 'probeE': 'tab:purple',
        'probeF': 'tab:gray'}

    nodes = list(graph.nodes())  # List of node keys (val not included).
    nodeprops = {'radius': 1}
    node_r = nodeprops['radius']
    radius = nxviz.geometry.circos_radius(len(nodes), node_r)
    nodeprops["linewidth"] = radius * 0.01
    plot_radius = radius
    edge_colors = ['black'] * len(graph.edges())
    edge_widths = [1] * len(graph.edges())
    edgeprops = {"facecolor": "none", "alpha": 0.2}

    # Nodes order.
    probe_rank = {p:i for i, p in enumerate(layout_probe)}

    if graph_template is None:
      nodes_sorted = sorted(graph.nodes(data=True),
                            key=lambda x: x[1]['degree'], reverse=True)
      # nodes_sorted = sorted(nodes_sorted, key=lambda x: x[1]['probe'])
    else:
      nodes_sorted = sorted(graph_template.nodes(data=True),
                            key=lambda x: x[1]['degree'], reverse=True)
      # nodes_sorted = sorted(nodes_sorted, key=lambda x: x[1]['probe'])

    nodes_sorted = sorted(nodes_sorted, key=lambda x: probe_rank[x[1]['probe']])
    nodes = [key for key, val in nodes_sorted]

    # Nodes layout.
    def node_theta(nodelist, node):
      # The theta value is in [-pi, pi].
      i = nodelist.index(node)
      delta = 2 * np.pi / len(nodelist)
      theta = i * delta
      if theta >= np.pi:
        theta = theta - 2*np.pi
      return theta

    xs, ys = [], []
    for node in nodes:
      x, y = nxviz.geometry.get_cartesian(
          r=radius, theta=node_theta(nodes, node))
      xs.append(x)
      ys.append(y)
    node_coords = {"x": xs, "y": ys}

    plt.figure(figsize=(5, 5))
    ax = plt.gca()
    # Nodes.
    lw = nodeprops["linewidth"]
    for i, node in enumerate(nodes):
      x = node_coords["x"][i]
      y = node_coords["y"][i]
      # color = node_colors[i]
      node_probe = graph.nodes[node]['probe']
      node_color = probe_color[node_probe]

      # node_patch = matplotlib.patches.Circle((x, y), node_r, lw=lw, color=node_color, zorder=2)
      node_patch = matplotlib.patches.Circle((x, y), 0.2, lw=0.2, color=node_color, zorder=2)
      ax.add_patch(node_patch)

    # Edges.
    for i, (start, end) in enumerate(graph.edges()):
      start_theta = node_theta(nodes, start)
      end_theta = node_theta(nodes, end)

      verts = [
        nxviz.geometry.get_cartesian(plot_radius, start_theta),
        (0, 0),
        nxviz.geometry.get_cartesian(plot_radius, end_theta),
      ]

      color = edge_colors[i]
      # color = 'red' if graph.get_edge_data(start, end)['weight'] > 0 else 'blue'
      edge_weight = graph.get_edge_data(start, end)['weight']

      codes = [matplotlib.path.Path.MOVETO, matplotlib.path.Path.CURVE3,
               matplotlib.path.Path.CURVE3]
      lw = edge_widths[i]
      path = matplotlib.path.Path(verts, codes)

      patch = matplotlib.patches.PathPatch(
          path, lw=edge_weight, edgecolor=color, zorder=1, **edgeprops)
      ax.add_patch(patch)

      # arrowstyle = f'fancy,head_length={50},head_width={10},tail_width={2}'
      # arrow = matplotlib.patches.FancyArrowPatch(path=path, lw=edge_weight,
      #     edgecolor=color, arrowstyle=arrowstyle, **edgeprops )
      # ax.add_artist(arrow)


    # Labels.
    map_probe_to_areas = self.session.map_probe_to_ecephys_structure_acronym(
        visual_only=True)
    print(map_probe_to_areas)
    map_probe_to_areas = {'probeA': 'AM', 'probeB': 'AM', 'probeC': 'V1',
        'probeD': 'LM', 'probeE': 'AL', 'probeF': 'RL'}
    probe_label_cnt = {'probeA': 0, 'probeB': 0, 'probeC': 0, 'probeD': 0,
        'probeE': 0, 'probeF': 0}
    for i, node in enumerate(nodes):
      node_probe = graph.nodes[node]['probe']
      probe_label_cnt[node_probe] += 1
      if probe_label_cnt[node_probe] == 1:
        theta=node_theta(nodes, node) + 0.2
        x, y = nxviz.geometry.get_cartesian(r=radius*1.08, theta=theta)
        deg = theta / np.pi * 180 - 90
        deg = deg - 180 if theta < 0 else deg
        ax.text(x, y, map_probe_to_areas[node_probe],
                rotation=deg, ha='center', va='center', fontsize=12)

    ax.relim()
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis('off')

    return graph


  def build_graph_from_regression_pval(
      self,
      df_inference,
      layout_units=None,
      layout_probe=None,
      graph_template=None,
      figure_path=None,
      verbose=False):
    """Plot pairs of units."""
    graph = nx.Graph(name='units connections')

    if layout_units is None:
      units = self.selected_units
    else:
      units = layout_units

    # Add nodes layout.
    for unit_id in units.index.values:
        unit_probe = units.loc[unit_id, 'probe_description']
        graph.add_node(unit_id, probe=unit_probe)

    print('num edges', df_inference.significant.sum())
    # Add edges.
    for key in df_inference.index:

        # Detect significance.
        if not df_inference.loc[key].significant:
            continue

        (unit_x, unit_y) = key
        edge_weight = 1 if df_inference.loc[key].h > 0 else -1
        graph.add_weighted_edges_from([(unit_x, unit_y, edge_weight)])

    # Add information after building the graph.
    for v in graph:
        graph.nodes[v]['degree'] = graph.degree(v)

    # Plot chord graph.
    self.plot_chord_diagram(graph, layout_probe=layout_probe,
        graph_template=graph_template, figure_path=figure_path)
    return graph


  def build_graph_from_xorr_pairs(
      self,
      layout_units,
      xorr_pairs,
      detect_threshold=0.02,
      graph_template=None,
      figure_path=None,
      verbose=False):
    """Plot pairs of units."""
    if layout_units is None:
      units = self.selected_units
    else:
      units = layout_units
    probes = units['probe_description'].unique()
    graph = nx.Graph(name='units connections')

    # Add nodes layout.
    for unit_id in units.index.values:
      unit_probe = units.loc[unit_id, 'probe_description']
      graph.add_node(unit_id, probe=unit_probe)

    # Add edges.
    for key in xorr_pairs:
      if not isinstance(key, tuple):
        continue
      (unit_x, unit_y) = key

      # Detect significance.
      xorr_pair = xorr_pairs[(unit_x, unit_y)]
      lags = xorr_pairs['lags']
      xorr_raw = xorr_pair.raw
      xorr_mean = xorr_pair.mean
      xorr_CI_down = xorr_pair.CI_down
      xorr_CI_up = xorr_pair.CI_up
      (up_significant, down_significant
          ) = self.jittertool.verify_xorr_significance(
          lags, xorr_raw, xorr_mean, xorr_CI_down, xorr_CI_up,
          detect_threshold=detect_threshold, verbose=False)
      if not up_significant and not down_significant:
        continue

      graph.add_weighted_edges_from([(unit_x, unit_y, 1)])

      # Plot the sieved pair.
      if verbose == 1:
        jit_win = xorr_pairs['jitter_window_width']
        spk_bin = xorr_pairs['spk_bin_width']
        probe_x = self.selected_units.loc[unit_x, 'probe_description']
        probe_y = self.selected_units.loc[unit_y, 'probe_description']
        plot_label = (f'{probe_x} {unit_x} ==> {probe_y} {unit_y}    ' +
            f'jit-win={jit_win*1000} ms,  spk-bin={spk_bin*1000} ms')
        self.plot_cross_correlation_jitter(lags, xorr_raw,
            xorr_pair.mean, xorr_pair.CI_down, xorr_pair.CI_up,
            plot_label=plot_label, ci_alpha=xorr_pair.ci_alpha)

    # Add information after building the graph.
    for v in graph:
      graph.nodes[v]['degree'] = graph.degree(v)

    # Plot chord graph.
    if verbose == 2:
      self.plot_chord_diagram(graph, graph_template, figure_path)
      # self.plot_nxviz_graph(graph)
      # self.plot_networkx_graph(graph)
    # Plot chord graph.
    elif verbose == 3:
      self.plot_adj_mat(graph, graph_template, figure_path)

    return graph


  def group_sort_nodes_by_attr(
      self,
      graph,
      group_attr,
      sort_attr):
    """Group then sort nodes."""
    # Group nodes.
    groups = collections .defaultdict(list)
    nodelist = graph.nodes(data=True)
    for node in nodelist:
      groups[node[1][group_attr]].append(node)

    # Sort nodes within each group.
    for key in groups:
      nodelist_group = groups[key]
      nodes_sorted = sorted(nodelist_group,
                            key=lambda x: x[1][sort_attr], reverse=True)
      nodes_sorted = [(key, val) for key, val in nodes_sorted]
      groups[key] = nodes_sorted

    return groups


  def plot_adj_mat(
      self,
      graph,
      graph_template=None,
      figure_path=None):
    """Plot graph as adjacency matrix."""
    # Nodes order.
    if graph_template is None:
      nodes_sorted = sorted(graph.nodes(data=True),
                            key=lambda x: x[1]['degree'], reverse=True)
      nodes_sorted = sorted(nodes_sorted,
                            key=lambda x: x[1]['probe'])
      nodelist = [key for key, val in nodes_sorted]
    else:
      nodes_sorted = sorted(graph_template.nodes(data=True),
                            key=lambda x: x[1]['degree'], reverse=True)
      nodes_sorted = sorted(nodes_sorted,
                            key=lambda x: x[1]['probe'])
      nodelist = [key for key, val in nodes_sorted]

    adj_mat = nx.to_numpy_matrix(graph, nodelist=nodelist)
    np.fill_diagonal(adj_mat, 1)

    gs_kw = dict(width_ratios=[1])
    fig, axs = plt.subplots(figsize=(6, 6),
        gridspec_kw=gs_kw, nrows=1, ncols=1)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0.05)
    ax = fig.add_subplot(axs)
    ax.axis('off')
    seaborn.heatmap(adj_mat, cbar=False, square=True)
    plt.show()


  def plot_chord_diagram(
      self,
      graph,
      layout_probe=None,
      graph_template=None,
      figure_path=None):
    """Plot circular graph.

    Args:
      graph: networkx graph.
    """
    probe_color = {'probeA': 'tab:blue', 'probeB': 'tab:orange',
        'probeC': 'tab:green', 'probeD': 'tab:red', 'probeE': 'tab:purple',
        'probeF': 'tab:gray'}

    nodes = list(graph.nodes())  # List of node keys (val not included).
    nodeprops = {'radius': 1}
    node_r = nodeprops['radius']
    radius = nxviz.geometry.circos_radius(len(nodes), node_r)
    nodeprops["linewidth"] = radius * 0.01
    plot_radius = radius
    edge_colors = ['black'] * len(graph.edges())
    edge_widths = [1] * len(graph.edges())
    edgeprops = {"facecolor": "none", "alpha": 0.2}

    # Nodes order.
    if graph_template is None:
      nodes_sorted = sorted(graph.nodes(data=True),
                            key=lambda x: x[1]['degree'], reverse=True)
    else:
      nodes_sorted = sorted(graph_template.nodes(data=True),
                            key=lambda x: x[1]['degree'], reverse=True)

    if layout_probe is None:
      nodes_sorted = sorted(nodes_sorted, key=lambda x: x[1]['probe'])
    else:
      probe_rank = {p:i for i, p in enumerate(layout_probe)}
      nodes_sorted = sorted(nodes_sorted, key=lambda x: probe_rank[x[1]['probe']])
    nodes = [key for key, val in nodes_sorted]

    # Nodes layout.
    def node_theta(nodelist, node):
      # The theta value is in [-pi, pi].
      i = nodelist.index(node)
      delta = 2 * np.pi / len(nodelist)
      theta = i * delta
      if theta >= np.pi:
        theta = theta - 2*np.pi
      return theta

    xs, ys = [], []
    for node in nodes:
      x, y = nxviz.geometry.get_cartesian(
          r=radius, theta=node_theta(nodes, node))
      xs.append(x)
      ys.append(y)
    node_coords = {"x": xs, "y": ys}

    plt.figure(figsize=(5, 5))
    ax = plt.gca()
    # Nodes.
    lw = nodeprops["linewidth"]
    for i, node in enumerate(nodes):
        x = node_coords["x"][i]
        y = node_coords["y"][i]
        # color = node_colors[i]
        node_probe = graph.nodes[node]['probe']
        node_color = probe_color[node_probe]

        node_patch = matplotlib.patches.Circle(
            (x, y), node_r, lw=lw, color=node_color, zorder=2)
        ax.add_patch(node_patch)

    # Edges.
    for i, (start, end) in enumerate(graph.edges()):
      start_theta = node_theta(nodes, start)
      end_theta = node_theta(nodes, end)

      verts = [
          nxviz.geometry.get_cartesian(plot_radius, start_theta),
          (0, 0),
          nxviz.geometry.get_cartesian(plot_radius, end_theta),
      ]
      # color = edge_colors[i]
      color = 'red' if graph.get_edge_data(start, end)['weight'] > 0 else 'blue'

      codes = [matplotlib.path.Path.MOVETO, matplotlib.path.Path.CURVE3,
               matplotlib.path.Path.CURVE3]
      lw = edge_widths[i]
      path = matplotlib.path.Path(verts, codes)
      patch = matplotlib.patches.PathPatch(
          path, lw=lw, edgecolor=color, zorder=1, **edgeprops)
      ax.add_patch(patch)

    # Labels.
    map_probe_to_areas = self.session.map_probe_to_ecephys_structure_acronym(
        visual_only=True)
    print(map_probe_to_areas)
    map_probe_to_areas = {'probeA': 'AM', 'probeB': 'AM', 'probeC': 'V1',
        'probeD': 'LM', 'probeE': 'AL', 'probeF': 'RL'}
    probe_label_cnt = {'probeA': 0, 'probeB': 0, 'probeC': 0, 'probeD': 0,
        'probeE': 0, 'probeF': 0}
    for i, node in enumerate(nodes):
      node_probe = graph.nodes[node]['probe']
      probe_label_cnt[node_probe] += 1
      if probe_label_cnt[node_probe] == 1:
        theta=node_theta(nodes, node) + 0.2
        x, y = nxviz.geometry.get_cartesian(r=radius*1.08, theta=theta)
        deg = theta / np.pi * 180 - 90
        deg = deg - 180 if theta < 0 else deg
        ax.text(x, y, map_probe_to_areas[node_probe],
                rotation=deg, ha='center', va='center', fontsize=12)

    ax.relim()
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis('off')
    if figure_path is not None:
      plt.savefig(figure_path, bbox_inches='tight')
      print('Save figure to:', figure_path)


  def plot_nxviz_graph(
      self,
      graph):
    """Plot quickly using the nxviz."""
    graph_figure = nxviz.plots.CircosPlot(graph, figsize=(10, 10),
        node_labels=False, node_grouping='probe', node_color='color')
    graph_figure.draw()


  def plot_networkx_graph(
      self,
      graph):
    """Plot quickly using the networkx."""
    probe_color = {'probeA': 'k', 'probeB': 'r',
        'probeC': 'g', 'probeD': 'b', 'probeE': 'y',
        'probeF': 'lightgrey'}
    node_color_list = []
    for v in graph:
      node_color_list.append(probe_color[graph.nodes[v]['probe']])

    plt.figure(figsize=(8, 8))
    positions = nx.circular_layout(graph)
    nx.draw_networkx_nodes(graph, pos=positions, node_size=80,
        node_color=node_color_list)
    nx.draw_networkx_edges(graph, pos=positions, edge_color='lightgrey')


  def graph_statistics(
      self,
      graph):
    """Calculates basic statistics of the graph."""
    num_nodes = len(graph.nodes())
    num_edges = len(graph.edges())
    node_degree = nx.get_node_attributes(graph, 'degree')
    node_degree_values = np.array(list(node_degree.values()))
    non_zero_cnt = np.sum(node_degree_values != 0)
    non_zero_ratio = non_zero_cnt / num_nodes

    hub_nodes = []
    for node in node_degree:
      if node_degree[node] > 6:
        hub_nodes.append(node)

    print(f'#nodes {num_nodes}  #edges {num_edges}')
    print(f'#non-zero degree nodes {non_zero_cnt}  ratio {non_zero_ratio}')
    print(f'#hub nodes: {len(hub_nodes)}')
    hub_nodes_probes = self.selected_units.loc[hub_nodes, 'probe_description']
    hub_nodes_probes_cnt = hub_nodes_probes.value_counts()
    print(hub_nodes_probes_cnt)

    map_probe_to_areas = self.session.map_probe_to_ecephys_structure_acronym(
        visual_only=True)
    areas_labels = []
    for probe in hub_nodes_probes_cnt.index.values:
      areas_labels.append(map_probe_to_areas[probe][0])

    plt.figure(figsize=[6, 3])
    ax = plt.gca()
    seaborn.distplot(node_degree_values, kde=False, color='tab:grey')
    plt.ylim(0, 20)
    plt.xlabel('Node degree')

    plt.figure(figsize=[6, 3])
    ax = plt.gca()
    seaborn.barplot(np.arange(len(hub_nodes_probes_cnt)),
        hub_nodes_probes_cnt.values, facecolor='tab:grey')

    plt.xlabel('#hub nodes')
    plt.ylabel('probe')
    ax.set_xticklabels(areas_labels)


  def find_hub_units(
      self,
      graph,
      degree_threshold=30):
    """Find nodes with largest degree."""
    node_degrees = nx.get_node_attributes(graph, 'degree')
    selected_nodes_info = []
    selected_nodes = []

    for node in node_degrees:
      node_degree = node_degrees[node]
      if node_degree >= degree_threshold:
        selected_nodes_info.append((node, node_degree))
        selected_nodes.append(node)

    return selected_nodes, selected_nodes_info


  def plot_unit_xorr(
      self,
      unit,
      xorr_pairs,
      detect_threshold=0.02,
      num_rows_cols=(6,6)):
    """Select and plot xorr edges connected to unit."""
    host_probe = self.selected_units.loc[unit, 'probe_description']
    fig_cnt = 0

    num_rows, num_cols = num_rows_cols
    gs_kw = dict(width_ratios=[1]*num_cols, height_ratios=[1]*num_rows)
    fig, axs = plt.subplots(figsize=(num_cols*3, num_rows*2), gridspec_kw=gs_kw,
        nrows=num_rows, ncols=num_cols)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
    axs = axs.reshape(-1)

    for key in xorr_pairs:
      if not isinstance(key, tuple):
        continue
      unit_x, unit_y = key
      if unit == unit_x:
        order = True
        step = 1
        guest_unit = unit_y
        guest_probe = self.selected_units.loc[unit_y, 'probe_description']
      elif unit == unit_y:
        order = False
        step = -1
        guest_unit = unit_x
        guest_probe = self.selected_units.loc[unit_x, 'probe_description']
      else:
        continue

      # Detect significance.
      xorr_pair = xorr_pairs[(unit_x, unit_y)]
      lags = xorr_pairs['lags'][::step]
      xorr_raw = xorr_pair.raw
      xorr_mean = xorr_pair.mean
      xorr_CI_down = xorr_pair.CI_down
      xorr_CI_up = xorr_pair.CI_up

      (up_significant, down_significant
          ) = self.jittertool.verify_xorr_significance(
          lags, xorr_raw, xorr_mean, xorr_CI_down, xorr_CI_up,
          detect_threshold=detect_threshold, verbose=False)

      if not up_significant and not down_significant:
        continue

      ax = fig.add_subplot(axs[fig_cnt])
      ax.tick_params(left=False, labelleft=False, labelbottom=False, bottom=True,
                     top=False, labeltop=False, direction='in')

      plt.fill_between(lags * 1000, xorr_CI_down - xorr_mean, 
                       xorr_CI_up - xorr_mean,
                       facecolor='lightgrey', alpha=0.8)
      plt.plot(lags * 1000, xorr_raw - xorr_mean, c='k')
      plt.axvline(x=0, lw=0.4, c='darkgrey')
      plt.axhline(y=0, lw=0.4, c='darkgrey')
      
      # plt.text(0.7, 0.9, 'mean corrected', transform=ax.transAxes)
      plt.ylim([-50, 80])
      plt.xlim(lags[0] * 1000 - 5, lags[-1] * 1000 + 5)

      if fig_cnt == num_cols * (num_rows - 1):
        ax.tick_params(left=True, labelleft=True, labelbottom=True, bottom=True)
        plt.xlabel('lag [ms]')

      fig_cnt += 1
      if fig_cnt >= num_rows * num_cols + 1:
        warnings.warn('Number of figures exceeds figure layouts.')
        break

    while fig_cnt < num_rows * num_cols:
      ax = fig.add_subplot(axs[fig_cnt])
      ax.axis('off')
      fig_cnt += 1
    plt.show()


  def xorr_raw_clustering(
      self,
      xorr_pairs,
      detect_threshold=0.02,
      num_clusters=20,
      num_rows_cols=(4,4),
      figure_title=None,
      figure_path=None):
    """Cluster the raw xorr."""
    xorr_mat = []

    for key in xorr_pairs:
      if not isinstance(key, tuple):
        continue
      (unit_x, unit_y) = key
      xorr_pair = xorr_pairs[(unit_x, unit_y)]
      lags = xorr_pairs['lags']
      xorr_raw = xorr_pair.raw
      xorr_mean = xorr_pair.mean
      xorr_CI_down = xorr_pair.CI_down
      xorr_CI_up = xorr_pair.CI_up

      # Detect significance.
      (up_significant, down_significant
          ) = self.jittertool.verify_xorr_significance(
          lags, xorr_raw, xorr_mean, xorr_CI_down, xorr_CI_up,
          detect_threshold=detect_threshold, verbose=False)
      if not up_significant and not down_significant:
        continue
      xorr_corrected = xorr_raw-xorr_mean
      # Uniform the direction.
      if np.sum(xorr_corrected[lags > 0]) < np.sum(xorr_corrected[lags < 0]):
        xorr_corrected = xorr_corrected[::-1]
      xorr_mat.append(xorr_corrected)

    # Perform clustering on the xorr mean.
    xorr_mat = np.vstack(xorr_mat)
    num_xorrs = xorr_mat.shape[0]
    if num_xorrs < 3:
      return
    if num_xorrs < num_clusters:
      num_clusters = num_xorrs
    kmeans = sklearn.cluster.KMeans(
        n_clusters=num_clusters, random_state=0).fit(xorr_mat)
    # kmeans = sklearn.cluster.KMeans(
    #     n_clusters=num_clusters, random_state=0).fit(xorr_mat[:,10:-10])
    cluster_sample_cnt = []
    for cluster_id in range(num_clusters):
      cluster_labels = np.where(kmeans.labels_ == cluster_id)[0]
      num_samples = len(cluster_labels)
      cluster_sample_cnt.append((cluster_id, num_samples))
    cluster_sample_cnt = sorted(cluster_sample_cnt, key=lambda x: x[1], reverse=True)
    print('(cluster_id, cnt)', cluster_sample_cnt)

    num_rows, num_cols = num_rows_cols
    gs_kw = dict(width_ratios=[1]*num_cols, height_ratios=[1]*num_rows)
    fig, axs = plt.subplots(figsize=(num_cols*4, num_rows*2), gridspec_kw=gs_kw,
        nrows=num_rows, ncols=num_cols)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
    axs = axs.reshape(-1)

    for fig_id, (cluster_id, num_samples) in enumerate(cluster_sample_cnt):
      if fig_id >= num_cols * num_rows:
        break
      ax = fig.add_subplot(axs[fig_id])
      ax.tick_params(left=False, labelleft=False, labelbottom=False, bottom=True,
                     top=False, labeltop=False, direction='in')

      cluster_labels = np.where(kmeans.labels_ == cluster_id)[0]
      num_samples = len(cluster_labels)
      cluster_samples = xorr_mat[cluster_labels]
      plt.axvline(x=0, color='lightgrey', lw=1)
      plt.axhline(y=0, color='lightgrey', lw=1)
      plt.plot(lags * 1000, cluster_samples.T, 'lightgrey', lw=0.1)
      plt.plot(lags * 1000, cluster_samples.mean(axis=0), 'k')
      plt.ylim(-40, 60)
      plt.text(0.05, 0.85, f'id={cluster_id}', transform=ax.transAxes)
      plt.text(0.8, 0.85, f'n={num_samples}',  c='k', transform=ax.transAxes)

      if fig_id == num_cols * (num_rows - 1):
        ax.tick_params(left=True, labelleft=True, labelbottom=True, bottom=True)
        plt.xlabel('lag [ms]')

    fig.suptitle(figure_title, fontsize=16)
    if figure_path is not None:
      plt.savefig(figure_path, bbox_inches='tight')
      print('Save figure to:', figure_path)
    plt.show()


  def xorr_raw_clustering_grouped(
      self,
      xorr_pairs,
      detect_threshold=0.02,
      output_dir=None):
    """Cluster the raw xorr for each pair of probes."""
    xorr_pairs_grouped = self.group_xorr_pairs(xorr_pairs, verbose=False)

    map_probe_to_areas = self.session.map_probe_to_ecephys_structure_acronym(
        visual_only=True)
    print(map_probe_to_areas)
    map_probe_to_areas = {'probeA': 'AM', 'probeB': 'AM', 'probeC': 'V1',
        'probeD': 'LM', 'probeE': 'AL', 'probeF': 'RL'}

    for key in xorr_pairs_grouped:
      probe_x, probe_y = key
      figure_title = (probe_x + '_' + map_probe_to_areas[probe_x] + '___' +
                      probe_y + '_' + map_probe_to_areas[probe_y])
      print(key, figure_title)
      figure_path = os.path.join(output_dir, figure_title+'_xorr_clusters.pdf')
      self.xorr_raw_clustering(xorr_pairs_grouped[key], detect_threshold=0.02,
          num_clusters=8, num_rows_cols=(2,3), figure_title=figure_title,
          figure_path=figure_path)


  def cfpp_clustering(
      self,
      model_list,
      num_clusters=8):
    """Cluster the raw xorr."""
    beta_threshold = 100
    num_models = len(model_list)
    model_par0 = model_list[0][2]
    num_beta = len(model_par0['beta'])
    if 'append_nuisance' in model_par0:
      num_nuisance = len(model_par0['append_nuisance'])
      num_beta = num_beta - num_nuisance
    else:
      num_nuisance = 0
    print('num_models', num_models, 'num non-nuisance beta', num_beta)
    beta_mat = np.zeros([num_models, num_beta])
    num_itrs = np.zeros(num_models)

    _,filter_t,_,_ = self.jittertool.reconstruct_bspline_basis(model_par0, dt=0.001)
    num_plot_bins = len(filter_t)
    filter_mat = np.zeros([num_models, num_plot_bins])

    outlier_cnt = 0
    for m, (group_id, (neuron_x, neuron_y), model_par) in enumerate(model_list):
      num_itrs[m] = model_par['num_itrs']
      beta_slot = model_par['beta'][num_nuisance:,0]
      if any(np.abs(beta_slot) > beta_threshold):
        outlier_cnt += 1
        continue
      beta_mat[m] = beta_slot
      _,_,h,_ = self.jittertool.reconstruct_bspline_basis(model_par, dt=0.001)
      filter_mat[m] = h

    # Perform clustering on the betas.
    kmeans = sklearn.cluster.KMeans(
        n_clusters=num_clusters, random_state=0).fit(beta_mat)

    neuron_pairs = [model_list[m][0] for m in range(num_models)]
    filter_membership = pd.DataFrame(
        kmeans.labels_.astype(int),
        index=pd.MultiIndex.from_tuples(neuron_pairs),
        columns=['group_id'])
    cluster_cnt = filter_membership.value_counts()
    cluster_cnt = [(idx[0], cluster_cnt[idx]) for idx in cluster_cnt.index]
    print('outlier_cnt', outlier_cnt)
    print('cluster membership cnt', cluster_cnt)

    return filter_membership, beta_mat, filter_t, filter_mat


  @classmethod
  def init_filter_membership(
      cls,
      neuron_pairs,
      trial_ids=None,
      random_val=False):
    """Initialize `filter_membership` as DataFrame."""
    num_pairs = len(neuron_pairs)
    if trial_ids is not None:
      num_trials = len(trial_ids)
      if random_val:
        group_ids = np.random.randint(0, 5, size=[num_pairs, num_trials])
      else:
        group_ids = np.zeros([num_pairs, num_trials]).astype(int)
      filter_membership = pd.DataFrame(
          group_ids,
          index=pd.MultiIndex.from_tuples(neuron_pairs),
          columns=trial_ids)
    else:
      if random_val:
        group_ids = np.random.randint(0, 5, size=num_pairs)
      else:
        group_ids = np.zeros(num_pairs)
      filter_membership = pd.DataFrame(
          group_ids,
          index=pd.MultiIndex.from_tuples(neuron_pairs))
    return filter_membership


  def filter_membership_statistics(
      self,
      filter_membership,
      ylim=None,
      title=None,
      file_path=None,
      verbose=True):
    """Basic statistics of filter_membership."""
    cluster_cnt = filter_membership.apply(pd.value_counts, sort=False, dropna=False)
    cluster_cnt = cluster_cnt.sum(axis=1)
    cluster_cnt =[(index, row) for index, row in cluster_cnt.iteritems()]
    cluster_cnt.sort(key=lambda x: x[0], reverse=False)
    if verbose:
      print('filter_membership.shape', filter_membership.shape)
      print('membership cnt', cluster_cnt)

    if verbose == 2:
      cluster_cnt_filter = filter_membership.apply(pd.value_counts,
          sort=False, axis=1, dropna=False)
      cluster_mean = cluster_cnt_filter.mean(axis=0)
      cluster_std = cluster_cnt_filter.std(axis=0)
      cluster_mse = cluster_std / np.sqrt(cluster_cnt_filter.shape[0])
      # print(cluster_mean)
      # print(cluster_std)
      # print(cluster_mse)
      x_pos = [0,1,2,3,4,5]
      gs_kw = dict(width_ratios=[1], height_ratios=[1])
      fig, axs = plt.subplots(figsize=(4, 2), gridspec_kw=gs_kw,
          nrows=1, ncols=1)
      plt.subplots_adjust(left=None, right=None, hspace=0.15, wspace=0.15)
      ax = fig.add_subplot(axs)
      ax.bar(x_pos, cluster_mean, yerr=cluster_std, align='center', alpha=0.5,
          color='grey', ecolor='black', capsize=2)
      ax.set_ylabel('Number of trials')
      plt.xticks(x_pos, ['no effect', 'excitatory', 'inhibitory',
          'oscillatory-1', 'oscillatory-2', 'empty trial'], rotation=-45, ha='left',
          rotation_mode='anchor')
      ax.set_title(title)
      if ylim is not None:
        plt.ylim(ylim)
      else:
        plt.ylim(0, 100)
      if file_path is not None:
        plt.savefig(file_path)
        print('save figure:', file_path)
      plt.show()

    return cluster_cnt


  def build_neuron_graph(
      self,
      filter_membership,
      sort_nodes=False,
      verbose=False,
      file_path=None):
    """Build graph for the filters.

    Args:
        filter_membership: It has to be one column.
    """
    graph = nx.DiGraph()
    for i, (neuron_x, neuron_y) in enumerate(filter_membership.index):
      graph.add_edge(neuron_x, neuron_y,
          weight=filter_membership.loc[(neuron_x, neuron_y)])
    # Set node attr.
    nodes = graph.nodes()
    attrs = {node:{'probe':self.selected_units.loc[node, 'probe_description']}
             for node in nodes}
    nx.set_node_attributes(graph, attrs)
    for v in graph:
      graph.nodes[v]['degree'] = graph.degree(v, weight='weight').values[0]

    if verbose == 1:
      cluster_cnt = filter_membership.value_counts()
      cluster_cnt = [(idx[0], cluster_cnt[idx]) for idx in cluster_cnt.index]
      print('cluster membership cnt', cluster_cnt)

    if verbose == 2:
      cluster_cnt = filter_membership.value_counts()
      cluster_cnt = [(idx[0], cluster_cnt[idx]) for idx in cluster_cnt.index]
      print('cluster membership cnt', cluster_cnt)

      if sort_nodes:
        nodes = graph.nodes(data=True)
        nodes = sorted(nodes, key=lambda x: x[1]['degree'], reverse=True)
        nodes = sorted(nodes, key=lambda x: x[1]['probe'], reverse=False)
        nodelist = [node[0] for node in nodes]

        probelist = [node[1]['probe'] for node in nodes]
        probe_unit_cnt = pd.DataFrame(probelist).value_counts(sort=False).values
        probe_unit_cum = np.cumsum(np.hstack([[0], probe_unit_cnt]))
      else:
        nodelist = graph.nodes()

      graph_mask = nx.DiGraph()
      for i, (neuron_x, neuron_y) in enumerate(filter_membership.index):
        graph_mask.add_edge(neuron_x, neuron_y)
      adj = nx.to_numpy_matrix(graph, nodelist=nodelist)
      adj_mask = nx.to_numpy_matrix(graph_mask, nodelist=nodelist).astype(bool)
      adj_mask = ~adj_mask

      fig, axs = plt.subplots(figsize=(6, 5), nrows=1, ncols=1)
      ax = fig.add_subplot(axs)
      seaborn.heatmap(adj, square=True, mask=adj_mask, cmap='RdYlBu')
      plt.axhline(0, c='lightgrey', lw=2)
      plt.axhline(adj.shape[0], c='lightgrey', lw=2)
      plt.axvline(0, color='lightgrey', lw = 2)
      plt.axvline(adj.shape[1], c='lightgrey', lw=2)

      if sort_nodes:
        plt.xticks(probe_unit_cum, probe_unit_cum)
        plt.yticks(probe_unit_cum, probe_unit_cum)
        probe_names = ['V1', 'LM', 'AL']
        for l, val in enumerate(probe_unit_cum):
          plt.axhline(val, c='grey', lw=0.5)
          plt.axvline(val, c='grey', lw=0.5)
          if l <= len(probe_unit_cum)-2:
            plt.text(val+10, -1, f'{probe_names[l]}')

      if file_path is not None:
        plt.savefig(file_path, bbox_inches='tight')
        print('Save figure to:', file_path)
      plt.show()

    return graph


  def plot_filter_clusters(
      self,
      model_list,
      filter_membership,
      beta_mat,
      filter_t,
      filter_mat,
      num_rows_cols=(4,4),
      figure_title=None,
      file_path=None):
    """Plot each clusters."""
    num_models = len(model_list)
    labels_ = np.zeros(num_models)
    for m, ((neuron_x, neuron_y), model_par) in enumerate(model_list):
      labels_[m] = filter_membership.loc[(neuron_x, neuron_y)]
    cluster_cnt = filter_membership.value_counts()
    cluster_cnt = [(idx[0], cluster_cnt[idx]) for idx in cluster_cnt.index]

    num_rows, num_cols = num_rows_cols
    gs_kw = dict(width_ratios=[1]*num_cols, height_ratios=[1]*num_rows)
    fig, axs = plt.subplots(figsize=(num_cols*4, num_rows*2), gridspec_kw=gs_kw,
        nrows=num_rows, ncols=num_cols)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
    axs = axs.reshape(-1)
    for ax in axs:
      ax.tick_params(left=True, labelleft=False, labelbottom=False, bottom=True,
                     direction='in')
    for fig_id, (cluster_id, num_samples) in enumerate(cluster_cnt):
      if fig_id >= num_cols * num_rows:
        break
      ax = fig.add_subplot(axs[fig_id])
      ax.tick_params(left=True, labelleft=False, labelbottom=False, bottom=True,
                     direction='in')

      cluster_labels = np.where(labels_ == cluster_id)[0]
      num_samples = len(cluster_labels)
      cluster_samples = beta_mat[cluster_labels]
      plt.axvline(x=0, color='lightgrey', lw=1)
      plt.axhline(y=0, color='lightgrey', lw=1)
      plt.plot(cluster_samples.T, 'lightgrey', lw=0.1)
      plt.plot(cluster_samples.mean(axis=0), 'k')
      plt.ylim(-5, 15)
      plt.text(0.05, 0.85, f'id={cluster_id}', transform=ax.transAxes)
      plt.text(0.8, 0.85, f'n={num_samples}',  c='k', transform=ax.transAxes)

      if fig_id == num_cols * (num_rows - 1):
        ax.tick_params(left=True, labelleft=True, labelbottom=True, bottom=True)
        plt.xlabel('beta index')

    # fig.suptitle(figure_title, fontsize=16)
    # if file_path is not None:
    #   plt.savefig(file_path, bbox_inches='tight')
    #   print('Save figure to:', file_path)
    plt.show()

    # Plot filters.
    fig, axs = plt.subplots(figsize=(num_cols*4, num_rows*2), gridspec_kw=gs_kw,
        nrows=num_rows, ncols=num_cols)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
    axs = axs.reshape(-1)
    for ax in axs:
      ax.tick_params(left=True, labelleft=False, labelbottom=False, bottom=True,
                     direction='in')
    for fig_id, (cluster_id, num_samples) in enumerate(cluster_cnt):
      if fig_id >= num_cols * num_rows:
        break
      ax = fig.add_subplot(axs[fig_id])
      ax.tick_params(left=True, labelleft=False, labelbottom=False, bottom=True,
                     direction='in')

      cluster_labels = np.where(labels_ == cluster_id)[0]
      num_samples = len(cluster_labels)
      cluster_samples = filter_mat[cluster_labels]
      plt.axvline(x=0, color='lightgrey', lw=1)
      plt.axhline(y=0, color='lightgrey', lw=1)
      plt.plot(filter_t*1000, cluster_samples.T, 'lightgrey', lw=0.1)
      plt.plot(filter_t*1000, cluster_samples.mean(axis=0), 'k')
      plt.ylim(-4, 8)
      plt.text(0.05, 0.85, f'id={cluster_id}', transform=ax.transAxes)
      plt.text(0.8, 0.85, f'n={num_samples}',  c='k', transform=ax.transAxes)

      if fig_id == num_cols * (num_rows - 1):
        ax.tick_params(left=True, labelleft=True, labelbottom=True, bottom=True)
        plt.xlabel('lag [ms]')

    fig.suptitle(figure_title, fontsize=16)
    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      print('Save figure to:', file_path)
    plt.show()


  def relabel_clusters(
      self,
      filter_membership,
      map_raw_new_order,
      verbose=False):
    """Relabel clusters according to `fit_cluster_filter`."""
    filter_membership = filter_membership.replace({'group_id': map_raw_new_order})
    if verbose:
      cluster_cnt = filter_membership.value_counts()
      cluster_cnt = [(idx[0], cluster_cnt[idx]) for idx in cluster_cnt.index]
      print('cluster membership cnt', cluster_cnt)
    return filter_membership


  @classmethod
  def stack_spike_times_by_pairs_all(
      cls,
      spike_times,
      filter_membership,
      batch_size=1000,
      random_seed=None,
      verbose=False):
    """Stack spike times together."""
    random.seed(random_seed)
    pair_trial_list = [(filter_membership.index[r], filter_membership.columns[c])
        for r, c in np.argwhere((filter_membership>=0).values)]

    if batch_size is not None and batch_size < len(pair_trial_list):
      pair_trial_batch = random.sample(pair_trial_list, batch_size)
    else:
      pair_trial_batch = pair_trial_list

    if verbose:
      print(f'#all trials:{len(pair_trial_list)}\t' +
            f'#batch trials:{len(pair_trial_batch)}')
      print(pair_trial_batch)

    spike_times_x, spike_times_y = [], []
    for (neuron_x, neuron_y), trial_id in pair_trial_batch:
      spike_times_x = spike_times_x + spike_times.loc[neuron_x,[trial_id]].tolist()
      spike_times_y = spike_times_y + spike_times.loc[neuron_y,[trial_id]].tolist()

    return spike_times_x, spike_times_y


  @classmethod
  def stack_spike_times_by_pairs(
      cls,
      spike_times,
      filter_membership,
      group_id,
      batch_size=None,
      verbose=False):
    """Stack spike times together from group_id."""
    pair_trial_list = [(filter_membership.index[r], filter_membership.columns[c])
        for r, c in np.argwhere((filter_membership==group_id).values)]
    if batch_size is not None and batch_size < len(pair_trial_list):
      pair_trial_batch = random.sample(pair_trial_list, batch_size)
    else:
      pair_trial_batch = pair_trial_list

    if verbose:
      print(f'#all trials:{len(pair_trial_list)}\t' + 
            f'#batch trials:{len(pair_trial_batch)}')

    spike_times_x, spike_times_y = [], []
    for (neuron_x, neuron_y), trial_id in pair_trial_batch:
      spike_times_x = spike_times_x + spike_times.loc[neuron_x,[trial_id]].tolist()
      spike_times_y = spike_times_y + spike_times.loc[neuron_y,[trial_id]].tolist()

    return spike_times_x, spike_times_y


  @classmethod
  def stack_spike_times_multivariate(
      cls,
      spike_times,
      neuron_ids,
      trial_ids,
      verbose=False):
    """Stack spike times together."""
    num_neurons = len(neuron_ids)
    spike_times_multi = [0] * num_neurons
    for n, neuron in enumerate(neuron_ids):
      spike_times_multi[n] = spike_times.loc[neuron,trial_ids].tolist()

    return spike_times_multi


  @classmethod
  def pairwise_bivariate_regression_runner(self, neuron_pair):
    """The function has to be pickalbe.
    So need to use classmethod for multiprocessing runner.
    """
    print(neuron_pair, 'start')
    model_hat = self.jittertool.bivariate_continuous_time_coupling_filter_regression(
        spike_times_x, spike_times_y, trial_window, model_par, mute_warning=True)
    print(neuron_pair, 'done')
    return neuron_pair, model_hat


  def pairwise_bivariate_regression(
      self,
      filter_membership,
      model_par=None,
      verbose=False,
      file_dir=None,
      parallel=False,
      num_threads=30):
    """Regression for all indicies (pairs)."""
    spike_times = self.spike_times
    trial_window = self.trial_window
    file_path = None

    if parallel:
      import multiprocessing
      pool = multiprocessing.Pool(num_threads)

      global model_list
      model_list = []
      def get_results(result):
        model_list.append(result)

      for i, neuron_pair in enumerate(filter_membership.index):
        if i > 30:
          break

        filter_membership_sub = filter_membership.loc[[(neuron_x,neuron_y)],:]
        spike_times_x, spike_times_y = self.stack_spike_times_by_pairs_all(
            spike_times, filter_membership_sub, batch_size=None, verbose=False)

        pool.apply_async(self.pairwise_bivariate_regression_runner,
            args=(neuron_pair, filter_membership_sub, spike_times_x, spike_times_y,
                trial_window, model_par),
            callback=get_results)

        # res = pool.apply_async(self.runner, args=(neuron_pair,)).get()
        # print(res)
        # print(out[0])

      pool.close()
      pool.join()
      return model_list

    else:
      model_list = []
      if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()
      trange = enumerate(tqdm(filter_membership.index, ncols=100, file=sys.stdout))

      for i, (neuron_x, neuron_y) in trange:
        filter_membership_sub = filter_membership.loc[[(neuron_x,neuron_y)],:]
        spike_times_x, spike_times_y = self.stack_spike_times_by_pairs_all(
            spike_times, filter_membership_sub, batch_size=None, verbose=False)

        model_hat = self.jittertool.bivariate_continuous_time_coupling_filter_regression(
            spike_times_x, spike_times_y, trial_window, model_par, mute_warning=True)
        model_list.append(((neuron_x, neuron_y), model_hat))

        if verbose:
          self.jittertool.spike_times_statistics(spike_times_x, trial_window[1], verbose=1)
          self.jittertool.spike_times_statistics(spike_times_y, trial_window[1], verbose=1)

          print(f'{(neuron_x, neuron_y)}')
          if file_dir is not None:
            file_path = file_dir + f'trial{trial_id}_regression.pdf'
          self.jittertool.plot_continuous_time_bivariate_regression_model_par(
              model_list[-1][1], ylim=[-10, 10])

    return model_list


  @classmethod
  def pairwise_bivariate_full_regression_runner(self, neuron_pair):
    """The function has to be pickalbe.
    So need to use classmethod for multiprocessing runner.
    """
    print(neuron_pair, 'start')
    model_hat = self.jittertool.bivariate_continuous_time_coupling_filter_full_regression(
        spike_times_x, spike_times_y, trial_window, model_par, mute_warning=True)
    print(neuron_pair, 'done')
    return neuron_pair, model_hat


  def pairwise_bivariate_full_regression(
      self,
      filter_membership,
      model_par=None,
      verbose=False,
      file_dir=None,
      parallel=False,
      num_threads=30):
    """Regression for all indicies (pairs)."""
    spike_times = self.spike_times
    trial_window = self.trial_window
    file_path = None

    if parallel:
      import multiprocessing
      pool = multiprocessing.Pool(num_threads)

      global model_list
      model_list = []
      def get_results(result):
        model_list.append(result)

      for i, neuron_pair in enumerate(filter_membership.index):
        if i > 30:
          break

        filter_membership_sub = filter_membership.loc[[(neuron_x,neuron_y)],:]
        spike_times_x, spike_times_y = self.stack_spike_times_by_pairs_all(
            spike_times, filter_membership_sub, batch_size=None, verbose=False)

        pool.apply_async(self.pairwise_bivariate_full_regression_runner,
            args=(neuron_pair, filter_membership_sub, spike_times_x, spike_times_y,
                trial_window, model_par),
            callback=get_results)

        # res = pool.apply_async(self.runner, args=(neuron_pair,)).get()
        # print(res)
        # print(out[0])

      pool.close()
      pool.join()
      return model_list

    else:
      model_list = []
      if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()
      trange = enumerate(tqdm(filter_membership.index, ncols=100, file=sys.stdout))

      for i, (neuron_x, neuron_y) in trange:
        filter_membership_sub = filter_membership.loc[[(neuron_x,neuron_y)],:]
        spike_times_x, spike_times_y = self.stack_spike_times_by_pairs_all(
            spike_times, filter_membership_sub, batch_size=None, verbose=False)

        model_hat = self.jittertool.pairwise_bivariate_full_regression_runner(
            spike_times_x, spike_times_y, trial_window, model_par, mute_warning=True)
        model_list.append(((neuron_x, neuron_y), model_hat))

        if verbose:
          self.jittertool.spike_times_statistics(spike_times_x, trial_window[1], verbose=1)
          self.jittertool.spike_times_statistics(spike_times_y, trial_window[1], verbose=1)

          print(f'{(neuron_x, neuron_y)}')
          if file_dir is not None:
            file_path = file_dir + f'trial{trial_id}_regression.pdf'
          self.jittertool.plot_continuous_time_bivariate_regression_model_par(
              model_list[-1][1], ylim=[-10, 10])

    return model_list


  def pairwise_bivariate_regression_inference(
      self,
      model_list,
      filter_membership):
    """Inference for the output from `pairwise_bivariate_regression`."""
    selected_units = self.selected_units
    df_inference = pd.DataFrame(index=filter_membership.index)
    df_inference.index.names = ['source', 'target']

    for m, (neuron_pair, model_par) in enumerate(model_list):
        pval = self.jittertool.bivariate_continuous_time_coupling_filter_regression_inference_single(model_par)
        df_inference.loc[neuron_pair, 'pval'] = pval
        num_nuisance = len(model_par['append_nuisance'])
        df_inference.loc[neuron_pair, 'h'] = model_par['beta'][num_nuisance,0]

    # display(df_inference)
    # display(df_inference[df_inference.pval<1e-4])
    df_inference = df_inference.reset_index()
    df_inference = pd.merge(df_inference, selected_units[['probe_description']], left_on='source', right_on='unit_id')
    df_inference = df_inference.rename(columns={'probe_description': 'source_probe'})
    df_inference = pd.merge(df_inference, selected_units[['probe_description']], left_on='target', right_on='unit_id')
    df_inference = df_inference.rename(columns={'probe_description': 'target_probe'})

    df_inference = pd.merge(df_inference, selected_units[['ecephys_structure_acronym']], left_on='source', right_on='unit_id')
    df_inference = df_inference.rename(columns={'ecephys_structure_acronym': 'source_area'})
    df_inference = pd.merge(df_inference, selected_units[['ecephys_structure_acronym']], left_on='target', right_on='unit_id')
    df_inference = df_inference.rename(columns={'ecephys_structure_acronym': 'target_area'})

    df_inference = df_inference.set_index(['source','target'])
    return df_inference



  # Warning: this is for group-wise fitting.
  def each_pair_all_trials_bivariate_regression(
      self,
      filter_membership,
      data_within='trial',
      model_par=None,
      group_ids=[0,1,2,3,4],
      verbose=False,
      file_dir=None):
    """Regression for all pairs within groups."""
    spike_times = self.spike_times
    trial_window = self.trial_window
    model_list = []
    file_path = None

    # With jitter-correction.
    model_par_template = {'filter_type': 'bspline', 'num_knots': 8,
        'knot_space_par': 0.05, 'filter_length': 0.05, 'num_tail_drop': 1,
        'append_nuisance': ['const', 'triangle_kernel'], 'kernel_width': 0.12,
        'const_offset': 0, 'learning_rate': 0.9, 'max_num_itrs': 50,
        'epsilon': 1e-5}

    # No jitter-correction.
    # model_par_template = {'filter_type': 'bspline', 'num_knots': 8,
    #     'knot_space_par': 0.05, 'filter_length': 0.05, 'num_tail_drop': 1,
    #     'append_nuisance': ['const'],
    #     'const_offset': 0, 'learning_rate': 0.9, 'max_num_itrs': 50,
    #     'epsilon': 1e-5}

    if model_par is None:
      model_par = model_par_template

    if data_within == 'trial':
      if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()
      trange = enumerate(tqdm(filter_membership.columns, ncols=100, file=sys.stdout))
      for i, trial_id in trange:
        filter_membership_sub = filter_membership.loc[:,[trial_id]]
        for group_id in group_ids:
          spike_times_x, spike_times_y = self.stack_spike_times_by_pairs(
              spike_times, filter_membership_sub, group_id=group_id,
              verbose=False)
          if group_id in [0,1,2]:
            model_par['filter_length'] = 0.05
          elif group_id in [3,4]:
            model_par['filter_length'] = 0.04
          model_hat = self.jittertool.bivariate_continuous_time_coupling_filter_regression(
              spike_times_x, spike_times_y, trial_window, model_par, mute_warning=True)
          model_list.append((group_id, trial_id, model_hat))

          if verbose:
            print(f'group_id {group_id}  {trial_id}')
            if file_dir is not None:
              file_path = file_dir + (f'trial{trial_id}_group{group_id}_'+
                  'within_trial_nuisance_regression.pdf')
            self.jittertool.plot_continuous_time_bivariate_regression_model_par(
                model_list[-1][2], ylim=[-7, 22], file_path=file_path)

    elif data_within == 'neuron':
      if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()
      trange = enumerate(tqdm(filter_membership.index, ncols=100, file=sys.stdout))
      for i, (neuron_x, neuron_y) in trange:
        filter_membership_sub = filter_membership.loc[[(neuron_x,neuron_y)],:]
        for group_id in group_ids:
          spike_times_x, spike_times_y = self.stack_spike_times_by_pairs(
              spike_times, filter_membership_sub, group_id=group_id,
              verbose=False)
          if group_id in [0,1,2]:
            model_par['filter_length'] = 0.05
          elif group_id in [3,4]:
            model_par['filter_length'] = 0.04
          model_hat = self.jittertool.bivariate_continuous_time_coupling_filter_regression(
              spike_times_x, spike_times_y, trial_window, model_par, mute_warning=True)
          model_list.append((group_id, (neuron_x, neuron_y), model_hat))

          if verbose:
            print(f'group_id {group_id}  {(neuron_x, neuron_y)}')
            if file_dir is not None:
              file_path = file_dir + (f'trial{trial_id}_group{group_id}_'+
                  'within_neuron_nuisance_regression.pdf')
            self.jittertool.plot_continuous_time_bivariate_regression_model_par(
                model_list[-1][2], ylim=[-20, 20])

    return model_list


  def each_pair_all_trials_bivariate_regression_jitter(
      self,
      filter_membership,
      data_within='trial',
      model_par=None,
      verbose=False,
      file_dir=None):
    """Regression for all pairs."""
    spike_times = self.spike_times
    trial_window = self.trial_window
    group_ids = [0,1,2,3,4]
    file_path = None
    jitter_window_width = 0.12
    max_num_itrs = 60
    num_jitter = 50
    epsilon = 1e-5
    lr=0.9
    model_list = []

    model_par_square = {
        'filter_type': 'square', 'filter_length': 0.05,
        'append_nuisance': ['const'],
        'const_offset': 0, 'learning_rate': lr, 'random_seed':0,
        'max_num_itrs': max_num_itrs, 'epsilon': epsilon,
        'num_jitter': num_jitter, 'jitter_window_width': jitter_window_width}
    model_par_bspline = {
        'filter_type': 'bspline', 'num_knots': 8, 'knot_space_par': 0.05,
        'num_tail_drop': 1, 'filter_length': 0.04,
        'append_nuisance': ['const'],
        'const_offset': 0, 'learning_rate': lr, 'random_seed':0,
        'max_num_itrs': max_num_itrs, 'epsilon': epsilon,
        'num_jitter': num_jitter, 'jitter_window_width': jitter_window_width}

    if data_within == 'trial':
      if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()
      trange = enumerate(tqdm(filter_membership.columns, ncols=100, file=sys.stdout))
      for i, trial_id in trange:
        filter_membership_sub = filter_membership.loc[:,[trial_id]]
        for group_id in group_ids:
          spike_times_x, spike_times_y = self.stack_spike_times_by_pairs(
              spike_times, filter_membership_sub, group_id=group_id,
              verbose=False)
          if group_id in [0,1,2]:
            model_par = model_par_square
          elif group_id in [3,4]:
            model_par = model_par_bspline
          model_par['model_id'] = group_id

          (model_par_raw, model_par_jitter
              ) = self.jittertool.bivariate_continuous_time_coupling_filter_regression_jitter(
              spike_times_x, spike_times_y, trial_window, model_par, verbose=False)

          if verbose:
            print(f'group_id {group_id}  {trial_id}')
            if file_dir is not None:
              file_path = file_dir + (f'trial{trial_id}_group{group_id}_'+
                  'within_trial_jitter_regression.pdf')
            self.jittertool.plot_continuous_time_bivariate_regression_jitter_model_par(
                model_par_raw, model_par_jitter, ylim=[-7, 22], file_path=file_path)

    elif data_within == 'neuron':
      if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()
      trange = enumerate(tqdm(filter_membership.index, ncols=100, file=sys.stdout))
      for i, (neuron_x, neuron_y) in trange:
        filter_membership_sub = filter_membership.loc[[(neuron_x,neuron_y)],:]
        for group_id in group_ids:
          spike_times_x, spike_times_y = self.stack_spike_times_by_pairs(
              spike_times, filter_membership_sub, group_id=group_id,
              verbose=False)
          if group_id in [0,1,2]:
            model_par['filter_length'] = 0.05
          elif group_id in [3,4]:
            model_par['filter_length'] = 0.04
          model_par['model_id'] = group_id

          (model_par_raw, model_par_jitter
              ) = self.jittertool.bivariate_continuous_time_coupling_filter_regression_jitter(
              spike_times_x, spike_times_y, trial_window, model_par, verbose=False)

          if verbose:
            print(f'group_id {group_id}  {trial_id}')
            if file_dir is not None:
              file_path = file_dir + (f'trial{trial_id}_group{group_id}_'+
                  'within_neuron_jitter_regression.pdf')
            self.jittertool.plot_continuous_time_bivariate_regression_jitter_model_par(
                model_par_raw, model_par_jitter, ylim=[-7, 22], file_path=file_path)

    return model_list


  def plot_model_list_by_group_id(
      self,
      model_list,
      group_ids,
      group_model_pars=None,
      file_path=None):
    """Plot all filters by group_id.

    This function plots the results of `each_pair_all_trials_bivariate_regression`.

    Args:
      data_group: data format of model_list, either `trial` or `neuron`.
      group_model_pars: dict. outcome from `update_cluster_filter_bspline`,
          `update_cluster_filter_joint_trail`.
    """
    num_groups = len(group_ids)

    gs_kw = dict(width_ratios=[1]*num_groups, height_ratios=[1])
    fig, axs = plt.subplots(figsize=(num_groups*4, 2), gridspec_kw=gs_kw,
        nrows=1, ncols=num_groups)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
    axs = [axs] if num_groups == 1 else axs

    for fig_id, group_id in enumerate(group_ids):
      ax = fig.add_subplot(axs[fig_id])
      ax.tick_params(left=True, labelleft=False, labelbottom=False, bottom=True,
                     direction='in')
      if fig_id == 0:
        ax.tick_params(labelleft=True, labelbottom=True)

      h_mat = []
      for model_group_id, data_idx, model_par in model_list:
        # `data_idx` can be trial_id or neuron_pair.
        if model_group_id != group_id:
          continue
        t, h, _ = self.jittertool.reconstruct_basis(model_par)
        if any(np.abs(h) > 60) or any(np.isnan(h)):
          continue
        h_mat.append(h)

      h_mat = np.vstack(h_mat)
      h_mean = h_mat.mean(axis=0)
      h_ci_up = np.quantile(h_mat, 0.975, axis=0)
      h_ci_down = np.quantile(h_mat, 0.025, axis=0)
      plt.axhline(0, color='lightgrey', lw=0.8)
      plt.plot(t*1000, h_mean, 'k', label='Mean')
      plt.fill_between(t*1000, h_ci_up, h_ci_down,
                       facecolor='lightgrey', alpha=0.3, label='95% CI')

      if group_model_pars is not None:
        group_model_par = group_model_pars[group_id]
        t, h, _ = self.jittertool.reconstruct_basis(group_model_par)
        plt.plot(t*1000, h, 'g', label='Model')

      plt.ylim(-10, 22)
      plt.xlim(0, 60)
      if fig_id == 0:
        plt.legend(ncol=1)
        plt.xlabel('Time [ms]')
        plt.ylabel('Firing rate [spikes/s]')

    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      print('Save figure to:', file_path)
    plt.show()


  @classmethod
  def plot_model_list_checkerboard(
      cls,
      model_list,
      file_path=None):
    """Plot all filters in one checkerboard."""
    group_ids, pairs, _ = zip(*model_list)
    row_neurons, col_neurons = zip(*pairs)
    row_neurons = np.unique(row_neurons)
    col_neurons = np.unique(col_neurons)
    num_rows, num_cols = len(row_neurons), len(col_neurons)

    gs_kw = dict(width_ratios=[1]*num_cols, height_ratios=[1]*num_rows)
    fig, axs = plt.subplots(figsize=(num_rows*1.5, num_cols*1), gridspec_kw=gs_kw,
        nrows=num_rows, ncols=num_cols)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
    for m, (neuron_x, neuron_y) in enumerate(pairs):
      row_id = np.where(row_neurons == neuron_x)[0][0]
      col_id = np.where(col_neurons == neuron_y)[0][0]
      ax = fig.add_subplot(axs[row_id, col_id])
      group_id, (neuron_x, neuron_y), model_par = model_list[m]
      t, h, h_std = jitter.JitterTool.reconstruct_basis(model_par)
      ax.tick_params(left=True, labelleft=False, labelbottom=False, bottom=True,
                     direction='in')
      if row_id == num_rows-1 and col_id == 0:
        ax.tick_params(left=True, labelleft=True, labelbottom=True, bottom=True,
                       direction='in')
        plt.yticks([-10, 10], [-10, 10])
      if row_id == 0 and col_id == 0:
        ax.tick_params(left=True, labelleft=True, labelbottom=False, bottom=True,
                       direction='in')
        plt.yticks([-40, 20], [-40, 20])
      if row_id == col_id:
        ax.set_facecolor('lightgrey')

      plt.text(0.98, 0.95, f'{row_id},{col_id}', transform=ax.transAxes,
          ha='right', va='top', fontsize=8)
      plt.axvline(0, color='grey', lw=1)
      plt.axhline(0, color='grey', lw=1)
      plt.plot(t*1000, h, 'k')
      if row_id == col_id:
        plt.ylim(-40, 20)
      else:
        plt.ylim(-10, 10)
      plt.xlim(0)

    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      print('Save figure to:', file_path)
      plt.close()
    else:
      plt.show()


  def each_pair_all_trials_ccg(
      self,
      filter_membership,
      data_within='trial',
      verbose=False,
      file_dir=None):
    """Regression for all pairs."""
    spike_times = self.spike_times
    trial_window = self.trial_window
    group_ids = [0,1,2,3,4]
    model_list = []
    file_path = None

    spk_bin_width = 0.002
    lag_range = [-0.02, 0.07]
    jitter_window_width = 0.12
    distribution_type='mc_sim'
    num_jitter = 200
    trial_length = self.trial_window[1]

    if data_within == 'trial':
      if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()
      trange = enumerate(tqdm(filter_membership.columns, ncols=100, file=sys.stdout))
      for i, trial_id in trange:
        filter_membership_sub = filter_membership.loc[:,[trial_id]]
        for group_id in group_ids:
          spike_times_x, spike_times_y = self.stack_spike_times_by_pairs(
              spike_times, filter_membership_sub, group_id=group_id,
              verbose=False)

          if file_dir is not None:
            file_path = file_dir + f'trial{trial_id}_group{group_id}_within_trial.pdf'
          self.jittertool.cross_correlation_jitter(spike_times_x, spike_times_y,
              spk_bin_width, trial_length, lag_range, jitter_window_width,
              distribution_type, num_jitter, 
              ci_alpha=0.05, verbose=1,file_path=file_path)

    elif data_within == 'neuron':
      if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()
      trange = enumerate(tqdm(filter_membership.index, ncols=100, file=sys.stdout))
      for i, (neuron_x, neuron_y) in trange:
        filter_membership_sub = filter_membership.loc[[(neuron_x,neuron_y)],:]
        for group_id in group_ids:
          spike_times_x, spike_times_y = self.stack_spike_times_by_pairs(
              spike_times, filter_membership_sub, group_id=group_id,
              verbose=False)

          if file_dir is not None:
            file_path = file_dir + f'trial{trial_id}_group{group_id}_within_neuron.pdf'
          self.jittertool.cross_correlation_jitter(spike_times_x, spike_times_y,
              spk_bin_width, trial_length, lag_range, jitter_window_width,
              distribution_type, num_jitter,
              ci_alpha=0.05, verbose=1,file_path=file_path)

    return model_list


  def estimate_nuisance_kernel_width_by_group(
      self,
      filter_membership,
      verbose=False):
    """Fit each cluster of filters.

    This treating all trials especially the baseline the same.

    Args:
      batch_training: the related parameter `batch_size` is determined by number
      of trials, not number of neurons.
    """
    spike_times = self.spike_times
    trial_window = self.trial_window
    check_stats = jitter.JitterTool.spike_times_statistics
    # print('filter_membership.shape', filter_membership.shape)

    batch_size = 1000
    model_par1 = {'dt': 0.001, 'trial_window': trial_window,
        'append_nuisance': ['const', 'gaussian_kernel']}
    model_par2 = {'dt': 0.001, 'trial_window': trial_window,
        'append_nuisance': ['const', 'gaussian_kernel']}

    spike_times_x, spike_times_y = self.stack_spike_times_by_pairs(
        spike_times, filter_membership, group_id=0, batch_size=batch_size,
        verbose=True)
    # check_stats(spike_times_x, trial_window[1], verbose=0)
    # check_stats(spike_times_y, trial_window[1], verbose=0)
    jitter.JitterTool.estimate_optimal_jitter_window_width(
        spike_times_x, spike_times_y, model_par1)
    jitter.JitterTool.estimate_optimal_jitter_window_width(
        spike_times_x, spike_times_y, model_par2)

    spike_times_x, spike_times_y = self.stack_spike_times_by_pairs(
        spike_times, filter_membership, group_id=1, batch_size=batch_size,
        verbose=verbose)
    jitter.JitterTool.estimate_optimal_jitter_window_width(
        spike_times_x, spike_times_y, model_par1)
    jitter.JitterTool.estimate_optimal_jitter_window_width(
        spike_times_x, spike_times_y, model_par2)

    spike_times_x, spike_times_y = self.stack_spike_times_by_pairs(
        spike_times, filter_membership, group_id=2, batch_size=batch_size,
        verbose=verbose)
    jitter.JitterTool.estimate_optimal_jitter_window_width(
        spike_times_x, spike_times_y, model_par1)
    jitter.JitterTool.estimate_optimal_jitter_window_width(
        spike_times_x, spike_times_y, model_par2)

    spike_times_x, spike_times_y = self.stack_spike_times_by_pairs(
        spike_times, filter_membership, group_id=3, batch_size=batch_size,
        verbose=verbose)
    jitter.JitterTool.estimate_optimal_jitter_window_width(
        spike_times_x, spike_times_y, model_par1)
    jitter.JitterTool.estimate_optimal_jitter_window_width(
        spike_times_x, spike_times_y, model_par2)

    spike_times_x, spike_times_y = self.stack_spike_times_by_pairs(
        spike_times, filter_membership, group_id=4, batch_size=batch_size,
        verbose=verbose)
    jitter.JitterTool.estimate_optimal_jitter_window_width(
        spike_times_x, spike_times_y, model_par1)
    jitter.JitterTool.estimate_optimal_jitter_window_width(
        spike_times_x, spike_times_y, model_par2)


  def estimate_nuisance_kernel_width_together(
      self,
      filter_membership,
      kernel_width_grid=None,
      batch_size=5000,
      random_seed=None,
      verbose=False):
    """Fit each cluster of filters.

    This treating all trials especially the baseline the same.

    Args:
      batch_size: the related parameter `batch_size` is determined by number
      of trials, not number of neurons.
    """
    spike_times = self.spike_times
    trial_window = self.trial_window
    check_stats = jitter.JitterTool.spike_times_statistics
    # print('filter_membership.shape', filter_membership.shape)

    spike_times_x, spike_times_y = self.stack_spike_times_by_pairs_all(
        spike_times, filter_membership, batch_size=batch_size,
        random_seed=random_seed, verbose=False)
    # check_stats(spike_times_x, trial_window[1], verbose=0)
    # check_stats(spike_times_y, trial_window[1], verbose=0)

    model_par = {'dt': 0.001, 'trial_window': trial_window,
        'append_nuisance': ['const', 'gaussian_kernel']}
    return jitter.JitterTool.estimate_optimal_jitter_window_width(
        spike_times_x, spike_times_y, model_par, kernel_width_grid=kernel_width_grid)

    # model_par = {'dt': 0.001, 'trial_window': trial_window,
    #     'append_nuisance': ['const', 'triangle_kernel']}
    # return jitter.JitterTool.estimate_optimal_jitter_window_width(
    #     spike_times_x, spike_times_y, model_par, kernel_width_grid=kernel_width_grid)


  def update_cluster_filter_bspline(
      self,
      filter_membership,
      batch_training=False,
      verbose=False):
    """Fit each cluster of filters.

    `batch_size` is determined by number of trials, not number of neurons.
    """
    spike_times = self.spike_times
    trial_window = self.trial_window

    kernel_width = 0.12
    model_pars = {}

    if batch_training:
      batch_size = 3000
      max_num_itrs = 8
      epsilon = 0.05
      lr=0.9
      fit_func = self.jittertool.bivariate_continuous_time_coupling_filter_regression_batch
    else:
      batch_size = None
      max_num_itrs = 60
      epsilon = 1e-5
      lr=0.9
      fit_func = self.jittertool.bivariate_continuous_time_coupling_filter_regression

    for model_id in [0,1,2,3,4]:
      if model_id in [0,1,2]:
        filter_length = 0.05
      elif model_id in [3,4]:
        filter_length = 0.04

      # model_par = {'filter_type': 'bspline', 'num_knots': 8, 'knot_space_par': 0.05,
      #     'num_tail_drop': 1, 'filter_length': filter_length,
      #     'append_nuisance': ['const', 'triangle_kernel'], 'kernel_width': kernel_width,
      #     'const_offset': 0, 'learning_rate': lr, 'random_seed':0,
      #     'batch_size':batch_size, 'max_num_itrs': max_num_itrs, 'epsilon': epsilon,
      #     'model_id': model_id}
      model_par = {'filter_type': 'bspline', 'num_knots': 8, 'knot_space_par': 0.05,
          'num_tail_drop': 1, 'filter_length': filter_length,
          'append_nuisance': ['const'],
          'const_offset': 0, 'learning_rate': lr, 'random_seed':0,
          'batch_size':batch_size, 'max_num_itrs': max_num_itrs, 'epsilon': epsilon,
          'model_id': model_id}
      spike_times_x, spike_times_y = self.stack_spike_times_by_pairs(
          spike_times, filter_membership, group_id=model_id, verbose=True)
      model_pars[model_id] = fit_func(
          spike_times_x, spike_times_y, trial_window, model_par, verbose=verbose)
      print(f'Finish {model_id}.')

    return model_pars


  def update_cluster_filter_jitter(
      self,
      filter_membership,
      batch_training=False,
      verbose=False):
    """Fit each cluster of filters.

    `batch_size` is determined by number of trials, not number of neurons.
    """
    spike_times = self.spike_times
    trial_window = self.trial_window
    print('filter_membership.shape', filter_membership.shape)

    jitter_window_width = 0.3
    num_jitter = 100
    model_pars = {}

    if batch_training:
      batch_size = 5000
      max_num_itrs = 8
      epsilon = 0.3
      lr=0.9
      fit_func = self.jittertool.bivariate_continuous_time_coupling_filter_regression_batch
    else:
      batch_size = None
      max_num_itrs = 50
      epsilon = 1e-5
      lr=0.9
      fit_func = self.jittertool.bivariate_continuous_time_coupling_filter_regression

    for model_id in [0,1,2,3,4]:
      model_par_square = {
          'filter_type': 'square', 'filter_length': 0.05,
          'append_nuisance': ['const'],
          'const_offset': 0, 'learning_rate': lr, 'random_seed':0,
          'batch_size':batch_size, 'max_num_itrs': max_num_itrs, 'epsilon': epsilon,
          'num_jitter': num_jitter, 'jitter_window_width': jitter_window_width,
          'model_id': model_id}
      model_par_bspline = {
          'filter_type': 'bspline', 'num_knots': 8, 'knot_space_par': 0.05,
          'num_tail_drop': 1, 'filter_length': 0.04,
          'append_nuisance': ['const'],
          'const_offset': 0, 'learning_rate': lr, 'random_seed':0,
          'batch_size':batch_size, 'max_num_itrs': max_num_itrs, 'epsilon': epsilon,
          'num_jitter': num_jitter, 'jitter_window_width': jitter_window_width,
          'model_id': model_id}
      if model_id in [0,1,2]:
        model_par = model_par_square
      elif model_id in [3,4]:
        model_par = model_par_bspline

      spike_times_x, spike_times_y = self.stack_spike_times_by_pairs(
          spike_times, filter_membership, group_id=model_id, verbose=True)
      (model_par_raw, model_par_jitter
          ) = self.jittertool.bivariate_continuous_time_coupling_filter_regression_jitter(
          spike_times_x, spike_times_y, trial_window, model_par, verbose=True)
      model_pars[model_id] = (model_par_raw, model_par_jitter)
      if verbose:
        self.jittertool.plot_continuous_time_bivariate_regression_jitter_model_par(
            model_par_raw, model_par_jitter, ylim=[-7, 22])
      print(f'Finish {model_id}.')

    return model_pars


  def plot_multiple_filters(
      self,
      model_pars,
      model_pars_ref=None,
      num_rows_cols=(2,4),
      ylim=None,
      file_path=None):
    """Plot `model_pars` all together.

    Args:
      model_pars_ref: a list of reference models.
    """
    num_rows, num_cols = num_rows_cols
    CI_scale = scipy.stats.norm.ppf(0.975)  # 95% CI.

    gs_kw = dict(width_ratios=[1]*num_cols, height_ratios=[1]*num_rows)
    fig, axs = plt.subplots(figsize=(num_cols*3.5, num_rows*2), gridspec_kw=gs_kw,
        nrows=num_rows, ncols=num_cols)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0.1)
    axs = axs.reshape(-1)
    for ax in axs:
      ax.tick_params(left=True, labelleft=False, labelbottom=True, bottom=True,
                     direction='in')
    for fig_id, model_id in enumerate(model_pars):
      model_par = model_pars[model_id]

      if fig_id >= num_cols * num_rows:
        break
      filter_t, h, h_std = self.jittertool.reconstruct_basis(model_par)

      ax = fig.add_subplot(axs[fig_id])
      ax.tick_params(left=True, labelleft=False, labelbottom=True, bottom=True,
                     direction='in')
      plt.axvline(x=0, color='lightgrey', lw=1)
      plt.axhline(y=0, color='lightgrey', lw=1)

      if model_pars_ref is not None:
        t_ref, h_ref, h_std_ref = self.jittertool.reconstruct_basis(
            model_pars_ref[model_id])
        plt.plot(t_ref*1000, h_ref, 'g', lw=2, label='Filter reference')

      plt.plot(filter_t*1000, h, 'k', lw=2)
      # plt.fill_between(filter_t, h+h_std*CI_scale, h-h_std*CI_scale,
      #                  facecolor='lightgrey', alpha=0.3, label='95% CI')
      plt.ylim(ylim)
      plt.xlim(0, 60)
      # plt.text(0.85, 0.85, f'id={model_id}', transform=ax.transAxes)
      if fig_id == num_cols * (num_rows - 1):
        ax.tick_params(left=True, labelleft=True, labelbottom=True, bottom=True)
        plt.xlabel('Time [ms]')
        plt.ylabel('Firing rate [spikes/sec]')
      # if fig_id == 0:
      #   plt.legend(loc='upper left')

    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      print('Save figure to:', file_path)
    plt.show()


  def update_cluster_filter_joint_trail(
      self,
      filter_membership,
      kernel_width=0.06,
      batch_training=False,
      verbose=False):
    """Fit each cluster of filters.

    This treating all trials especially the baseline the same.

    Args:
      batch_training: the related parameter `batch_size` is determined by number
      of trials, not number of neurons.
    """
    spike_times = self.spike_times
    trial_window = self.trial_window
    print('filter_membership.shape', filter_membership.shape)

    filter_length = 0.05  # 0.05
    # append_nuisance = ['const', 'triangle_kernel']  # Old before Nov 2021
    # kernel_width = 0.16  # Old before Nov 2021
    append_nuisance = ['const', 'gaussian_kernel']
    # kernel_width = 0.06

    random_seed = 0
    model_pars = {}

    if batch_training:
      batch_size = 3000
      max_num_itrs = 8
      epsilon = 0.05
      lr=0.9
      fit_func = self.jittertool.bivariate_continuous_time_coupling_filter_regression_batch
    else:
      batch_size = None
      max_num_itrs = 60
      epsilon = 1e-5
      lr=0.5
      fit_func = self.jittertool.bivariate_continuous_time_coupling_filter_regression

    # Cluster 0: zero constant. No need to fit.
    model_id = 0
    model_pars[model_id] = {'filter_type': 'none',
        'append_nuisance': append_nuisance, 'kernel_width': kernel_width,
        'num_basis': 2, 'const_offset': 0, 'learning_rate': lr,
        'max_num_itrs': max_num_itrs, 'epsilon': epsilon, 'model_id': model_id}

    # Cluster 1: positive constant.
    model_id = 1
    model_par = {'filter_type': 'square', 'filter_length': filter_length,
        'append_nuisance': append_nuisance, 'kernel_width': kernel_width,
        'const_offset': 0, 'learning_rate': lr, 'random_seed':random_seed,
        'batch_size':batch_size, 'max_num_itrs': max_num_itrs, 'epsilon': epsilon,
        'model_id': model_id}
    spike_times_x, spike_times_y = self.stack_spike_times_by_pairs(
        spike_times, filter_membership, group_id=model_id, verbose=verbose)
    model_pars[model_id] = fit_func(
        spike_times_x, spike_times_y, trial_window, model_par, verbose=verbose)
    if verbose:
      print('Finish 1.')

    # Cluster 2: negative constant.
    model_id = 2
    model_par = {'filter_type': 'square', 'filter_length': filter_length,
        'append_nuisance': append_nuisance, 'kernel_width': kernel_width,
        'const_offset': 0, 'learning_rate': lr, 'random_seed':random_seed,
        'batch_size':batch_size, 'max_num_itrs': max_num_itrs, 'epsilon': epsilon,
        'model_id': model_id}
    spike_times_x, spike_times_y = self.stack_spike_times_by_pairs(
        spike_times, filter_membership, group_id=model_id, verbose=verbose)
    model_pars[model_id] = fit_func(
        spike_times_x, spike_times_y, trial_window, model_par, verbose=verbose)
    if verbose:
      print('Finish 2.')

    # Cluster 3: bump head.
    model_id = 3
    model_par = {'filter_type': 'bspline', 'num_knots': 8, 'knot_space_par': 0.05,
        'num_tail_drop': 1, 'filter_length': 0.04,
        'append_nuisance': append_nuisance, 'kernel_width': kernel_width,
        'const_offset': 0, 'learning_rate': lr, 'random_seed':random_seed,
        'batch_size':batch_size, 'max_num_itrs': max_num_itrs, 'epsilon': epsilon,
        'model_id': model_id}
    spike_times_x, spike_times_y = self.stack_spike_times_by_pairs(
        spike_times, filter_membership, group_id=model_id, verbose=verbose)
    model_pars[model_id] = fit_func(
        spike_times_x, spike_times_y, trial_window, model_par, verbose=verbose)
    if verbose:
      print('Finish 3.')

    # Cluster 4: sharp head.
    model_id = 4
    model_par = {'filter_type': 'bspline', 'num_knots': 8, 'knot_space_par': 0.05,
        'num_tail_drop': 1, 'filter_length': 0.04,
        'append_nuisance': append_nuisance, 'kernel_width': kernel_width,
        'const_offset': 0, 'learning_rate': lr, 'random_seed':random_seed,
        'batch_size':batch_size, 'max_num_itrs': max_num_itrs, 'epsilon': epsilon,
        'model_id': model_id}
    spike_times_x, spike_times_y = self.stack_spike_times_by_pairs(
        spike_times, filter_membership, group_id=model_id, verbose=verbose)
    model_pars[model_id] = fit_func(
        spike_times_x, spike_times_y, trial_window, model_par, verbose=verbose)
    if verbose:
      print('Finish 4.')

    return model_pars


  # TODO: Fit all trials. But each trial has separate nuisance coefficients.
  def update_cluster_filter_trialwise_nuisance(
      self,
      filter_membership,
      batch_training=False,
      verbose=False):
    """Fit each cluster of filters with trial-wise nuisance.

    This treating all trials with different baseline. It fit the model by
    alternating between fitting baselines and jointly fitting filters.

    `batch_size` is determined by number of trials, not number of neurons.
    """
    pass
  #   spike_times = self.spike_times
  #   trial_window = self.trial_window

  #   filter_length = 0.03
  #   append_nuisance = ['block_const', 'block_triangle_kernel']
  #   kernel_width = 0.12  # 0.12
  #   random_seed = 0
  #   model_pars = {}

  #   if batch_training:
  #     batch_size = 3000
  #     max_num_itrs = 8
  #     epsilon = 0.05
  #     lr=0.9
  #     # fit_func = self.jittertool.bivariate_continuous_time_coupling_filter_regression_batch
  #   else:
  #     batch_size = None
  #     max_num_itrs = 100
  #     epsilon = 1e-5
  #     lr=0.8
  #     fit_func = self.jittertool.bivariate_continuous_time_coupling_filter_regression_block

  #   # Cluster 0: zero constant. No need to fit.
  #   model_id = 0
  #   model_pars[model_id] = {'filter_type': 'none',
  #       'append_nuisance': append_nuisance, 'kernel_width': kernel_width,
  #       'num_basis': 2, 'const_offset': 0, 'learning_rate': lr,
  #       'max_num_itrs': max_num_itrs, 'epsilon': epsilon, 'model_id': model_id}

  #   # Cluster 1: positive constant.
  #   model_id = 1
  #   model_par = {'filter_type': 'square', 'filter_length': filter_length,
  #       'append_nuisance': append_nuisance, 'kernel_width': kernel_width,
  #       'const_offset': 0, 'learning_rate': lr, 'random_seed':random_seed,
  #       'batch_size':batch_size, 'max_num_itrs': max_num_itrs, 'epsilon': epsilon,
  #       'model_id': model_id}
  #   spike_times_x, spike_times_y = self.stack_spike_times_by_pairs(
  #       spike_times, filter_membership, group_id=model_id, verbose=True)

  #   model_pars[model_id] = fit_func(
  #       spike_times_x, spike_times_y, trial_window, model_par, verbose=verbose)
  #   print('Finish 1.')

  #   return model_pars


  @classmethod
  def update_membership_portion(
      cls,
      num_groups=5,
      filter_membership=None):
    """Update membership portion."""
    num_filters = filter_membership.size
    alpha = np.zeros(num_groups) + num_filters
    alpha[0] = num_filters * 1.1
    group_cnt = filter_membership.apply(pd.value_counts, sort=False)
    group_cnt = group_cnt.sum(axis=1).values
    group_portion = group_cnt + alpha
    group_portion = group_portion / group_portion.sum()

    print('Group portion:', group_portion)
    return group_portion


  @classmethod
  def update_filter_membership(
      cls,
      spike_times,
      trial_window,
      group_portion,
      group_model_pars,
      filter_membership,
      trial_ids=None,
      kernel_width=0.06,
      update_type='fit',
      parallel=False,
      verbose=False):
    """Fit the membership of each filter.

    This is a classmethod for parallel programming purpose. A realized object
    can not be parallelized by joblib properly.

    Args:
      trial_ids: the function will fit all `trial_ids` together, not individually.
    """
    group_ids = group_model_pars.keys()
    trial_ids = spike_times.columns.values if trial_ids is None else trial_ids

    filter_membership = filter_membership.copy()
    group_model_pars = group_model_pars.copy()
    for model_id in group_model_pars:
      model_par = group_model_pars[model_id].copy()
      model_par['fix_filter'] = True
      if 'beta' in model_par:
        model_par['beta_fix'] = model_par['beta'].copy()
      if 'beta_init' in model_par:
        del model_par['beta_init']
      model_par['max_num_itrs'] = 20
      model_par['epsilon'] = 1e-5
      model_par['learning_rate'] = 0.5
      # model_par['kernel_width'] = 0.16  # Old. triangular_kernel
      model_par['append_nuisance'] = ['const', 'gaussian_kernel']
      model_par['kernel_width'] = kernel_width  # selected by plug-in estimator.
      group_model_pars[model_id] = model_par

    if hasattr(tqdm, '_instances'):
      tqdm._instances.clear()
    trange = (enumerate(tqdm(filter_membership.index, ncols=100)) if verbose==1
              else enumerate(filter_membership.index))
    for i, (neuron_x, neuron_y) in trange:
      # Random sample here if needed.
      spike_times_x = spike_times.loc[neuron_x,trial_ids].tolist()
      spike_times_y = spike_times.loc[neuron_y,trial_ids].tolist()
      num_spikes_x = [len(spikes) for spikes in spike_times_x]
      num_spikes_x = np.sum(num_spikes_x)
      num_spikes_y = [len(spikes) for spikes in spike_times_y]
      num_spikes_y = np.sum(num_spikes_y)
      # jitter.JitterTool.spike_times_statistics(spike_times_x, trial_window[1], verbose=1)
      # jitter.JitterTool.spike_times_statistics(spike_times_y, trial_window[1], verbose=1)

      neg_log_posts = []
      for group_id in group_ids:
        # If there's no spike in this trial, then no need to check the type.
        if num_spikes_x == 0 or num_spikes_y == 0:
          break
        model_tmp = jitter.JitterTool.bivariate_continuous_time_coupling_filter_regression(
            spike_times_x, spike_times_y, trial_window, group_model_pars[group_id],
            mute_warning=True, verbose=False)
        nll = model_tmp['nll']
        # num_pars = group_model_pars[group_id]['num_basis']
        # BIC = 2*nll + np.log(num_spikes_y)*num_pars (error: free parameters
        # number is the same.)
        # The degree of freedom for each model is the same, i.e. baseline +
        # nuisance variable. So BIC is equiv to nll.
        neg_log_posterior = nll - np.log(group_portion[group_id])
        neg_log_posts.append((group_id, neg_log_posterior))

      if update_type == 'fit' and num_spikes_x != 0 and num_spikes_y != 0:
        new_group_id = min(neg_log_posts, key = lambda x: x[1])[0]
      elif update_type == 'sample' and num_spikes_x != 0 and num_spikes_y != 0:
        array = np.array([val for _,val in neg_log_posts])
        prob = np.exp(- array - np.max(- array))
        prob = prob / prob.sum()
        new_group_id = np.random.choice(len(prob), size=1, p=prob)
      elif num_spikes_x == 0 or num_spikes_y == 0:
        # new_group_id = 0
        # new_group_id = np.random.randint(0, 5)
        new_group_id = np.nan

      if len(filter_membership.columns) > 1:
        filter_membership.loc[(neuron_x, neuron_y), trial_ids] = new_group_id
      else:
        filter_membership.loc[(neuron_x, neuron_y)] = new_group_id

      if verbose == 2:
        # print('new_group_id', new_group_id)
        print(neg_log_posts)
        plt.figure(figsize=[4,2])
        for (idx, val) in neg_log_posts:
          plt.plot(idx, val, 'x')
        # plt.ylim(-6700, -6500)
        plt.show()

    if parallel:
      return filter_membership.loc[:,trial_ids]
    else:
      return filter_membership


  @classmethod
  def update_filter_membership_multivariate(
      self,
      spike_times,
      trial_window,
      group_model_pars,
      neuron_drivers,
      nuisance_model_par,
      filter_membership,
      trial_ids=None,
      update_type='fit',
      parallel=False,
      verbose=False):
    """Fit the membership of each filter.

    This is a classmethod for parallel programming purpose. A realized object
    can not be parallelized by joblib properly.

    Args:
      trial_ids: the function will fit all `trial_ids` together, not individually.
      neuron_drivers: This lists the neurons that will be treated as the nuisacne
          variables making contributions to all pair regression.
    """
    group_ids = group_model_pars.keys()
    trial_ids = spike_times.columns.values if trial_ids is None else trial_ids
    num_drivers = len(neuron_drivers)

    filter_membership = filter_membership.copy()
    group_model_pars = group_model_pars.copy()
    for model_id in group_model_pars:
      model_par = group_model_pars[model_id].copy()
      model_par['fix_filter'] = True
      if 'beta' in model_par:
        # The beta_fix is a dict with neuron id as the key.
        model_par['beta_fix'] = model_par['beta'].copy()
      if 'beta_init' in model_par:
        del model_par['beta_init']
      # The const is needed.
      if 'const' not in model_par['append_nuisance']:
        model_par['append_nuisance'] = ['const'] + model_par['append_nuisance']
      model_par['max_num_itrs'] = 30
      model_par['epsilon'] = 1e-5
      model_par['learning_rate'] = 0.5
      model_par['kernel_width'] = 0.12
      group_model_pars[model_id] = model_par

    spike_times_driver = [0] * num_drivers
    for n, neuron_driver in enumerate(neuron_drivers):
      spike_times_driver[n] = spike_times.loc[neuron_driver,trial_ids].tolist()

    if hasattr(tqdm, '_instances'):
      tqdm._instances.clear()
    trange = (enumerate(tqdm(filter_membership.index, ncols=100)) if verbose==1
              else enumerate(filter_membership.index))
    for i, (neuron_x, neuron_y) in trange:
      # Random sample here if needed.
      spike_times_x = spike_times.loc[neuron_x,trial_ids].tolist()
      spike_times_y = spike_times.loc[neuron_y,trial_ids].tolist()
      num_spikes_y = [len(spikes) for spikes in spike_times_y]
      num_spikes_y = np.sum(num_spikes_y)
      # jitter.JitterTool.spike_times_statistics(spike_times_x, trial_window[1], verbose=1)
      # jitter.JitterTool.spike_times_statistics(spike_times_y, trial_window[1], verbose=1)

      BIC_list = []
      for group_id in group_ids:
        if num_spikes_y == 0:
          break
        model_tmp = jitter.JitterTool.multivariate_continuous_time_coupling_filter_regression(
            [spike_times_x] + spike_times_driver, spike_times_y, trial_window,
            [group_model_pars[group_id]] + [nuisance_model_par]*num_drivers,
            mute_warning=True, verbose=False)
        nll = model_tmp[0]['nll']

        # for item in model_tmp:
        #   print(item)
        #   print()

        # num_pars = group_model_pars[group_id]['num_basis']
        # BIC = 2*nll + np.log(num_spikes_y)*num_pars (error: free parameters
        # number is the same.)
        # The degree of freedom for each model is the same, i.e. baseline +
        # nuisance variable. So BIC is equiv to nll.
        BIC = nll
        BIC_list.append((group_id, BIC))
        # print('num_itrs', model_tmp['num_itrs'])

      if update_type == 'fit' and num_spikes_y != 0:
        new_group_id = min(BIC_list, key = lambda x: x[1])[0]
      elif update_type == 'sample' and num_spikes_y != 0:
        BIC_arr = np.array([val for _,val in BIC_list]) / 2
        prob = np.exp(- BIC_arr - np.max(- BIC_arr))
        prob = prob / prob.sum()
        new_group_id = np.random.choice(len(prob), size=1, p=prob)
      elif num_spikes_y == 0:
        new_group_id = 0

      if len(filter_membership.columns) > 1:
        filter_membership.loc[(neuron_x, neuron_y), trial_ids] = new_group_id
      else:
        filter_membership.loc[(neuron_x, neuron_y)] = new_group_id

      if verbose == 2 and i == -1:
        # print('new_group_id', new_group_id)
        print(BIC_list)
        print(prob)
        plt.figure(figsize=[4,2])
        for (idx, val) in BIC_list:
          plt.plot(idx, val, 'x')
        # plt.ylim(-6700, -6500)
        plt.show()

    if parallel:
      return filter_membership.loc[:,trial_ids]
    else:
      return filter_membership


  @classmethod
  def ks_test_models_list(
      cls,
      spike_times,
      trial_window,
      model_list,
      trial_ids=None,
      verbose=False):
    """KS test for each model."""
    trial_ids = spike_times.columns.values if trial_ids is None else trial_ids

    u_list_dict = {}
    if hasattr(tqdm, '_instances'):
      tqdm._instances.clear()
    trange = (enumerate(tqdm(model_list, ncols=100)) if verbose>0
              else enumerate(model_list))
    for i, ((neuron_x, neuron_y), model_par) in trange:
      u_list = []
      for r, trial_id in enumerate(trial_ids):
        spike_times_x = spike_times.loc[neuron_x,[trial_id]].tolist()
        spike_times_y = spike_times.loc[neuron_y,[trial_id]].tolist()
        num_spikes_y = [len(spikes) for spikes in spike_times_y]
        num_spikes_y = np.sum(num_spikes_y)
        num_spikes_x = [len(spikes) for spikes in spike_times_x]
        num_spikes_x = np.sum(num_spikes_x)
        # jitter.JitterTool.spike_times_statistics(spike_times_x, trial_window[1], verbose=1)
        # jitter.JitterTool.spike_times_statistics(spike_times_y, trial_window[1], verbose=1)

        # If the trial does not have a lot spikes, skip the verification.
        if num_spikes_y <= 5 or num_spikes_x <= 5:
          continue

        u_vals = jitter.JitterTool.ks_test(
            spike_times_x, spike_times_y, trial_window, model_par,
            dt=0.001, test_size=0.01, verbose=False)
        u_list.append(u_vals)

      u_list = np.hstack(u_list)
      u_list = u_list[(u_list>=0) & (u_list<=1)]
      u_list_dict[(neuron_x, neuron_y)] = u_list

      if verbose == 3:
        CI_trap, mcdf, ecdf, CI_up, CI_dn = jitter.JitterTool.check_ks(u_list,
            test_size=0.01, bin_width=0.02, verbose=True)

    return u_list_dict


  @classmethod
  def ks_test_group_wise(
      cls,
      spike_times,
      trial_window,
      group_model_pars,
      filter_membership,
      trial_ids=None,
      parallel=False,
      verbose=False):
    """Fit the membership of each filter.

    This is a classmethod for parallel programming purpose. A realized object
    can not be parallelized by joblib properly.

    Args:
      trial_ids: the function will fit all `trial_ids` together, not individually.
    """
    num_trials = len(trial_ids)
    group_ids = group_model_pars.keys()
    trial_ids = spike_times.columns.values if trial_ids is None else trial_ids

    filter_membership = filter_membership.copy()
    group_model_pars = group_model_pars.copy()
    for model_id in group_model_pars:
      model_par = group_model_pars[model_id].copy()
      model_par['fix_filter'] = True
      if 'beta' in model_par:
        model_par['beta_fix'] = model_par['beta'].copy()
      if 'beta_init' in model_par:
        del model_par['beta_init']
      model_par['max_num_itrs'] = 20
      model_par['epsilon'] = 1e-5
      model_par['learning_rate'] = 0.3
      # model_par['kernel_width'] = 0.16  # Old. triangular_kernel
      model_par['append_nuisance'] = ['const', 'gaussian_kernel']
      model_par['kernel_width'] = 0.06  # selected by plog-in estimator.
      group_model_pars[model_id] = model_par

    u_list_dict = {}
    if hasattr(tqdm, '_instances'):
      tqdm._instances.clear()
    # trange = (enumerate(tqdm(filter_membership.index, ncols=100)) if verbose==1
    #           else enumerate(filter_membership.index))
    trange = enumerate(filter_membership.index)
    for i, (neuron_x, neuron_y) in trange:
      # if i != 69:
      #   continue
      if neuron_x % 3 == 0 and neuron_y % 3 == 0:
        print(i, neuron_x, neuron_y)
      else:
        continue

      u_list = []
      for r, trial_id in enumerate(trial_ids):
        spike_times_x = spike_times.loc[neuron_x,[trial_id]].tolist()
        spike_times_y = spike_times.loc[neuron_y,[trial_id]].tolist()
        num_spikes_y = [len(spikes) for spikes in spike_times_y]
        num_spikes_y = np.sum(num_spikes_y)
        num_spikes_x = [len(spikes) for spikes in spike_times_x]
        num_spikes_x = np.sum(num_spikes_x)
        # jitter.JitterTool.spike_times_statistics(spike_times_x, trial_window[1], verbose=1)
        # jitter.JitterTool.spike_times_statistics(spike_times_y, trial_window[1], verbose=1)
        group_id = filter_membership.loc[(neuron_x, neuron_y),[trial_id]].values[0]

        # If the trial does not have a lot spikes, skip the verification.
        if num_spikes_y <= 5 or num_spikes_x <= 5:
          continue

        # group_model_pars[group_id] = {
        #     'filter_type': 'bspline', 'num_knots': 6, 'knot_space_par': 0.1,
        #     'filter_length': 0.06, 'num_tail_drop': 1,
        #     'append_nuisance': ['const', 'gaussian_kernel'], 'kernel_width': 0.06,
        #     'const_offset': 0, 'learning_rate': 0.2, 'max_num_itrs': 20,
        #     'epsilon': 1e-5}
        # group_model_pars[group_id] = {
        #     'filter_type': 'none',
        #     'append_nuisance': ['const', 'gaussian_kernel'], 'kernel_width': 0.06,
        #     'const_offset': 0, 'learning_rate': 0.9, 'max_num_itrs': 100,
        #     'epsilon': 1e-5}
        # group_model_pars[group_id] = {
        #     'filter_type': 'square', 'filter_length': 0.05,
        #     'append_nuisance': ['const', 'gaussian_kernel'], 'kernel_width': 0.06,
        #     'const_offset': 0, 'learning_rate': 0.9, 'max_num_itrs': 100,
        #     'epsilon': 1e-5}
        model_par_hat = jitter.JitterTool.bivariate_continuous_time_coupling_filter_regression(
            spike_times_x, spike_times_y, trial_window, group_model_pars[group_id],
            mute_warning=True, verbose=False)

        # Since the filter is set as fixed, the coefficients for the filter
        # need to be added.
        model_par_test = model_par_hat.copy()
        if 'beta_fix' in group_model_pars[group_id]:
          beta_filter_ind = len(model_par_test['append_nuisance'])
          beta_filter = group_model_pars[group_id]['beta_fix'][beta_filter_ind:,:]
          model_par_test['beta'] = np.vstack([model_par_test['beta'], beta_filter])
          model_par_test.pop('beta_hessian')

        u_vals = jitter.JitterTool.ks_test(
            spike_times_x, spike_times_y, trial_window, model_par_test,
            dt=0.001, test_size=0.05, verbose=False)
        u_list.append(u_vals)

      u_list = np.hstack(u_list)
      u_list = u_list[(u_list>=0) & (u_list<=1)]
      u_list_dict[(neuron_x, neuron_y)] = u_list
      CI_trap, mcdf, ecdf, CI_up, CI_dn = jitter.JitterTool.check_ks(u_list,
          test_size=0.01, bin_width=0.02, verbose=True)
    return u_list_dict

  @classmethod
  def plot_ks_u_list_dict(
      cls,
      u_list_dict,
      bin_width=0.01,
      test_size=0.01,
      file_path=None):
    """Plots the KS plot together."""
    num_ks = len(u_list_dict)
    bins = np.linspace(0, 1, int(1 / bin_width) + 1)
    c_alpha = np.sqrt(-np.log(test_size / 2) / 2)

    num_cols = 6
    num_rows = np.ceil(num_ks / num_cols).astype(int)
    gs_kw = dict(width_ratios=[1]*num_cols, height_ratios=[1]*num_rows)
    fig, axs = plt.subplots(figsize=(15, 2*num_rows), gridspec_kw=gs_kw,
        nrows=num_rows, ncols=num_cols)
    plt.subplots_adjust(left=None, right=None, hspace=0.1, wspace=0.1)
    axs = axs.reshape(-1)
    for i, (neuron_x, neuron_y) in enumerate(u_list_dict.keys()):
      u_vals = u_list_dict[(neuron_x, neuron_y)]
      num_samples = len(u_vals)
      ax = fig.add_subplot(axs[i])
      ax.tick_params(labelleft=False, labelbottom=False)
      if i == num_cols * (num_rows-1):
        ax.tick_params(labelleft=True, labelbottom=True)
        plt.xlabel('Theoretical quantile')
        plt.ylabel('Empirical quantile')

      epdf, bin_edges = np.histogram(u_vals, bins=bins, density=True)
      mcdf = bin_edges[1:]
      ecdf = np.cumsum(epdf) * bin_width
      CI_up = mcdf + c_alpha/np.sqrt(num_samples)
      CI_dn = mcdf - c_alpha/np.sqrt(num_samples)
      plt.plot(mcdf, ecdf, 'k')
      plt.plot(mcdf, CI_up, '--', color='lightgrey')
      plt.plot(mcdf, CI_dn, '--', color='lightgrey')
      plt.text(0.02, 0.75, rf'{neuron_x}$\to$'+f'\n{neuron_y}', transform=ax.transAxes)
      plt.axis([0,1,0,1])
      plt.xticks([0, 1], [0, 1])
      plt.yticks([0, 1], [0, 1])
      # plt.grid('on')
    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      print('save figure:', file_path)
    plt.show()


  def load_variable_list_across_trials(
      self,
      file_dir,
      prefix,
      trial_ids,
      max_itr,
      verbose=False):
    """Load a list of model."""
    num_trials = len(trial_ids)
    var_list = [0] * num_trials
    for i, trial_id in enumerate(trial_ids):
      keywords = (f'{prefix}*_trial{trial_id}_itr{max_itr}_*')
      files = glob(os.path.join(file_dir, keywords))
      var = util.load_variable(files[0], verbose=verbose)
      var_list[i] = (trial_id, var)

    return var_list


  def load_variable_list_across_iterations(
      self,
      file_dir,
      prefix,
      itrs,
      verbose=False):
    """Load a list of model."""
    num_itrs = len(itrs)
    var_list = [0] * num_itrs
    for i, itr in enumerate(itrs):
      keywords = (f'{prefix}*_itr{itr}_*')
      files = glob(os.path.join(file_dir, keywords))
      var = util.load_variable(files[0], verbose=verbose)
      var_list[i] = (itr, var)

    return var_list


  def plot_filter_membership_sample_trace(
      self,
      file_dir,
      prefix,
      itrs):
    """Plot trained data iterations."""
    num_itrs = len(itrs)
    var_list = []
    group_data = []
    select_itrs = []

    for i, itr in enumerate(itrs):
      keywords = (f'{prefix}*_itr{itr}_*')
      files = glob(os.path.join(file_dir, keywords))
      if len(files) == 0:
        continue
      select_itrs.append(itr)
      data = util.load_variable(files[0], verbose=False)
      if isinstance(data, pd.DataFrame):
        ind_cnt = self.filter_membership_statistics(data, verbose=False)
        group_data.append([cnt for ind, cnt in ind_cnt])
      elif isinstance(data, np.ndarray):
        group_data.append(data)

    group_data = np.array(group_data)
    num_groups = group_data.shape[1]
    plt.figure(figsize=[5, 2])
    for g in range(num_groups):
      plt.plot(select_itrs, group_data[:,g], '.-', lw=0.2, label=f'g {g}')
    plt.ylim(0)
    plt.legend(ncol=1, loc=(1.1, 0))
    plt.xlabel('Iteration')
    plt.show()


  def parse_model_pars_list(
      self,
      model_pars_list,
      model_id):
    """Compact a list of model parameters."""
    num_trials = len(model_pars_list)
    beta_mat = [0] * num_trials

    for i, (trial_id, model_pars) in enumerate(model_pars_list):
      beta_mat[i] = model_pars[model_id]['beta'].T
    beta_mat = np.vstack(beta_mat)
    return beta_mat


  def plot_trial_to_trial_betas(
      self,
      model_pars_list_0,
      model_pars_list_1):
    """Plot trial-to-trial dots of betas."""
    model_ids = [1,2,3,4]
    beta_ids = [[2], [2], [2,3,4,5], [2,3,4,5]]

    for m, model_id in enumerate(model_ids):
      beta_mat0 = self.parse_model_pars_list(model_pars_list_0, model_id=model_id)
      beta_mat1 = self.parse_model_pars_list(model_pars_list_1, model_id=model_id)
      num_cols = len(beta_ids[m])

      fig, axs = plt.subplots(figsize=(2.5*num_cols, 2.5), nrows=1, ncols=num_cols)
      plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0.05)
      axs = [axs] if num_cols == 1 else axs
      for c in range(num_cols):
        ax = fig.add_subplot(axs[c])
        ax.tick_params(left=True, labelleft=False, labelbottom=True, bottom=True,
                       direction='in')
        beta0 = beta_mat0[:,beta_ids[m][c]]
        beta1 = beta_mat1[:,beta_ids[m][c]]
        corr, p_val = scipy.stats.pearsonr(beta0, beta1)
        vmin = min(np.min(beta0), np.min(beta1))
        vmax = max(np.max(beta0), np.max(beta1))
        vrange = vmax - vmin
        vmin -= vrange*0.2
        vmax += vrange*0.2

        plt.plot(beta0, beta1, 'k.')
        plt.plot([-100, 100], [-100, 100], c='lightgrey', lw=0.4)
        ax.axis([vmin, vmax, vmin, vmax])
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        plt.text(0.05, 0.9, f'r={corr:.2f} p={p_val:.2e}', transform=ax.transAxes)

        if c == 0:
          plt.title(f'group={model_ids[m]}')
          plt.xlabel('V1\u27F6LM model coefficient')
          plt.ylabel('LM\u27F6AL model coefficient')

    plt.show()


  def parse_filter_membership_list(
      self,
      filter_membership_list,
      build_graph=True):
    """Compact a list of model parameters."""
    num_trials = len(filter_membership_list)
    graph_list = [0] * num_trials
    cluster_cnt_list = [0] * num_trials
    adjvec_list = [0] * num_trials

    for i, (trial_id, filter_membership) in enumerate(filter_membership_list):
      if build_graph:
        graph_list[i] = self.build_neuron_graph(filter_membership)
      cluster_cnt = filter_membership.value_counts(sort=False)
      cluster_cnt_list[i] = [cluster_cnt[idx] for idx in cluster_cnt.index]
      adjvec_list[i] = filter_membership.values

    cluster_cnt_mat = np.array(cluster_cnt_list)
    adjvec_mat = np.hstack(adjvec_list)
    return graph_list, cluster_cnt_mat, adjvec_mat


  def plot_trial_to_trial_filter_memberships_cnt(
      self,
      filter_membership_list_0,
      filter_membership_list_1,
      file_path=None):
    """Plot trial-to-trial memberships."""
    num_groups = 5

    if isinstance(filter_membership_list_0, list):
      num_trials = len(filter_membership_list_0)
      print('num_trials', num_trials)
      _, cluster_cnt_mat0, _ = self.parse_filter_membership_list(
          filter_membership_list_0, build_graph=False)
      _, cluster_cnt_mat1, _ = self.parse_filter_membership_list(
          filter_membership_list_1, build_graph=False)
    elif isinstance(filter_membership_list_0, pd.core.frame.DataFrame):
      num_trials = filter_membership_list_0.shape[1]
      cluster_cnt_mat0 = filter_membership_list_0.apply(pd.value_counts, sort=False)
      cluster_cnt_mat0 = cluster_cnt_mat0.values.T
      cluster_cnt_mat1 = filter_membership_list_1.apply(pd.value_counts, sort=False)
      cluster_cnt_mat1 = cluster_cnt_mat1.values.T

    gs_kw = dict(width_ratios=[1]*num_groups, height_ratios=[1])
    fig, axs = plt.subplots(figsize=(3.2*num_groups, 2.7), gridspec_kw=gs_kw,
        nrows=1, ncols=num_groups)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0.2)
    for group_id in range(num_groups):
      cluster_cnt0 = cluster_cnt_mat0[:,group_id]
      cluster_cnt1 = cluster_cnt_mat1[:,group_id]
      corr, p_val = scipy.stats.pearsonr(cluster_cnt0, cluster_cnt1)
      ax = fig.add_subplot(axs[group_id])
      # vmin = min(np.min(cluster_cnt0), np.min(cluster_cnt1))
      # vmax = max(np.max(cluster_cnt0), np.max(cluster_cnt1))
      # vrange = vmax - vmin
      # vmin -= vrange*0.2
      # vmax += vrange*0.2
      vmin, vmax = 0, 400

      ax.tick_params(left=True, labelleft=(group_id==0), bottom=True,
          labelbottom=True, direction='in')
      plt.plot([-100, 1000], [-100, 1000], c='lightgrey', lw=1)
      plt.plot(cluster_cnt0, cluster_cnt1, 'k.')
      plt.title(f'group {group_id}')
      ax.axis([vmin, vmax, vmin, vmax])
      plt.locator_params(axis='x', nbins=4)
      plt.locator_params(axis='y', nbins=4)
      plt.text(0.05, 0.9, f'r={corr:.2f} p={p_val:.2e}', transform=ax.transAxes)
      if group_id == 0:
        plt.xlabel('V1-LM pair counts')
        plt.ylabel('LM-AL pair counts')

    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      print('Save figure to:', file_path)
    plt.show()


  def plot_trial_to_trial_filter_memberships_graph(
      self,
      filter_membership_list_0,
      filter_membership_list_1):
    """Plot trial-to-trial memberships."""
    num_groups = 5
    num_trials = len(filter_membership_list_0)
    graphs0, _, _ = self.parse_filter_membership_list(
        filter_membership_list_0, build_graph=True)
    graphs1, _, _ = self.parse_filter_membership_list(
        filter_membership_list_1, build_graph=True)

    gs_kw = dict(width_ratios=[1,1]*num_groups, height_ratios=[1]*num_trials)
    fig, axs = plt.subplots(figsize=(20, 40), gridspec_kw=gs_kw,
        nrows=num_trials, ncols=2*num_groups)
    plt.subplots_adjust(left=None, right=None, hspace=0.02, wspace=0.02)
    for r in range(num_trials):
      adj0 = nx.to_numpy_matrix(graphs0[r])
      adj1 = nx.to_numpy_matrix(graphs1[r])

      for group_id in range(num_groups):
        adj_mask_0 = (adj0==group_id)+0
        adj_mask_1 = (adj1==group_id)+0

        ax = fig.add_subplot(axs[r,0+group_id*2])
        ax.axis('off')
        seaborn.heatmap(adj_mask_0, cbar=False, square=False)

        ax = fig.add_subplot(axs[r,1+group_id*2])
        ax.axis('off')
        seaborn.heatmap(adj_mask_1, cbar=False, square=False)

    plt.show()


  def calculate_filter_memberships_corr(
      self,
      filter_membership_list_0,
      filter_membership_list_1):
    """Plot trial-to-trial memberships."""
    if isinstance(filter_membership_list_0, list):
      num_trials = len(filter_membership_list_0)
      _, _, adjvec0 = self.parse_filter_membership_list(
          filter_membership_list_0, build_graph=False)
      _, _, adjvec1 = self.parse_filter_membership_list(
          filter_membership_list_1, build_graph=False)
      num_pairs0 = adjvec0.shape[0]
      num_pairs1 = adjvec1.shape[0]
      indices0 = filter_membership_list_0[0].index
      indices1 = filter_membership_list_1[0].index
    elif isinstance(filter_membership_list_0, pd.core.frame.DataFrame):
      num_trials = filter_membership_list_0.shape[1]
      num_pairs0 = filter_membership_list_0.shape[0]
      num_pairs1 = filter_membership_list_1.shape[0]
      adjvec0 = filter_membership_list_0.values
      adjvec1 = filter_membership_list_1.values
      indices0 = filter_membership_list_0.index
      indices1 = filter_membership_list_1.index

    # Method 1: pearson corr.
    # corr = np.zeros([num_pairs0, num_pairs1])
    # for r in range(num_pairs0):
    #   for c in range(num_pairs1):
    #     corr[r,c],_ = scipy.stats.pearsonr(adjvec0[r], adjvec1[c])
    # Method 2: category-wise corr.
    # corr = 0
    # for cat0 in [0,1,2]:
    #   for cat1 in [1,2]:
    #     adjvec0_cnt = (adjvec0 == cat0) + 0
    #     adjvec1_cnt = (adjvec1 == cat1) + 0
    #     corr += adjvec0_cnt @ adjvec1_cnt.T
    # plt.figure(figsize=[10, 8])
    # seaborn.heatmap(corr, square=True)  # cmap='PiYG'
    # plt.figure(figsize=[4, 2])
    # seaborn.distplot(corr, kde=False, norm_hist=True)

    # Method 3: Cramer's V using contingency table.
    num_cats = 5
    categories = [0,1,2,3,4]
    contingency_mat = np.zeros([num_cats, num_cats, num_pairs0, num_pairs1])

    for cat0 in categories:
      for cat1 in categories:
        # adjvec0: neurons x trials.
        adjvec0_cnt = (adjvec0 == cat0) + 0
        adjvec1_cnt = (adjvec1 == cat1) + 0
        contingency_mat[cat0, cat1] = adjvec0_cnt @ adjvec1_cnt.T

    chi2t_mat = np.zeros([num_pairs0, num_pairs1])
    pval_mat = np.zeros([num_pairs0, num_pairs1])
    neglog_pval_tab = np.zeros([num_pairs0, num_pairs1])
    if hasattr(tqdm, '_instances'):
      tqdm._instances.clear()
    for r in tqdm(range(num_pairs0), ncols=100, file=sys.stdout):
      # if r > 100:
      #   break
      for c in range(num_pairs1):
        tab = contingency_mat[:,:,r,c]
        tab = tab[:,~np.all(tab == 0, axis=0)]
        tab = tab[~np.all(tab == 0, axis=1)]
        chi2t, pval, dof, _ = scipy.stats.chi2_contingency(tab, correction=True)
        chi2t_mat[r,c] = chi2t
        pval_mat[r,c] = pval
        if not np.isnan(pval):
          neglog_pval_tab[r,c] = -np.log(pval)
        else:
          neglog_pval_tab[r,c] = np.nan

    pval_df = pd.DataFrame(pval_mat, index=indices0, columns=indices1)
    chi2t_df = pd.DataFrame(chi2t_mat, index=indices0, columns=indices1)
    return contingency_mat, chi2t_df, pval_df


  @classmethod
  def plot_statistics_contingency_tables(
      cls,
      contingency_mat):
    """Plot contingency table statistics as a verification."""
    total_cnts = contingency_mat.sum(axis=(0,1))

    plt.figure(figsize=[4, 2])
    seaborn.distplot(total_cnts, color='grey', kde=False, norm_hist=True)
    plt.show()


  def plot_filter_memberships_corr(
      self,
      contingency_mat,
      chi2t_df,
      pval_df):
    """Plot the results of `calculate_filter_memberships_corr`."""
    num_cats, _, num_pairs0, num_pairs1 = contingency_mat.shape
    num_trials = np.sum(contingency_mat[:,:,0,0])

    def pval_sig_cnt(pval_df, threshold):
      return (pval_df < threshold).sum().sum()
    def pval_sig_ratio(pval_df, threshold):
      return (pval_df < threshold).sum().sum() / pval_df.size
    print('p-val\t\tcnt\tportion')
    for thr in [0.05, 0.01, 1e-3, 1e-5, 1e-7, 1e-10]:
      print(f'p-val<{thr}:\t{pval_sig_cnt(pval_df, thr)}'+
            f'\t{pval_sig_ratio(pval_df, thr):.2e}')

    def neg_log(x):
      if not np.isnan(x) and x > 0:
        return -np.log10(x)
      elif not np.isnan(x) and x <= 0:
        np.nan
      else:
        return np.nan

    # The multi-index for axis=0 and 1 are matched on level 0.
    pval_df = pval_df.swaplevel(i=0, j=1, axis=0)
    pval_df = pval_df.sort_index(axis=0, level=0)
    pval_df = pval_df.sort_index(axis=1, level=0)

    neglog_pval_df = pval_df.applymap(neg_log)
    rows = neglog_pval_df.sum(axis=1)
    rows = rows.sort_values(ascending=False, na_position='last')
    df = neglog_pval_df.reindex(rows.index)
    cols = df.sum(axis=0)
    cols = cols.sort_values(ascending=False, na_position='last')
    neglog_pval_tab_sorted = df.reindex(columns=cols.index)

    fig, axs = plt.subplots(figsize=[20, 7], ncols=2, nrows=1)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
    ax = fig.add_subplot(axs[0])
    seaborn.heatmap(neglog_pval_df, cbar=False, square=True, cmap='jet')
    plt.locator_params(axis='x', nbins=20)
    plt.locator_params(axis='y', nbins=20)
    plt.ylabel('V1\u27F6LM pairs')
    plt.xlabel('LM\u27F6AL pairs')

    ax = fig.add_subplot(axs[1])
    seaborn.heatmap(neglog_pval_tab_sorted, square=True, cmap='jet')  # cmap='PiYG'
    plt.locator_params(axis='x', nbins=20)
    plt.locator_params(axis='y', nbins=20)
    plt.ylabel('V1\u27F6LM pairs')
    plt.xlabel('LM\u27F6AL pairs')
    plt.show()

    # Cramer's V.
    # cramerv = np.sqrt(chi2t_df / num_trials / num_cats)
    # plt.figure(figsize=[10, 8])
    # seaborn.heatmap(cramerv, square=True)  # cmap='PiYG'
    # plt.show()

    # plt.figure(figsize=[10, 8])
    # seaborn.heatmap(chi2t_df, square=True)  # cmap='PiYG'
    # plt.show()

    # plt.figure(figsize=[10, 2])
    # plt.subplot(121)
    # seaborn.distplot(chi2t_tab, kde=False, norm_hist=True)
    # plt.subplot(122)
    # seaborn.distplot(pval_data, bins=500, kde=False, norm_hist=True)
    # t = np.linspace(0, 6, 100)
    # plt.plot(t, scipy.stats.expon.pdf(t))
    # plt.xlim(0, 8)
    # plt.show()


  def plot_filter_memberships_corr_beautiful(
      self,
      contingency_mat,
      chi2t_df,
      pval_df,
      level_order='10',
      multitest_method='fdr_by',
      file_path=None):
    """Plot the results of `calculate_filter_memberships_corr`.

    Args:
      level_order: '01' sort level-0 first, then sort level-1.
                   '10' sort level-1 first, then sort level-0.
      multitest_method: fdr_bh, fdr_by, bonferroni,
    """
    pval_df = pval_df.copy()
    # This needs to be run first before sorting the pval_df.
    diag_neurons, bro_rec_neurons,_,_ = self.significant_pair_block_pattern_analysis(
        pval_df, test_alpha=0.05, multitest_method=multitest_method, verbose=False)

    def fdr_cnt(pval_df, threshold):
      reject,_,_,_ = multipletests(pval_df.values.reshape(-1),
          alpha=threshold, method='fdr_by')
      return sum(reject)
    def bonferroni_cnt(pval_df, threshold):
      reject,_,_,_ = multipletests(pval_df.values.reshape(-1),
          alpha=threshold, method='bonferroni')
      return sum(reject)
    def pval_sig_cnt(pval_df, threshold):
      return (pval_df < threshold).sum().sum()
    def pval_sig_ratio(pval_df, threshold):
      return (pval_df < threshold).sum().sum() / pval_df.size
    print('Total sample size', pval_df.size)
    print('p-val\t\tcnt\tportion')
    for thr in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
      print(f'p-val<{thr}:\t{pval_sig_cnt(pval_df, thr)}'+
            f'\t{pval_sig_ratio(pval_df, thr):.2e}')
    print('FDR BY')
    for thr in [0.05, 0.01]:
      print(f'alpha<{thr}:\t{fdr_cnt(pval_df, thr)}')
    print('FWER Bonferroni')
    for thr in [0.05, 0.01]:
      print(f'alpha<{thr}:\t{bonferroni_cnt(pval_df, thr)}')

    # The multi-index for axis=0 and 1 are matched on level 0.
    # Within in each level-0 index block, all level-1 are sorted in the same way.
    # level 0 is on the outside, and level 1 is on the inside.
    if level_order == '10':
      pval_df = pval_df.swaplevel(i=0, j=1, axis=0)
      pval_df = pval_df.sort_index(axis=0, level=1)
      pval_df = pval_df.sort_index(axis=0, level=0)
      pval_df = pval_df.sort_index(axis=1, level=1)
      pval_df = pval_df.sort_index(axis=1, level=0)

      row_index_leve0 = pval_df.index.get_level_values(0).unique()
      col_index_leve0 = pval_df.columns.get_level_values(0).unique()
      level0_neuron_match = np.array_equal(row_index_leve0, col_index_leve0)
      if level0_neuron_match:
        print('Row-col level-0 match!')
      else:
        print(row_index_leve0)
        print(col_index_leve0)

      neuron_label = np.zeros(len(row_index_leve0))
      neuron_label[::2] += 1
      neuron_label_map = dict(zip(row_index_leve0, neuron_label))

    elif level_order == '01':
      # The multi-index for axis=0 and 1 are matched on level 1.
      # level 0 is on the outside, and level 1 is on the inside.
      pval_df = pval_df.swaplevel(i=0, j=1, axis=0)
      pval_df = pval_df.sort_index(axis=0, level=0)
      pval_df = pval_df.sort_index(axis=0, level=1)
      pval_df = pval_df.sort_index(axis=1, level=0)
      pval_df = pval_df.sort_index(axis=1, level=1)

      row_index_leve1 = pval_df.index.get_level_values(1).unique()
      neuron_label = np.zeros(len(row_index_leve1))
      neuron_label[::2] += 1
      neuron_label_row_map = dict(zip(row_index_leve1, neuron_label))

      col_index_leve1 = pval_df.columns.get_level_values(1).unique()
      neuron_label = np.zeros(len(col_index_leve1))
      neuron_label[::2] += 1
      neuron_label_col_map = dict(zip(col_index_leve1, neuron_label))

    def neg_log(x):
      if np.isnan(x) or x <= 0:
        return np.nan
      else:
        return -np.log10(x)
    # plot_pval = pval_df.applymap(neg_log)

    def pval_threshold(x):
      if np.isnan(x) or x <= 0:
        return np.nan
      elif x > 0.05:
        return np.nan
      elif x > 0.01:
        return 1
      elif x > 0:
        return 2

    if multitest_method is not None:
      reject,pvals_corrected,_,_ = multipletests(pval_df.values.reshape(-1),
          alpha=0.05, method=multitest_method)
      pval_df.iloc[:,:] = pvals_corrected.reshape(pval_df.values.shape)

    plot_pval = pval_df.applymap(pval_threshold)

    # A small test: number of rejections at level alpha=0.05 is equal to the
    # number of p-values under the threshold 0.05. This mean we can reject the
    # hypothesis using normal procedure or threhold on adjusted p-values.
    # print(sum(reject), sum(pvals_corrected<0.05))
    # print(len(reject), len(pvals_corrected))

    gs_kw = dict(width_ratios=[0.02, 1, 0.02], height_ratios=[0.02, 1, 0.02])
    fig, axs = plt.subplots(figsize=(7, 7), gridspec_kw=gs_kw, nrows=3, ncols=3)
    cbar_ax = fig.add_axes([.93, .33, .02, .33])

    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0)
    ax = fig.add_subplot(axs[0,0])
    ax.axis('off')
    ax = fig.add_subplot(axs[2,2])
    ax.axis('off')
    ax = fig.add_subplot(axs[0,2])
    ax.axis('off')
    ax = fig.add_subplot(axs[2,0])
    ax.axis('off')

    ax = fig.add_subplot(axs[1,0])
    ax.axis('off')
    if level_order == '10':
      row_idx = pval_df.index.get_level_values(0)
      row_idx = pd.DataFrame(row_idx).replace(neuron_label_map)
    elif level_order == '01':
      row_idx = pval_df.index.get_level_values(1)
      row_idx = pd.DataFrame(row_idx).replace(neuron_label_row_map)
    row_idx = np.array(row_idx).reshape(-1,1)
    seaborn.heatmap(row_idx, cbar=False, cmap='binary', vmin=-0.2, vmax=1)
    ax = fig.add_subplot(axs[1,2])
    ax.axis('off')
    seaborn.heatmap(row_idx, cbar=False, cmap='binary', vmin=-0.2, vmax=1)

    ax = fig.add_subplot(axs[0,1])
    ax.axis('off')
    if level_order == '10':
      col_idx = pval_df.columns.get_level_values(0)
      col_idx = pd.DataFrame(col_idx).replace(neuron_label_map)
    elif level_order == '01':
      col_idx = pval_df.columns.get_level_values(1)
      col_idx = pd.DataFrame(col_idx).replace(neuron_label_col_map)
    col_idx = np.array(col_idx).reshape(1,-1)
    seaborn.heatmap(col_idx, cbar=False, cmap='binary', vmin=-0.2, vmax=1)
    ax = fig.add_subplot(axs[2,1])
    ax.axis('off')
    seaborn.heatmap(col_idx, cbar=False, cmap='binary', vmin=-0.2, vmax=1)

    cmap = matplotlib.colors.ListedColormap(['white', 'mediumseagreen', 'red'])
    ax = fig.add_subplot(axs[1,1])
    plt.tick_params(bottom=False, left=False,
                    labelleft=False, labelbottom=False)
    ax = seaborn.heatmap(plot_pval, cmap=cmap, vmin=0, vmax=3,
        cbar_ax=cbar_ax, cbar_kws={'label': r'$\alpha$'})
    cbar = ax.collections[0].colorbar
    cbar.outline.set_edgecolor('k')
    cbar.outline.set_linewidth(2)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['1', '0.05', '0.01', '0'])
    if level_order == '10':
      row_index_leve0 = pval_df.index.get_level_values(0)
      col_index_leve0 = pval_df.columns.get_level_values(0)
      for n in diag_neurons:
        top = np.where(row_index_leve0==n)[0][0]
        down = np.where(row_index_leve0==n)[0][-1]+1
        left = np.where(col_index_leve0==n)[0][0]
        right = np.where(col_index_leve0==n)[0][-1]+1
        ax.add_patch(matplotlib.patches.Rectangle(
            (left, top), right-left, down-top,
            fill=False, edgecolor='cyan', lw=2, ls='-'))
    elif level_order == '01':
      row_index_leve1 = pval_df.index.get_level_values(1)
      col_index_leve1 = pval_df.columns.get_level_values(1)
      for (row, col) in bro_rec_neurons:
        top = np.where(row_index_leve1==row)[0][0]
        down = np.where(row_index_leve1==row)[0][-1]+1
        left = np.where(col_index_leve1==col)[0][0]
        right = np.where(col_index_leve1==col)[0][-1]+1
        ax.add_patch(matplotlib.patches.Rectangle(
            (left, top), right-left, down-top,
            fill=False, edgecolor='cyan', lw=2, ls='-'))
    plt.locator_params(axis='x', nbins=20)
    plt.locator_params(axis='y', nbins=20)
    plt.ylabel('V1\u27F6LM pairs', fontsize=15, labelpad=15)
    plt.xlabel('LM\u27F6AL pairs', fontsize=15, labelpad=15)

    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight', dpi=400)
      print('Save figure to:', file_path)
    plt.show()


  def bootstrap_contingency_entries(
      cls,
      contingency_table,
      num_samples=100,
      random_seed=None,
      verbose=False):
    """Bootstrap contingency table entries."""
    np.random.seed(random_seed)
    num_rows, num_cols = contingency_table.shape
    total = contingency_table.sum().astype(int)
    prows = contingency_table.sum(axis=1) / total
    pcols = contingency_table.sum(axis=0) / total
    bootstrap_tabs = np.zeros([num_samples, num_rows, num_cols])

    for s in range(num_samples):
      for e in range(total):
        r_ind = np.random.choice(range(num_rows), p=prows)
        c_ind = np.random.choice(range(num_cols), p=pcols)
        bootstrap_tabs[s,r_ind, c_ind] += 1

    # seaborn.distplot(bootstrap_tabs[:,0,0], bins=30)
    # plt.show()
    # seaborn.distplot(bootstrap_tabs[:,2,2], bins=30)
    # plt.show()
    # print(bootstrap_tabs.mean(axis=0))
    # print(bootstrap_tabs.std(axis=0))

    p_vals = np.zeros_like(contingency_table)
    for r in range(num_rows):
      for c in range(num_cols):
        right_tail = np.sum(bootstrap_tabs[:,r,c] >= contingency_table[r,c]) / num_samples
        left_tail = np.sum(bootstrap_tabs[:,r,c] < contingency_table[r,c]) / num_samples
        p_vals[r,c] = min(left_tail, right_tail)
    print('raw', p_vals)

    _,pvals_corrected,_,_ = multipletests(p_vals.reshape(-1),
        alpha=0.05, method='bonferroni')
    pvals_corrected = pvals_corrected.reshape(num_rows, num_cols)
    print('bonferroni', pvals_corrected)

    _,pvals_corrected,_,_ = multipletests(p_vals.reshape(-1),
        alpha=0.05, method='fdr_bh')
    pvals_corrected = pvals_corrected.reshape(num_rows, num_cols)
    print('fdr_bh', pvals_corrected)

    _,pvals_corrected,_,_ = multipletests(p_vals.reshape(-1),
        alpha=0.05, method='fdr_by')
    pvals_corrected = pvals_corrected.reshape(num_rows, num_cols)
    print('fdr_by', pvals_corrected)



  def significant_pair_block_pattern_analysis(
      self,
      pval_df,
      contingency_mat=None,
      multitest_method='fdr_by',
      test_alpha=0.05,
      verbose=True):
    """Plot the results of `calculate_filter_memberships_corr`.

    Args:
      multitest_method: fdr_bh, fdr_by, bonferroni,
    """
    pval_df = pval_df.copy()
    if multitest_method is not None:
      _,pvals_corrected,_,_ = multipletests(pval_df.values.reshape(-1),
          alpha=test_alpha, method=multitest_method)
      pval_mat = pvals_corrected.reshape(pval_df.values.shape)
      pval_df.iloc[:,:] = pval_mat
    broadcast = pval_df.index.get_level_values(0).unique()
    inter = pval_df.index.get_level_values(1).unique()
    receiver = pval_df.columns.get_level_values(1).unique()

    total_cnt = (pval_df < test_alpha).values.sum()
    # The original format of the index is like: V1-->LM.
    # The format of column index is like: LM-->AL.

    #---------- Count hub pathways, like manyV1-->LM-->manyAL ----------
    diag_block_pair_cnt = 0
    diag_neurons = []
    diag_pairs = {}
    for inter_neuron in inter:
      row_subindices = pval_df.index.get_level_values(1) == inter_neuron
      col_subindices = pval_df.columns.get_level_values(0) == inter_neuron
      sub_df = pval_df.iloc[row_subindices, col_subindices]
      cnt = (sub_df < test_alpha).values.sum()

      if cnt > 0.06 * 24*24:
        if verbose:
          print('Diag neuron:', inter_neuron, cnt)
        diag_block_pair_cnt += cnt
        diag_neurons.append(inter_neuron)
        # Find all pairs.
        row_ids, col_ids = np.where((sub_df < test_alpha).values)
        row_pairs = sub_df.index.values[row_ids]
        col_pairs = sub_df.columns.values[col_ids]
        if contingency_mat is not None:
          contingency_submat = contingency_mat.transpose(2,3,0,1)
          contingency_submat = contingency_submat[np.ix_(row_subindices, col_subindices)]
          contingency = contingency_submat[row_ids,col_ids]
          diag_pairs[inter_neuron] = list(zip(row_pairs, col_pairs, contingency))
        else:
          diag_pairs[inter_neuron] = list(zip(row_pairs, col_pairs))

    #---------- Count broadcast receiver pathways, V1-->manyLM-->AL. ----------
    broadcast_receiver_cnt = 0
    subdiag_cnt = 0
    bro_rec_neurons = []
    bro_rec_pairs = {}
    for source_neuron in broadcast:
      for target_neuron in receiver:
        row_subindices = pval_df.index.get_level_values(0) == source_neuron
        col_subindices = pval_df.columns.get_level_values(1) == target_neuron
        sub_df = pval_df.iloc[row_subindices, col_subindices]
        cnt = (sub_df < test_alpha).values.sum()
        if cnt > 0.051 * 24*24:
          if verbose:
            print(source_neuron, target_neuron, cnt)
          broadcast_receiver_cnt += cnt
          # Set a new df so index of `sub_df` is not changed.
          sub_df_sort = sub_df.sort_index(axis=0, level=1)
          sub_df_sort = sub_df_sort.sort_index(axis=1, level=0)
          subdiag_cnt += sum(np.diag(sub_df_sort.values) < test_alpha)
          bro_rec_neurons.append((source_neuron, target_neuron))
          # Find all pairs.
          row_ids, col_ids = np.where((sub_df < test_alpha).values)
          row_pairs = sub_df.index.values[row_ids]
          col_pairs = sub_df.columns.values[col_ids]
          if contingency_mat is not None:
            contingency_submat = contingency_mat.transpose(2,3,0,1)
            contingency_submat = contingency_submat[np.ix_(row_subindices, col_subindices)]
            contingency = contingency_submat[row_ids,col_ids]
            bro_rec_pairs[(source_neuron, target_neuron)] = list(
                zip(row_pairs, col_pairs, contingency))
          else:
            bro_rec_pairs[(source_neuron, target_neuron)] = list(
                zip(row_pairs, col_pairs))

    if verbose:
      print(f'Total:{total_cnt}\thub:{diag_block_pair_cnt} '+
            f'({diag_block_pair_cnt/total_cnt:.5f})\t'+
            f'bro-rec:{broadcast_receiver_cnt} '+
            f'({broadcast_receiver_cnt/total_cnt:.5f})\t'
            f'sum:{diag_block_pair_cnt+broadcast_receiver_cnt}\t'+
            f'intersect:{diag_block_pair_cnt+broadcast_receiver_cnt-subdiag_cnt}')
      print('len(diag_pairs), len(bro_rec_pairs)',
            len(diag_pairs), len(bro_rec_pairs))

    return diag_neurons, bro_rec_neurons, diag_pairs, bro_rec_pairs


  def select_membership_block_from_significant_pairs(
      self,
      spike_times,
      filter_pairs,
      filter_membership):
    """This select strong coulped neurons and trials.

    We select spike_times by aggregating every pair of filters. The pair of
    filters share the required types `filter_type_1` and `filter_type_2`.
    Then we concatenate those spike trains together. Note that some pairs may
    have the same trial.
    """
    block_id = 7
    filter_type_1 = 1
    filter_type_2 = 1

    block_pairs = filter_pairs[block_id].copy()
    block_pairs_sorted = block_pairs.sort(
        key=lambda x: x[2][filter_type_1,filter_type_2], reverse=True)

    spike_times_1, spike_times_2, spike_times_3, spike_times_4 = [], [], [], []
    for (neuron_1, neuron_2), (neuron_3, neuron_4), contingency in block_pairs:
      cnt = contingency[filter_type_1,filter_type_2]
      if cnt < 20:
        break
      df1 = filter_membership.loc[(neuron_1, neuron_2)]
      df2 = filter_membership.loc[(neuron_3, neuron_4)]
      df = ((df1==filter_type_1)&(df2==filter_type_2))
      trial_ids = df[df==True].index.values
      spike_times_1.extend(spike_times.loc[neuron_1, trial_ids].tolist())
      spike_times_2.extend(spike_times.loc[neuron_2, trial_ids].tolist())
      spike_times_3.extend(spike_times.loc[neuron_3, trial_ids].tolist())
      spike_times_4.extend(spike_times.loc[neuron_4, trial_ids].tolist())

    return spike_times_1, spike_times_2, spike_times_3, spike_times_4


  def select_neurons_trials_from_significant_pairs(
      self,
      filter_pairs,
      filter_membership):
    """This select strong coulped neurons and trials.

    This functions' input come from `significant_pair_block_pattern_analysis`.
    """
    block_id = 7
    filter_type_1 = 1
    filter_type_2 = 1

    # filter_pairs: blocks x pair_tuples.
    # pair_tuple: ((n1,n2),(n3,n4),contingency)
    ((neuron_1, neuron_2), (neuron_3, neuron_4), contingency) = max(
        filter_pairs[block_id], key=lambda x: x[2][filter_type_1,filter_type_2])
    df1 = filter_membership.loc[(neuron_1, neuron_2)]
    df2 = filter_membership.loc[(neuron_3, neuron_4)]
    df = ((df1==filter_type_1)&(df2==filter_type_2))
    trial_ids = df[df==True].index.values

    print((neuron_1, neuron_2), (neuron_3, neuron_4))
    print(contingency)
    print('num trials:', len(trial_ids))

    return (neuron_1, neuron_2), (neuron_3, neuron_4), trial_ids

    # Unit test for selected contingency.
    df1 = filter_membership.loc[(neuron_1, neuron_2)]
    df2 = filter_membership.loc[(neuron_3, neuron_4)]
    contingency_target = np.zeros_like(contingency)
    for type1 in df1.unique():
      for type2 in df2.unique():
        u = (df1.values==type1)+0
        v = (df2.values==type2)+0

        contingency_target[int(type1), int(type2)] = np.dot(u,v.T)
    contingency_err = contingency - contingency_target
    print('Contingency err:', np.sum(contingency_err))


  def select_sub_filter_membership_by_neurons(
      self,
      filter_membership,
      source_neurons=None,
      target_neurons=None):
    """Select pairs containing `neurons`."""
    pairs = filter_membership.index.values

    select_pairs_source = []
    if source_neurons is not None:
      for neuron_x, neuron_y in pairs:
        if neuron_x in source_neurons:
          select_pairs_source.append((neuron_x, neuron_y))

    select_pairs_target = []
    if target_neurons is not None:
      for neuron_x, neuron_y in pairs:
        if neuron_y in target_neurons:
          select_pairs_target.append((neuron_x, neuron_y))

    select_pairs = select_pairs_source + select_pairs_target
    print('num selected pairs:', len(select_pairs))
    return filter_membership.loc[select_pairs]


  def plot_filter_memberships_significant_corr_adj(
      self,
      contingency_mat,
      chi2t_df,
      pval_df,
      pval_threshold=0.01,
      multitest_method='fdr_by',
      sort_nodes=True,
      file_path=None):
    """Plot the results of `calculate_filter_memberships_corr`."""
    pval_df = pval_df.copy()
    _,pvals_corrected,_,_ = multipletests(
        pval_df.values.reshape(-1), method=multitest_method)
    pval_df.iloc[:,:] = pvals_corrected.reshape(pval_df.values.shape)

    # Find significant correlated pairs.
    sig_pairs_idx_0, sig_pairs_idx_1 =  np.where(pval_df < pval_threshold)
    sig_pairs_idx_0 = pval_df.index.values[sig_pairs_idx_0]
    sig_pairs_idx_1 = pval_df.columns.values[sig_pairs_idx_1]

    sig_pairs_0 = pd.DataFrame(index=pval_df.index, columns=[0])
    sig_pairs_1 = pd.DataFrame(index=pval_df.columns, columns=[0])
    sig_pairs_0 = sig_pairs_0.fillna(0)
    sig_pairs_1 = sig_pairs_1.fillna(0)
    sig_pairs_0.loc[sig_pairs_idx_0] = 1
    sig_pairs_1.loc[sig_pairs_idx_1] = 1
    sig_pairs = pd.concat([sig_pairs_0, sig_pairs_1], axis=0)

    graph = self.build_neuron_graph(sig_pairs, sort_nodes=sort_nodes, verbose=2,
        file_path=file_path)


  def plot_filter_memberships_significant_corr_subadj(
      self,
      contingency_mat,
      chi2t_df,
      pval_df,
      pval_threshold=1e-5,
      sort_nodes=True,
      file_path=None):
    """Plot the results of `calculate_filter_memberships_corr`."""
    def neg_log(x):
      if not np.isnan(x) and x > 0:
        return -np.log(x)
      elif not np.isnan(x) and x <= 0:
        np.nan
      else:
        return np.nan
    neglog_pval_df = pval_df.applymap(neg_log)
    rows = neglog_pval_df.sum(axis=1)
    rows = rows.sort_values(ascending=False, na_position='last')
    df = neglog_pval_df.reindex(rows.index)
    pval_df = pval_df.reindex(rows.index)
    cols = df.sum(axis=0)
    cols = cols.sort_values(ascending=False, na_position='last')
    neglog_pval_tab_sorted = df.reindex(columns=cols.index)
    pval_df = pval_df.reindex(columns=cols.index)

    # Connectivity analysis.
    pval_df = pval_df < pval_threshold
    row_degree = pval_df.sum(axis=1)
    col_degree = pval_df.sum(axis=0)
    row_degree = row_degree[row_degree>0]
    col_degree = col_degree[col_degree>0]

    fig, axs = plt.subplots(figsize=[8, 2], ncols=2, nrows=1)
    plt.subplots_adjust(left=None, right=None, hspace=0, wspace=0.2)
    ax = fig.add_subplot(axs[0])
    seaborn.distplot(row_degree, bins=30, kde=False, norm_hist=True, color='grey')
    plt.title('degree of V1\u27F6LM')
    ax = fig.add_subplot(axs[1])
    seaborn.distplot(col_degree, bins=30, kde=False, norm_hist=True, color='grey')
    plt.title('degree of LM\u27F6AL')


  def plot_significant_contingency(
      self,
      filter_membership_list_0,
      filter_membership_list_1,
      contingency_table,
      chi2t_df,
      pval_df,
      pval_threshold=0.01,
      multitest_method='fdr_by',
      file_path=None):
    """Plot the results of `calculate_filter_memberships_corr`."""
    pval_df = pval_df.copy()
    _,pvals_corrected,_,_ = multipletests(
        pval_df.values.reshape(-1), method=multitest_method)
    pval_df.iloc[:,:] = pvals_corrected.reshape(pval_df.values.shape)
    sig_pairs_idx_0, sig_pairs_idx_1 =  np.where(pval_df < pval_threshold)

    if isinstance(filter_membership_list_0, list):
      sig_pairs_0 = pd.DataFrame().reindex_like(filter_membership_list_0[0][1])
      sig_pairs_1 = pd.DataFrame().reindex_like(filter_membership_list_1[0][1])
    elif isinstance(filter_membership_list_0, pd.core.frame.DataFrame):
      sig_pairs_0 = pd.DataFrame(index=filter_membership_list_0.index, columns=[0])
      sig_pairs_1 = pd.DataFrame(index=filter_membership_list_1.index, columns=[0])

    sig_pairs_0 = sig_pairs_0.fillna(0)
    sig_pairs_1 = sig_pairs_1.fillna(0)
    sig_pairs_0.iloc[sig_pairs_idx_0] = 1
    sig_pairs_1.iloc[sig_pairs_idx_1] = 1
    sig_pairs = pd.concat([sig_pairs_0, sig_pairs_1], axis=0)

    sig_contingency_table = contingency_table[:,:,sig_pairs_idx_0,sig_pairs_idx_1]
    sig_chi2t = chi2t_df.values[sig_pairs_idx_0, sig_pairs_idx_1]
    num_cats,_,num_data = sig_contingency_table.shape
    sample_size = np.sum(contingency_table[:,:,0,0])
    print('contingency_table.shape', contingency_table.shape)
    print('sig_contingency_table.shape', sig_contingency_table.shape)
    print('Total sample size:', sample_size)
    tab_features = sig_contingency_table.reshape(-1, num_data)
    # All the data.
    # tab_features = contingency_table.reshape(num_cats*num_cats, -1)

    all_tab_mean = contingency_table.mean(axis=(2,3))
    sig_tab_mean = sig_contingency_table.mean(axis=2)

    cramerv_sig = np.mean(np.sqrt(sig_chi2t / sample_size /5))
    cramerv_all = np.mean(np.sqrt(chi2t_df.values / sample_size /5), axis=(0,1))
    print(f'CramerV all: {cramerv_all:.3f}\tSig:{cramerv_sig:.3f}')

    plt.figure(figsize=[10, 4])
    plt.subplot(121)
    seaborn.heatmap(all_tab_mean, square=True, annot=True, fmt='.1f', cbar=False)
    plt.title('mean of all')
    plt.ylabel('V1\u27F6LM')
    plt.xlabel('LM\u27F6AL')
    plt.subplot(122)
    seaborn.heatmap(sig_tab_mean, square=True, annot=True, fmt='.1f', cbar=False)
    plt.title('mean of significance')

    if file_path is not None:
      plt.savefig(file_path, bbox_inches='tight')
      print('Save figure to:', file_path)
    plt.show()

    # Clustering analysis.
    # num_clusters = 6
    # kmeans = sklearn.cluster.KMeans(
    #     n_clusters=num_clusters, random_state=0).fit(tab_features.T)
    # cluster_sample_cnt = []
    # for cluster_id in range(num_clusters):
    #   cluster_labels = np.where(kmeans.labels_ == cluster_id)[0]
    #   num_samples = len(cluster_labels)
    #   cluster_sample_cnt.append((cluster_id, num_samples))
    # cluster_sample_cnt = sorted(cluster_sample_cnt, key=lambda x: x[1], reverse=True)
    # print('(cluster_id, cnt)', cluster_sample_cnt)


    # fig, axs = plt.subplots(figsize=(num_clusters*3.5, 2.5),
    #     ncols=num_clusters, nrows=1)
    # for i in range(num_clusters):
    #   ax = fig.add_subplot(axs[i])
    #   tab = kmeans.cluster_centers_[i].reshape(num_cats,num_cats)
    #   seaborn.heatmap(tab, square=True)
    # plt.show()

    # PCA analysis.
    # pca = sklearn.decomposition.PCA(n_components=5)
    # pca = pca.fit(tab_features.T)
    # features_pca = pca.components_.T
    # pca_sigma = pca.singular_values_
    # pca_mean = pca.mean_
    # plt.figure(figsize=[8,2])
    # plt.subplot(131)
    # plt.plot(features_pca[:,0], features_pca[:,1], '.')
    # plt.subplot(132)
    # plt.plot(features_pca[:,0], features_pca[:,2], '.')
    # plt.subplot(133)
    # plt.plot(features_pca[:,1], features_pca[:,2], '.')



  def plot_trial_to_trial_filter_memberships_adjvec_contrast(
      self,
      filter_membership_list_0,
      filter_membership_list_1):
    """Plot trial-to-trial memberships."""
    num_trials = len(filter_membership_list_0)
    _, _, adjvec0 = self.parse_filter_membership_list(
        filter_membership_list_0, build_graph=False)
    _, _, adjvec1 = self.parse_filter_membership_list(
        filter_membership_list_1, build_graph=False)

    # plt.figure(figsize=[5, 40])
    # seaborn.heatmap(adjvec0==4, square=False, cbar=False)
    # plt.show()

    plt.figure(figsize=[5, 40])
    seaborn.heatmap(adjvec1==0, square=False, cbar=False)
    plt.show()


  def test_func(self, x, arr):
    time.sleep(0.1)
    arr[x] = x*2

    # x = self.spike_times.copy()

