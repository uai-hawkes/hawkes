"""Data models."""
import os

from absl import logging
import collections
from collections import defaultdict
import io
import itertools
import numpy as np
# import matlab
# import matlab.engine
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
import scipy
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

import smoothing_spline
import util

MAX_CRUSH_CNT = 4


class AllenInstituteDataModel(object):

  def __init__(self, session=None):
    self.session = session

  def print_session_info(self):
    """Print a list of information for the session."""
    if self.session is None:
      print('session is None')
      return

    print(self.session._metadata)
    print('num units:', self.session.num_units)

  def get_trial_metric_per_unit_per_trial(
      self,
      stimulus_presentation_ids,
      unit_ids,
      trial_time_window,
      metric_type,
      dt=0.002,
      empty_fill=np.nan,
      verbose=False):
    """Plots selected units.

    Args:
      metric_type:
          'count',
          'spike_trains' (spike histogram, array of binary of interger counts),
          'spike_times' (a sequence of spike times)
    """
    stimulus_presentations = self.session._filter_owned_df(
        'stimulus_presentations', ids=stimulus_presentation_ids)
    units = self.session._filter_owned_df('units', ids=unit_ids)
    # display(units)

    spikes_table = self.session.trialwise_spike_times(
        stimulus_presentation_ids, unit_ids, trial_time_window)
    num_neurons = len(unit_ids)
    num_trials = len(stimulus_presentation_ids)
    metric_table = pd.DataFrame(index=unit_ids,
                                columns=stimulus_presentation_ids)
    metric_table.index.name = 'units'

    if metric_type == 'spike_trains':
      time_bins = np.linspace(
          trial_time_window[0], trial_time_window[1],
          int((trial_time_window[1] - trial_time_window[0]) / dt) + 1)

    for u, unit_id in enumerate(unit_ids):
      if verbose and (u % 40 == 0):
        print('neuron:', u)
      for s, stimulus_presentation_id in enumerate(stimulus_presentation_ids):
        spike_times = spikes_table[
            (spikes_table['unit_id'] == unit_id) &
            (spikes_table['stimulus_presentation_id'] ==
             stimulus_presentation_id)]
        spike_times = spike_times['time_since_stimulus_presentation_onset']
        if metric_type == 'count':
          metric_value = len(spike_times) if len(spike_times) != 0 else empty_fill
        elif metric_type == 'shift':
          metric_value = np.mean(spike_times) if len(spike_times) != 0 else empty_fill

        # The spike train is special, since DataFrame does not take array as the
        # entry, we have to use a separate data structure to store the spike
        # trains. The metric table is used to store the index mapping.
        elif metric_type == 'spike_trains':
          metric_value = np.histogram(spike_times, time_bins)[0]
        elif metric_type == 'spike_times':
          metric_value = np.array(spike_times)
        else:
          raise TypeError('Wrong type of metric')
        metric_table.loc[unit_id, stimulus_presentation_id] = metric_value
    # Very important step, change the datatype to numeric, otherwise functions
    # like correlation cannot be performed.
    if metric_type not in ['spike_trains', 'spike_times']:
      metric_table = metric_table.apply(pd.to_numeric, errors='coerce')
    return metric_table


  def plot_condition_metrics(
      self,
      metric_df,
      stimulus_table,
      stimulus_type='drifting_gratings',
      output_figure_path=None,
      show_plot=True):
    """Explores the metrics for each condition.

    Args:
      metric_df: neuron x trials pd.DataFrame.
      stimulus_type: 'drifting_gratings', 'drifting_gratings_75_repeats',
                     'drifting_gratings_contrast', 'dot_motion'
    """
    active_units_quantile_threshold = 0.8
    trials_groups = stimulus_table.groupby('stimulus_condition_id')
    units = metric_df.index.values
    condition_metric = pd.DataFrame(index=trials_groups.groups.keys(),
                                    columns=['count'])
    condition_metric.index.name = 'stimulus_condition_id'

    for condition_id, trial_df in trials_groups:
      trials_indices = trial_df.index.values
      sub_metric_df = metric_df.loc[:, trials_indices]
      mean_fr = sub_metric_df.mean(axis=1)
      fr_threshold = mean_fr.quantile(active_units_quantile_threshold)
      metric = mean_fr[mean_fr > fr_threshold].mean()
      condition_metric.loc[condition_id, 'count'] = metric

    #-------------------- drifting gratings ------------------
    if stimulus_type == 'drifting_gratings':
      # Baseline.
      baseline_df = stimulus_table[
          (stimulus_table['orientation'] == 'null') &
          (stimulus_table['temporal_frequency'] == 'null')]
      baseline_condition_id = baseline_df['stimulus_condition_id'].unique()[0]
      baseline_metric = condition_metric.loc[baseline_condition_id, 'count']

      # All other parameters.
      orientation = stimulus_table[
          stimulus_table['orientation'] != 'null']['orientation'].unique()
      orientation = np.sort(orientation)
      temporal_frequency = stimulus_table[
          stimulus_table['temporal_frequency'] != 'null'
          ]['temporal_frequency'].unique()
      temporal_frequency = np.sort(temporal_frequency)

      plt.figure(figsize=(6, 4))
      for i, o in enumerate(orientation):
        for j, f in enumerate(temporal_frequency):
          df = stimulus_table[(stimulus_table['orientation'] == o) &
                              (stimulus_table['temporal_frequency'] == f)]
          if len(df) == 0 or o == 'null' or f == 'null':
            continue
          condition_id = df['stimulus_condition_id'].unique()[0]
          metric_norm = condition_metric.loc[condition_id, 'count'] - baseline_metric
          metric_norm = 0 if metric_norm < 0 else metric_norm
          plt.scatter(i, j, s=metric_norm * 100, c='tab:blue')

      plt.xticks(np.arange(len(orientation)), orientation)
      plt.yticks(np.arange(len(temporal_frequency)), temporal_frequency)
      if output_figure_path:
        plt.savefig(output_figure_path, bbox_inches='tight')
        print('Save figure to: ', output_figure_path)
      if show_plot:
        plt.show()
      plt.close()

    #-------------------- drifting gratings contrast ------------------
    elif stimulus_type == 'drifting_gratings_contrast':
      # All other parameters.
      orientation = stimulus_table['orientation'].unique()
      orientation = np.sort(orientation)
      temporal_frequency = stimulus_table['temporal_frequency'].unique()
      temporal_frequency = np.sort(temporal_frequency)
      contrast = stimulus_table['contrast'].unique()
      contrast = np.sort(contrast)

      metric_min = condition_metric['count'].min()
      plt.figure(figsize=(8, 4))
      for i, o in enumerate(orientation):
        for j, c in enumerate(contrast):
          df = stimulus_table[(stimulus_table['orientation'] == o) &
                              (stimulus_table['contrast'] == c)]
          condition_id = df['stimulus_condition_id'].unique()[0]
          metric_norm = condition_metric.loc[condition_id, 'count'] - metric_min
          metric_norm = metric_norm + 0.1
          plt.scatter(j, i, s=metric_norm * 400, c='tab:blue')

      plt.yticks(np.arange(len(orientation)), orientation)
      plt.xticks(np.arange(len(contrast)), contrast)
      if output_figure_path:
        plt.savefig(output_figure_path, bbox_inches='tight')
        print('Save figure to: ', output_figure_path)
      if show_plot:
        plt.show()
      plt.close()

    #-------------------- dot motion ------------------
    elif stimulus_type == 'dot_motion':
      # All other parameters.
      direction = stimulus_table['Dir'].unique()
      direction = np.sort(direction)
      speed = stimulus_table['Speed'].unique()
      speed = np.sort(speed)

      metric_min = condition_metric['count'].min()
      plt.figure(figsize=(8, 4))
      for i, d in enumerate(direction):
        for j, s in enumerate(speed):
          df = stimulus_table[(stimulus_table['Dir'] == d) &
                              (stimulus_table['Speed'] == s)]
          condition_id = df['stimulus_condition_id'].unique()[0]
          metric_norm = condition_metric.loc[condition_id, 'count'] - metric_min
          metric_norm = metric_norm + 0.1
          plt.scatter(j, i, s=metric_norm * 400, c='tab:blue')

      plt.yticks(np.arange(len(direction)), direction)
      plt.xticks(np.arange(len(speed)), speed)
      if output_figure_path:
        plt.savefig(output_figure_path, bbox_inches='tight')
        print('Save figure to: ', output_figure_path)
      if show_plot:
        plt.show()
      plt.close()

    #-------------------- others ------------------
    else:
      print(condition_metric)


  def extract_spike_trains_per_area(
      self,
      selected_units,
      spike_trains,
      trials_groups,
      spike_train_time_line,
      dt):
    """Plots the spike trian for each condition.

    Args:
      spike_trains: neuron x trials pd.DataFrame.
    """
    active_units_quantile_threshold = 0.5
    map_probe_to_area = self.session.map_probe_to_ecephys_structure_acronym(
        visual_only=True)

    units = spike_trains.index.values
    probes = selected_units['probe_description'].unique()
    # condition_data = pd.DataFrame(index=trials_groups.groups.keys(),
    #                               columns=probes)
    # condition_data.index.name = 'stimulus_condition_id'
    all_trials = [trial_df.index.values for (_, trial_df) in trials_groups]
    all_trials = np.hstack(all_trials)
    data = pd.DataFrame(index=all_trials, columns=probes)
    data.index.name = 'trial_id'

    num_probes = len(probes)
    num_conditions = len(trials_groups)

    # fr_threshold = mean_fr.quantile(active_units_quantile_threshold)
    # active_units = mean_fr[mean_fr > fr_threshold].index.values

    for r, trial_id in enumerate(all_trials):
      for a, probe in enumerate(probes):
        probe_units = selected_units[
            selected_units['probe_description'] == probe].index.values
        # mean_fr = spike_trains.loc[probe_units, trial_id].applymap(
        #     lambda x: np.sum(x))
        train = spike_trains.loc[probe_units, trial_id]
        train = train.values.reshape(-1)
        train = np.stack(train)
        train = train.mean(axis=0)
        data.loc[trial_id, probe] = train / dt

    return data


  def plot_spike_trains_per_condition(
      self,
      selected_units,
      spike_trains,
      spike_counts,
      stimulus_table,
      spike_train_time_line,
      dt,
      stimulus_type='drifting_gratings',
      output_figure_path=None,
      show_plot=True):
    """Plots the spike trian for each condition.

    Args:
      spike_trains: neuron x trials pd.DataFrame.
    """
    active_units_quantile_threshold = 0.5
    map_probe_to_area = self.session.map_probe_to_ecephys_structure_acronym(
        visual_only=True)

    if spike_trains.shape != spike_counts.shape:
      raise ValueError('spike_trains spike_counts not match')

    trials_groups = stimulus_table.groupby('stimulus_condition_id')
    units = spike_trains.index.values
    probes = selected_units['probe_description'].unique()
    condition_data = pd.DataFrame(index=trials_groups.groups.keys(),
                                  columns=probes)
    condition_data.index.name = 'stimulus_condition_id'
    num_probes = len(probes)
    num_conditions = len(trials_groups)
    num_active_neurons = {}

    for c, (condition_id, trial_df) in enumerate(trials_groups):
      trials_indices = trial_df.index.values

      for a, probe in enumerate(probes):
        probe_units = selected_units[
            selected_units['probe_description'] == probe].index.values
        # Select the most active neurons.
        mean_fr = spike_counts.loc[probe_units, trials_indices].mean(axis=1)
        fr_threshold = mean_fr.quantile(active_units_quantile_threshold)
        active_units = mean_fr[mean_fr > fr_threshold].index.values
        train = spike_trains.loc[active_units, trials_indices]
        train = train.values.reshape(-1)
        train = np.stack(train)
        train = train.mean(axis=0)
        condition_data.loc[condition_id, probe] = train
        num_active_neurons[(condition_id, probe)] = len(active_units)

    #------------------------ drifting_gratings ------------------------
    if stimulus_type == 'drifting_gratings':
      # Baseline.
      baseline_df = stimulus_table[
          (stimulus_table['orientation'] == 'null') &
          (stimulus_table['temporal_frequency'] == 'null')]
      baseline_condition_id = baseline_df['stimulus_condition_id'].unique()[0]
      baseline_spike_train = condition_data.loc[baseline_condition_id, :]

      # All other parameters.
      orientation = stimulus_table[
          stimulus_table['orientation'] != 'null']['orientation'].unique()
      orientation = np.sort(orientation)
      temporal_frequency = stimulus_table[
          stimulus_table['temporal_frequency'] != 'null'
          ]['temporal_frequency'].unique()
      temporal_frequency = np.sort(temporal_frequency)

      plt.figure(figsize=(len(probes) * 5,
                 len(orientation) * len(temporal_frequency) * 2.6))
      for i, o in enumerate(orientation):
        for j, f in enumerate(temporal_frequency):

          df = stimulus_table[(stimulus_table['orientation'] == o) &
                              (stimulus_table['temporal_frequency'] == f)]
          if len(df) == 0 or o == 'null' or f == 'null':
            continue
          condition_id = df['stimulus_condition_id'].unique()[0]
          spike_train = condition_data.loc[condition_id, :]
          row_id = i * len(temporal_frequency) + j
          for a, probe in enumerate(probes):
            ax = plt.subplot(len(trials_groups), len(probes),
                             row_id*len(probes) + a+1)
            plt.plot(spike_train_time_line,
                     baseline_spike_train.loc[probe] / dt, 'tab:gray')
            plt.plot(spike_train_time_line, spike_train.loc[probe] / dt)
            plt.title(f'{probe} {map_probe_to_area[probe]} ' +
                      f'n={num_active_neurons[(condition_id, probe)]}',
                      fontdict = {'fontsize' : 8})
            if a == 0:
              plt.title(f'O={o}  F={f} C={condition_id}  ' +
                        f'{probe} {map_probe_to_area[probe]} ' +
                        f'n={num_active_neurons[(condition_id, probe)]}',
                        fontdict = {'fontsize' : 8})
              plt.ylabel('Firing rate [Hz]')
              plt.xlabel('Time [sec]')

            # if probe == 'probeC':
            #   plt.title(f'O={o}  F={f} C={condition_id} {probe}  n:{}')
            #   plt.ylim(0, 0.1)
            # elif probe == 'probeD':
            #   plt.ylim(0, 0.06)
            # elif probe == 'probeE':
            #   plt.ylim(0, 0.15)

      plt.tight_layout()
      if output_figure_path:
        plt.savefig(output_figure_path, bbox_inches='tight')
        print('Save figure to: ', output_figure_path)
      if show_plot:
        plt.show()
      else:
        plt.close()

    #------------------------ flashes ------------------------
    elif stimulus_type == 'flashes':
      colors = [-1, 1]
      gs_kw = dict(height_ratios=[1,1], width_ratios=[1]*len(probes))
      fig, axs = plt.subplots(figsize=(20, len(probes)*4), gridspec_kw=gs_kw,
          nrows=2, ncols=len(probes))
      plt.subplots_adjust(hspace=0.4)

      for i, color in enumerate(colors):
        df = stimulus_table[(stimulus_table['color'] == color)]
        condition_id = df['stimulus_condition_id'].unique()[0]
        spike_train = condition_data.loc[condition_id, :]
        for a, probe in enumerate(probes):
          ax = fig.add_subplot(axs[i, a])
          plt.plot(spike_train_time_line, spike_train.loc[probe] / dt, 'k')
          plt.title(f'{probe} {map_probe_to_area[probe]} ' +
                    f'n={num_active_neurons[(condition_id, probe)]}',
                    fontdict = {'fontsize' : 12})
          if a == 0:
            plt.title(f'color={color}  ' +
                      f'{probe} {map_probe_to_area[probe]} ' +
                      f'n={num_active_neurons[(condition_id, probe)]}',
                      fontdict = {'fontsize' : 12})
            plt.ylabel('Firing rate [Hz]')
            plt.xlabel('Time [sec]')

      if output_figure_path:
        plt.savefig(output_figure_path, bbox_inches='tight')
        print('Save figure to: ', output_figure_path)
      if show_plot:
        plt.show()
      plt.close()

    #------------------------ drifting_gratings_75_repeats ------------------------
    elif stimulus_type == 'drifting_gratings_75_repeats':
      # All other parameters.
      orientation = stimulus_table[
          stimulus_table['orientation'] != 'null']['orientation'].unique()
      orientation = np.sort(orientation)
      temporal_frequency = stimulus_table['temporal_frequency'].unique()
      temporal_frequency = np.sort(temporal_frequency)
      if len(temporal_frequency) > 1:
        raise ValueError(f'temporal_frequency is not unique {temporal_frequency}')
      temporal_frequency = temporal_frequency[0]
      contrast = stimulus_table['contrast'].unique()
      contrast = np.sort(contrast)

      gs_kw = dict(width_ratios=[1]*len(probes))
      fig, axs = plt.subplots(
          figsize=[len(probes)*6, len(orientation)*len(contrast)*3],
          nrows=len(orientation)*len(contrast), ncols=len(probes),
          gridspec_kw=gs_kw)
      plt.subplots_adjust(hspace=0.6)

      for i, cnst in enumerate(contrast):
        for j, ornt in enumerate(orientation):
          df = stimulus_table[
              (stimulus_table['contrast'] == cnst) &
              (stimulus_table['orientation'] == ornt) &
              (stimulus_table['temporal_frequency'] == temporal_frequency)]
          condition_id = df['stimulus_condition_id'].unique()[0]
          spike_train = condition_data.loc[condition_id, :]
          for a, probe in enumerate(probes):
            ax = fig.add_subplot(axs[i*len(orientation)+j, a])
            plt.plot(spike_train_time_line, spike_train.loc[probe] / dt, 'k')
            plt.title(f'{probe} {map_probe_to_area[probe]} ' +
                      f'n={num_active_neurons[(condition_id, probe)]}',
                      fontdict = {'fontsize' : 12})
            if a == 0:
              plt.title(f'{probe} {map_probe_to_area[probe]} ' +
                        f'ornt={ornt}  ' +
                        f'cnst={cnst}  ' +
                        f'tmpf={temporal_frequency}  ' +
                        f'n={num_active_neurons[(condition_id, probe)]}',
                        fontdict = {'fontsize' : 12})
              plt.ylabel('Firing rate [Hz]')
              plt.xlabel('Time [sec]')

      if output_figure_path:
        plt.savefig(output_figure_path, bbox_inches='tight')
        print('Save figure to: ', output_figure_path)
      if show_plot:
        plt.show()
      plt.close()

    #------------------------ drifting_gratings_contrast ------------------------
    elif stimulus_type == 'drifting_gratings_contrast':
      # All other parameters.
      orientation = stimulus_table['orientation'].unique()
      orientation = np.sort(orientation)
      temporal_frequency = stimulus_table['temporal_frequency'].unique()
      temporal_frequency = np.sort(temporal_frequency)
      if len(temporal_frequency) > 1:
        raise ValueError(f'temporal_frequency is not unique {temporal_frequency}')
      temporal_frequency = temporal_frequency[0]
      contrast = stimulus_table['contrast'].unique()
      contrast = np.sort(contrast)

      gs_kw = dict(width_ratios=[1]*len(probes))
      fig, axs = plt.subplots(
          figsize=[len(probes)*6, len(orientation)*len(contrast)*3],
          nrows=len(orientation)*len(contrast), ncols=len(probes),
          gridspec_kw=gs_kw)
      plt.subplots_adjust(hspace=0.6)

      for i, cnst in enumerate(contrast):
        for j, ornt in enumerate(orientation):
          df = stimulus_table[
              (stimulus_table['contrast'] == cnst) &
              (stimulus_table['orientation'] == ornt) &
              (stimulus_table['temporal_frequency'] == temporal_frequency)]
          condition_id = df['stimulus_condition_id'].unique()[0]
          spike_train = condition_data.loc[condition_id, :]
          for a, probe in enumerate(probes):
            ax = fig.add_subplot(axs[i*len(orientation)+j, a])
            plt.plot(spike_train_time_line, spike_train.loc[probe] / dt, 'k')
            plt.title(f'{probe} {map_probe_to_area[probe]} ' +
                      f'n={num_active_neurons[(condition_id, probe)]}',
                      fontdict = {'fontsize' : 12})
            if a == 0:
              plt.title(f'{probe} {map_probe_to_area[probe]} ' +
                        f'ornt={ornt}  ' +
                        f'cnst={cnst}  ' +
                        f'tmpf={temporal_frequency}  ' +
                        f'n={num_active_neurons[(condition_id, probe)]}',
                        fontdict = {'fontsize' : 12})
              plt.ylabel('Firing rate [Hz]')
              plt.xlabel('Time [sec]')

      if output_figure_path:
        plt.savefig(output_figure_path, bbox_inches='tight')
        print('Save figure to: ', output_figure_path)
      if show_plot:
        plt.show()
      plt.close()

    #------------------------ dot_motion ------------------------
    elif stimulus_type == 'dot_motion':
      # All other parameters.
      direction = stimulus_table['Dir'].unique()
      direction = np.sort(direction)
      speed = stimulus_table['Speed'].unique()
      speed = np.sort(speed)

      gs_kw = dict(width_ratios=[1]*len(probes))
      fig, axs = plt.subplots(
          figsize=[len(probes)*6, len(direction)*len(speed)*3],
          nrows=len(direction)*len(speed), ncols=len(probes),
          gridspec_kw=gs_kw)
      plt.subplots_adjust(hspace=0.6)

      for i, spd in enumerate(speed):
        for j, dr in enumerate(direction):
          df = stimulus_table[
              (stimulus_table['Dir'] == dr) &
              (stimulus_table['Speed'] == spd)]
          condition_id = df['stimulus_condition_id'].unique()[0]
          spike_train = condition_data.loc[condition_id, :]
          for a, probe in enumerate(probes):
            ax = fig.add_subplot(axs[i*len(direction)+j, a])
            plt.plot(spike_train_time_line, spike_train.loc[probe] / dt, 'k')
            plt.title(f'{probe} {map_probe_to_area[probe]} ' +
                      f'n={num_active_neurons[(condition_id, probe)]}',
                      fontdict = {'fontsize' : 12})
            if a == 0:
              plt.title(f'{probe} {map_probe_to_area[probe]} ' +
                        f'speed={spd}  ' +
                        f'dir={dr}  ' +
                        f'n={num_active_neurons[(condition_id, probe)]}',
                        fontdict = {'fontsize' : 12})
              plt.ylabel('Firing rate [Hz]')
              plt.xlabel('Time [sec]')

      if output_figure_path:
        plt.savefig(output_figure_path, bbox_inches='tight')
        print('Save figure to: ', output_figure_path)
      if show_plot:
        plt.show()
      plt.close()

    else:
      print(f'New stimulus type  {stimulus_type}.')


  def total_variance_per_condition(
      self,
      selected_units,
      spike_trains,
      spike_counts,
      trials_groups,
      spike_train_time_line,
      dt,
      kernel_par=5,
      show_figure=True):
    """Plots the spike trian for each condition.

    Args:
      spike_trains: neuron x trials pd.DataFrame.
    """
    if spike_trains.shape != spike_counts.shape:
      raise ValueError('spike_trains spike_counts not match')

    active_units_quantile_threshold = 0.5
    map_probe_to_area = self.session.map_probe_to_ecephys_structure_acronym(
        visual_only=True)
    condition_ids = trials_groups.groups.keys()
    units = spike_trains.index.values
    probes = selected_units['probe_description'].unique()
    num_probes = len(probes)
    num_conditions = len(condition_ids)

    condition_data = pd.DataFrame(index=condition_ids, columns=probes)
    condition_data.index.name = 'stimulus_condition_id'
    num_active_neurons = condition_data.copy()
    total_variance = condition_data.copy()

    for c, (condition_id, trial_df) in enumerate(trials_groups):
      trials_indices = trial_df.index.values
      orientation = trial_df.iloc[0]['orientation']
      temporal_frequency = trial_df.iloc[0]['temporal_frequency']
      if orientation == 'null' and temporal_frequency == 'null':
        baseline_condition_id = condition_id

      for a, probe in enumerate(probes):
        # Understanding gaussian_filter1d.
        # The following calculation is equivalent.
        # import scipy
        # from scipy.ndimage import gaussian_filter1d
        # t = np.arange(100)
        # x = np.zeros(100); x[50] = 1
        # y = gaussian_filter1d(x, 10)
        # z = scipy.stats.norm.pdf(t, loc=50, scale=10)

        probe_units = selected_units[
            selected_units['probe_description'] == probe].index.values
        # Select the most active neurons.
        mean_fr = spike_counts.loc[probe_units, trials_indices].mean(axis=1)
        fr_threshold = mean_fr.quantile(active_units_quantile_threshold)
        active_units = mean_fr[mean_fr > fr_threshold].index.values
        train = spike_trains.loc[active_units, trials_indices]
        train = train.values.reshape(-1)
        train = np.stack(train)
        psth = train.mean(axis=0) / dt
        psth = scipy.ndimage.gaussian_filter1d(psth, kernel_par)
        # psth = psth - np.mean(psth)
        # psth = psth / np.std(psth)
        tv = np.sum(np.abs(np.diff(psth)))
        total_variance.loc[condition_id, probe] = tv
        condition_data.loc[condition_id, probe] = psth
        num_active_neurons.loc[condition_id, probe] = len(active_units)

    if not show_figure:
      return total_variance

    # Baseline.
    baseline_spike_train = condition_data.loc[baseline_condition_id, :]

    gs_kw = dict(height_ratios=[1]*num_conditions, width_ratios=[1]*num_probes)
    fig, axs = plt.subplots(figsize=(num_probes*5, num_conditions*2.5),
        gridspec_kw=gs_kw, nrows=num_conditions, ncols=num_probes)
    plt.subplots_adjust(left=None, right=None, hspace=0.3, wspace=0.3)
    for c, condition_id in enumerate(condition_ids):
      for a, probe in enumerate(probes):
        psth = condition_data.loc[condition_id,probe]

        ax = fig.add_subplot(axs[c, a])
        plt.plot(spike_train_time_line,
                 baseline_spike_train.loc[probe], 'tab:gray')
        plt.plot(spike_train_time_line, psth)

        plt.text(0.05, 0.9, f'{total_variance.loc[condition_id, probe]:.1f}',
            color='r', size=10, transform=ax.transAxes)
        plt.text(0.2, 0.9, f'{total_variance.loc[condition_id, :].sum():.1f}',
            color='g', size=10, transform=ax.transAxes)

        ax.tick_params(labelbottom=False)
        if a == 0:
          plt.text(0.5, 0.9, f'C={condition_id}  ' +
             f'{probe} {map_probe_to_area[probe]} ' +
             f'n={num_active_neurons.loc[condition_id, probe]}',
              color='k', size=8, transform=ax.transAxes)
          plt.ylabel('Firing rate [Hz]')
          # plt.xlabel('Time [sec]')
        else:
          plt.text(0.6, 0.9, f'{probe} {map_probe_to_area[probe]} ' +
              f'n={num_active_neurons.loc[condition_id, probe]}',
              color='k', size=8, transform=ax.transAxes)

    return total_variance


  def plot_all_spike_trains_per_area(
      self,
      selected_units,
      spike_trains,
      spike_counts,
      spike_train_time_line,
      dt,
      trials_indices=None,
      baseline_trials_indices=None,
      output_figure_path=None,
      show_plot=True):
    """Explores the metrics for each area.

    Args:
      spike_trains: neuron x trials pd.DataFrame.
    """
    areas_names = ['V1', 'LM', 'AL']
    active_units_quantile_threshold = 0.5
    map_probe_to_area = self.session.map_probe_to_ecephys_structure_acronym(
        visual_only=True)

    if spike_trains.shape != spike_counts.shape:
      raise ValueError(f'spike_trains spike_counts not match ' + 
          f'spike_trains.shape {spike_trains.shape}  ' + 
          f'spike_counts.shape{spike_counts.shape}')
    probes = selected_units['probe_description'].unique()
    if trials_indices is None:
      trials_indices = spike_counts.columns.values

    num_bins = len(spike_train_time_line)
    num_probes = len(probes)

    task_psth = np.zeros([num_probes, num_bins])
    task_psth_smooth = np.zeros([num_probes, num_bins])
    baseline_psth = np.zeros([num_probes, num_bins])
    baseline_psth_smooth = np.zeros([num_probes, num_bins])
    num_active_neurons = {}

    for a, probe in enumerate(probes):
      probe_units = selected_units[
          selected_units['probe_description'] == probe].index.values
      # Select the most active neurons.
      mean_fr = spike_counts.loc[probe_units, trials_indices].mean(axis=1)
      fr_threshold = mean_fr.quantile(active_units_quantile_threshold)
      active_units = mean_fr[mean_fr > fr_threshold].index.values
      # print(f'{probe}  num neurons:{len(active_units)}')
      num_active_neurons[probe] = len(active_units)
      train = spike_trains.loc[active_units, trials_indices]
      train = train.values.reshape(-1)
      train = np.stack(train)
      task_psth[a] = train.mean(axis=0) / dt

      fit_model = smoothing_spline.SmoothingSpline()
      num_trains, num_bins = train.shape
      # print(f'Data size: {train.shape}')
      # Maual knots spline fitting.
      knots = [0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.2, 0.22, 0.25, 0.27,
               0.3, 0.35, 0.4, 0.45]
      basis, _ = fit_model.bspline_basis(
          spline_order=4, knots=knots,
          knots_range=[spike_train_time_line[0], spike_train_time_line[-1]],
          sample_points=spike_train_time_line, show_plot=False)
      log_lmbda_hat, par = fit_model.poisson_regression_smoothing_spline(
          train, spike_train_time_line, basis=basis, lambda_tuning=0, verbose=0)
      # Smoothing splines.
      # eta_smoothing = 2e-5 / num_trains
      # log_lmbda_hat, par = fit_model.poisson_regression_smoothing_spline(
      #     train, spike_train_time_line, lambda_tuning=eta_smoothing, verbose=0)
      task_psth_smooth[a] = np.exp(log_lmbda_hat) / dt

      if baseline_trials_indices is not None:
        baseline_train = spike_trains.loc[active_units,
                                          baseline_trials_indices]
        baseline_train = baseline_train.values.reshape(-1)
        baseline_train = np.stack(baseline_train)
        baseline_psth[a] = baseline_train.mean(axis=0) / dt

        num_trains_baseline, num_bins_baseline = train.shape
        eta_smoothing = 5e-4 / num_trains_baseline
        log_lmbda_hat, par = fit_model.poisson_regression_smoothing_spline(
          baseline_train, spike_train_time_line, lambda_tuning=eta_smoothing, verbose=0)
        baseline_psth_smooth[a] = np.exp(log_lmbda_hat) / dt

    # Plot together
    if show_plot:
      plt.figure(figsize=(5, 4))
      # for a, probe in enumerate(probes):
      #   plt.plot(spike_train_time_line / dt, task_psth_smooth[a], label=areas_names[a])
      plt.plot(spike_train_time_line * 1000, task_psth_smooth[0],
          'k', label=areas_names[0])
      plt.plot(spike_train_time_line * 1000, task_psth_smooth[1],
          '--k', label=areas_names[1])
      plt.plot(spike_train_time_line * 1000, task_psth_smooth[2],
          ':k', label=areas_names[2])

      plt.ylim(bottom=0)
      plt.ylim(0, 25)
      plt.ylabel('Firing rate [spikes/sec]', fontsize=16)
      plt.xlabel('Time [ms]', fontsize=16)
      plt.legend(fontsize=12)
      plt.tight_layout()
      # if output_dir:
      #   filename = os.path.join(output_dir,
      #       f'{self.session.ecephys_session_id}_50prct_psth_together.pdf')
      #   plt.savefig(filename)
      #   print('Save figure to: ', filename)
      plt.show()

    # Plot separately
    plt.figure(figsize=(num_probes*5, 8))
    for a, probe in enumerate(probes):
      ax = plt.subplot(2, len(probes), a+1)
      # The order of plotting does matter, they are in different layers.
      if baseline_trials_indices is not None:
        plt.plot(spike_train_time_line, baseline_psth[a], 'lightgrey')
      plt.plot(spike_train_time_line, task_psth[a], 'lightblue')

      if baseline_trials_indices is not None:
        plt.plot(spike_train_time_line, baseline_psth_smooth[a], 'k', label='Baseline')
      plt.plot(spike_train_time_line, task_psth_smooth[a], 'b', label='Task')
      plt.ylim(bottom=0)
      plt.ylim(0, 40)
      if a == 0:
        plt.ylabel('Firing rate [Hz]')
        plt.legend()
      plt.xlabel('Time [sec]')
      # plt.title(areas_names[a])
      plt.title(f'{probe} {map_probe_to_area[probe]} n={num_active_neurons[probe]}')
      # plt.grid()

    # Plot together
    ax = plt.subplot(2, len(probes), len(probes)+1)
    for a, probe in enumerate(probes):
      plt.plot(spike_train_time_line, task_psth_smooth[a], label=probe)
      # plt.ylim(bottom=0)
      plt.ylim(0, 25)
      plt.ylabel('Firing rate [Hz]')
      plt.xlabel('Time [sec]')
      plt.legend(loc='upper right')
      # plt.grid()
    plt.tight_layout()
    if output_figure_path:
      plt.savefig(output_figure_path, bbox_inches='tight')
      print('Save figure to: ', output_figure_path)
    if show_plot:
      plt.show()
    plt.close()

    # Plot together
    plt.figure(figsize=(5, 3))
    for a, probe in enumerate(probes):
      plt.plot(spike_train_time_line, task_psth_smooth[a], label=areas_names[a])
      # plt.ylim(bottom=0)
      plt.ylim(3, 22)
      plt.ylabel('Firing rate [Hz]', fontsize=12)
      plt.xlabel('Time [sec]', fontsize=12)
      plt.legend(loc='upper right')
      # plt.grid()
    plt.tight_layout()
    # if output_figure_path:
    #   plt.savefig(output_figure_path, bbox_inches='tight')
    #   print('Save figure to: ', output_figure_path)
    # if show_plot:
    #   plt.show()
    # plt.close()



  def seperate_probe_pairs_units_by_corr_matrix(
      self,
      corr_matrix,
      units_probes,
      corr_threshold=0.2,
      quantile=0.3):
    """Seperate the units for each pair of probes by the correlation matrix."""
    probes_category = units_probes.unique()

    unit_ids = []
    probe_from = []
    probe_to = []
    group_id = []

    for probe0, probe1 in itertools.combinations(probes_category, 2):
      probe_units0 = units_probes[units_probes == probe0].index.values
      probe_units1 = units_probes[units_probes == probe1].index.values
      sub_corr_matrix = corr_matrix.loc[probe_units0, probe_units1]
      sub_corr_matrix = sub_corr_matrix[sub_corr_matrix > corr_threshold]
      # Sum the correlation from one area to the other.
      sub_units0 = sub_corr_matrix.sum(axis=1)
      sub_units1 = sub_corr_matrix.sum(axis=0)
      count_threshold0 = np.quantile(sub_units0, quantile)
      count_threshold1 = np.quantile(sub_units1, quantile)
      sub_units0 = sub_units0[sub_units0 > count_threshold0].index.values
      sub_units1 = sub_units1[sub_units1 > count_threshold1].index.values
      sub_units0c = np.setdiff1d(probe_units0, sub_units0)
      sub_units1c = np.setdiff1d(probe_units1, sub_units1)
      sub_corr_matrix = sub_corr_matrix.loc[sub_units0, sub_units1]

      # units in axis 0 group_id
      unit_ids.append(sub_units0)
      probe_from.append([probe0] * len(sub_units0))
      probe_to.append([probe1] * len(sub_units0))
      group_id.append([0] * len(sub_units0))
      # units in axis 0 excluded
      unit_ids.append(sub_units0c)
      probe_from.append([probe0] * len(sub_units0c))
      probe_to.append([probe1] * len(sub_units0c))
      group_id.append([1] * len(sub_units0c))
      # units in axis 1 group_id
      unit_ids.append(sub_units1)
      probe_from.append([probe1] * len(sub_units1))
      probe_to.append([probe0] * len(sub_units1))
      group_id.append([0] * len(sub_units1))
      # units in axis 1 excluded
      unit_ids.append(sub_units1c)
      probe_from.append([probe1] * len(sub_units1c))
      probe_to.append([probe0] * len(sub_units1c))
      group_id.append([1] * len(sub_units1c))

    df = pd.DataFrame({
        'unit_ids': np.concatenate(unit_ids).astype(int),
        'probe_from': np.concatenate(probe_from),
        'probe_to': np.concatenate(probe_to),
        'group_id': np.concatenate(group_id)
    })
    df.set_index('unit_ids', inplace=True)

    return df

  def seperate_probe_pairs_units_by_psth_match_scores(
      self,
      unit_ids,
      psth_dict,
      score_quantile_threshold=0.6):
    """Seperate the units for each pair of probes by the psth match score."""
    units = self.session._filter_owned_df('units', ids=unit_ids)
    # probes_string = units['probe_description']
    # probes_category = probes_string.unique()

    units_list = []
    probe_from = []
    probe_to = []
    group_ids = []

    probe_pairs = psth_dict.keys()
    # Match the units in prob0 to the population PSTH in probe1.
    for probe0, probe1 in probe_pairs:
      # probe0 = 'probeC'
      # probe1 = 'probeE'
      print(probe0, probe1)
      group_index = 0
      probe_units0 = units[units['probe_description'] == probe0].index.values

      psth_data = psth_dict[(probe1, probe0)]
      # The PSTH for each trial come from the pair.
      metric_table = self.get_spikes_psth_match_scores(
          stimulus_presentation_ids=psth_data['stimulus_presentation_ids'],
          unit_ids=probe_units0,
          psth_trial_df=psth_data[group_index],
          trial_time_window=psth_data['trial_time_window'],
          dt=psth_data['dt'])

      metric_table = metric_table.sum(axis=1)  # Sum over trials axis.
      matched_units = metric_table[
          metric_table > metric_table.quantile(q=score_quantile_threshold)]
      matched_units = matched_units.index.values
      print('psth_data:', psth_data['unit_ids'])
      print('matched_units:', matched_units)

      # units in axis 0 group_ids. Matched group.
      units_list.append(matched_units)
      probe_from.append([probe0] * len(matched_units))
      probe_to.append([probe1] * len(matched_units))
      group_ids.append([0] * len(matched_units))
      # units in axis 0 excluded. Unmatched group.
      sub_units0c = np.setdiff1d(probe_units0, matched_units)
      units_list.append(sub_units0c)
      probe_from.append([probe0] * len(sub_units0c))
      probe_to.append([probe1] * len(sub_units0c))
      group_ids.append([1] * len(sub_units0c))

    df = pd.DataFrame({
        'unit_ids': np.concatenate(units_list).astype(int),
        'probe_from': np.concatenate(probe_from),
        'probe_to': np.concatenate(probe_to),
        'group_id': np.concatenate(group_ids)
    })

    return df

  def get_group_activity_per_trial(
      self,
      stimulus_presentation_ids,
      unit_ids,
      trial_time_window,
      metric_type,
      dt=0.005,
      smooth_sigma=None,
      verbose=False):
    """Get the PSTH of a  group for each trial.

    Args:
      metric_type:
          'count': total number of spikes of all neurons for each trial.
          'psth': The psth of all neurons for each trial.
          'autocorr': autocorrelation of the PSTH of all neurons for each trial.
    """
    spikes_table = self.session.trialwise_spike_times(
        stimulus_presentation_ids, unit_ids, trial_time_window)
    time_bins = np.linspace(
        trial_time_window[0], trial_time_window[1],
        int((trial_time_window[1] - trial_time_window[0]) / dt) + 1)
    if metric_type == 'count':
      result_table = pd.DataFrame(index=stimulus_presentation_ids, 
                                  columns=['spike_count'])
    elif metric_type == 'psth':
      result_table = pd.DataFrame(index=stimulus_presentation_ids, 
                                  columns=time_bins[:-1])
    # elif metric_type == 'intensity':
    #   # Start the Matlab engine if we need to use intensity estimated by BARS.
    #   self._start_matlab_engine()
    #   result_table = pd.DataFrame(index=stimulus_presentation_ids, 
    #                               columns=time_bins[:-1])
    elif metric_type == 'autocorr':
      lag_len = int((trial_time_window[1] - trial_time_window[0]) / dt / 2)
      lag_indices = np.arange(0, lag_len)
      result_table = pd.DataFrame(index=stimulus_presentation_ids,
                                  columns=lag_indices * dt)

    elif metric_type == 'shift':
      result_table = pd.DataFrame(index=stimulus_presentation_ids,
                                  columns=['trial_shift'])
    result_table.index.name = 'stimulus_presentations'

    for s, stimulus_presentation_id in enumerate(stimulus_presentation_ids):
      if verbose:
        print('Trial:', s, stimulus_presentation_id)
      spike_times = spikes_table[
          (spikes_table['stimulus_presentation_id'] ==
           stimulus_presentation_id)]
      spike_times = spike_times['time_since_stimulus_presentation_onset']

      if metric_type == 'count':
        result_table.loc[stimulus_presentation_id] = len(spike_times)

      elif metric_type == 'psth':
        psth = np.histogram(spike_times, time_bins)[0] * dt
        if smooth_sigma is not None and smooth_sigma > 0:
          psth = gaussian_filter1d(psth, smooth_sigma, axis=0)
        result_table.loc[stimulus_presentation_id] = psth

      # elif metric_type == 'intensity':
      #   bin_count = np.histogram(spike_times, time_bins)[0]
      #   if sum(bin_count) < 10:
      #     result_table.loc[stimulus_presentation_id] = bin_count * dt
      #     print('Warning: Too few spikes.')
      #     continue
      #   data = matlab.double(list(bin_count))
      #   data_range = matlab.double(list(trial_time_window))

      #   bars_model = self.matlab_engine.barsP(
      #       data, data_range, 1, stdout=self.stdout, stderr=self.stderr)
      #   print(self.stdout.getvalue())
      #   print(self.stderr.getvalue())
      #   intensity = np.array(bars_model['mean']).reshape(-1) * dt
      #   result_table.loc[stimulus_presentation_id] = intensity

      elif metric_type == 'autocorr':
        psth = np.histogram(spike_times, time_bins)[0] * dt
        if smooth_sigma is not None and smooth_sigma > 0:
          psth = gaussian_filter1d(psth, smooth_sigma, axis=0)
        autocorr, x = util.cross_corr(psth, psth,
                                      index_range=[0, lag_len], type='full')
        result_table.loc[stimulus_presentation_id] = autocorr

      elif metric_type == 'shift':
        result_table.loc[stimulus_presentation_id] = np.mean(spike_times)

    result_table = result_table.apply(pd.to_numeric, errors='coerce')
    return result_table

  def get_across_sub_areas_metrics_corr(
      self,
      sub_group_df,
      stimulus_presentation_ids,
      trial_time_window,
      metric_type,
      dt=0.1,
      output_figure_path=None,
      show_figure=True):
    """Plot between across areas correlation.

    Args:
      sub_group_df: It shows the correlated, uncorrelated neurons.
    """
    prob_index = {'probeA':1, 'probeB':2, 'probeC':3,
                  'probeD':4, 'probeE':5, 'probeF':6}
    probe_pairs = sub_group_df.groupby([
        'probe_from','probe_to']).size().reset_index()

    result_dict = defaultdict(lambda: defaultdict(list))

    for r in range(len(probe_pairs)):
      probe_from = probe_pairs.iloc[r]['probe_from']
      probe_to = probe_pairs.iloc[r]['probe_to']
      row_id = prob_index[probe_from]
      col_id = prob_index[probe_to]

      if col_id < row_id:
        continue
      print(probe_from, probe_to)

      sub_units0 = sub_group_df[
          (sub_group_df['probe_from'] == probe_from) &
          (sub_group_df['probe_to'] == probe_to) &
          (sub_group_df['group_id'] == 0)]['unit_ids'].values
      sub_units1 = sub_group_df[
          (sub_group_df['probe_from'] == probe_to) &
          (sub_group_df['probe_to'] == probe_from) &
          (sub_group_df['group_id'] == 0)]['unit_ids'].values
      sub_units0c = sub_group_df[
          (sub_group_df['probe_from'] == probe_from) &
          (sub_group_df['probe_to'] == probe_to) &
          (sub_group_df['group_id'] == 1)]['unit_ids'].values
      sub_units1c = sub_group_df[
          (sub_group_df['probe_from'] == probe_to) &
          (sub_group_df['probe_to'] == probe_from) &
          (sub_group_df['group_id'] == 1)]['unit_ids'].values

      result_table0 = self.get_group_activity_per_trial(
          stimulus_presentation_ids=stimulus_presentation_ids,
          unit_ids=sub_units0,
          trial_time_window=trial_time_window,
          metric_type=metric_type,
          dt=dt)

      result_table1 = self.get_group_activity_per_trial(
          stimulus_presentation_ids=stimulus_presentation_ids,
          unit_ids=sub_units1,
          trial_time_window=trial_time_window,
          metric_type=metric_type,
          dt=dt)

      result_table0c = self.get_group_activity_per_trial(
          stimulus_presentation_ids=stimulus_presentation_ids,
          unit_ids=sub_units0c,
          trial_time_window=trial_time_window,
          metric_type=metric_type,
          dt=dt)

      result_table1c = self.get_group_activity_per_trial(
          stimulus_presentation_ids=stimulus_presentation_ids,
          unit_ids=sub_units1c,
          trial_time_window=trial_time_window,
          metric_type=metric_type,
          dt=dt)

      features_corr = result_table0.corrwith(result_table1, axis = 0)
      features_corr_c = result_table0c.corrwith(result_table1c, axis = 0)

      result_dict[(probe_from, probe_to)]['corr'] = features_corr
      result_dict[(probe_from, probe_to)]['uncorr'] = features_corr_c

    return result_dict

  def get_across_sub_areas_metrics(
      self,
      sub_group_df,
      stimulus_presentation_ids,
      trial_time_window,
      metric_type,
      dt=0.1,
      smooth_sigma=None,
      output_figure_path=None,
      show_figure=True,
      verbose=False):
    """Plot between across areas correlation.

    Args:
      sub_group_df: It shows the correlated, uncorrelated neurons.
    """
    probe_pairs = sub_group_df.groupby([
        'probe_from','probe_to']).size().reset_index()

    result_dict = defaultdict(lambda: defaultdict(list))

    for r in range(len(probe_pairs)):
      probe_from = probe_pairs.iloc[r]['probe_from']
      probe_to = probe_pairs.iloc[r]['probe_to']

      if verbose:
        print(probe_from, probe_to)

      group_ids = sub_group_df[
          (sub_group_df['probe_from'] == probe_from) &
          (sub_group_df['probe_to'] == probe_to)]['group_id'].unique()
      for group_id in group_ids:
        sub_units = sub_group_df[
            (sub_group_df['probe_from'] == probe_from) &
            (sub_group_df['probe_to'] == probe_to) &
            (sub_group_df['group_id'] == group_id)].index.values

        result_table = self.get_group_activity_per_trial(
            stimulus_presentation_ids=stimulus_presentation_ids,
            unit_ids=sub_units,
            trial_time_window=trial_time_window,
            metric_type=metric_type,
            dt=dt,
            smooth_sigma=smooth_sigma)
        result_dict[(probe_from, probe_to)][group_id] = result_table
        result_dict[(probe_from, probe_to)]['unit_ids'] = sub_units
        result_dict[(probe_from, probe_to)]['dt'] = dt
        result_dict[(probe_from, probe_to)][
            'trial_time_window'] = trial_time_window
        result_dict[(probe_from, probe_to)][
            'stimulus_presentation_ids'] = stimulus_presentation_ids

    return result_dict

  def get_spikes_psth_match_scores(
      self,
      stimulus_presentation_ids,
      unit_ids,
      psth_trial_df,
      trial_time_window,
      dt):
    """Plots selected units."""
    stimulus_presentations = self.session._filter_owned_df(
        'stimulus_presentations', ids=stimulus_presentation_ids)
    units = self.session._filter_owned_df('units', ids=unit_ids)
    spikes_table = self.session.trialwise_spike_times(
        stimulus_presentation_ids, unit_ids, trial_time_window)
    num_neurons = len(unit_ids)
    num_trials = len(stimulus_presentation_ids)
    time_bins = np.linspace(
        trial_time_window[0], trial_time_window[1],
        int((trial_time_window[1] - trial_time_window[0]) / dt) + 1)
    # Check if the PSTH bins match.
    np.testing.assert_equal(time_bins[:-1], psth_trial_df.columns.values)

    metric_table = pd.DataFrame(index=unit_ids,
                                columns=stimulus_presentation_ids)
    metric_table.index.name = 'units'

    for u, unit_id in enumerate(unit_ids):
      for s, stimulus_presentation_id in enumerate(stimulus_presentation_ids):

        spike_times = spikes_table[
            (spikes_table['unit_id'] == unit_id) &
            (spikes_table['stimulus_presentation_id'] ==
             stimulus_presentation_id)]
        spike_times = spike_times['time_since_stimulus_presentation_onset']
        spike_train = np.histogram(spike_times, time_bins)[0]
        # Similar to correlation, unnormalized.
        metric_value = np.dot(
            spike_train, psth_trial_df.loc[stimulus_presentation_id])
        # Take log-likelihood as the score.
        # log_intensity = np.log(psth_trial_df.loc[stimulus_presentation_id])
        # metric_value = np.dot(spike_train, log_intensity)

        metric_table.loc[unit_id, stimulus_presentation_id] = metric_value

    metric_table = metric_table.apply(pd.to_numeric, errors='coerce')
    return metric_table

  def subgroups_search_by_psth_loop(
      self,
      pool_units_ids,
      sub_group_df_init,
      stimulus_presentation_ids,
      metric_type='psth',
      max_iter=5):
    """Search the subgroups between areas by matching the PSTH."""
    sub_group_df = sub_group_df_init

    for itr in range(max_iter):
      print('Iteration:', itr)
      psth_dict = self.get_across_sub_areas_metrics(
          sub_group_df,
          stimulus_presentation_ids=stimulus_presentation_ids,
          trial_time_window=[0, 2.],
          metric_type=metric_type,  # opt: psth, intensity.
          dt=0.005,
          smooth_sigma=2)

      sub_group_df1 = self.seperate_probe_pairs_units_by_psth_match_scores(
          unit_ids=pool_units_ids,
          psth_dict=psth_dict,
          score_quantile_threshold=0.7)

      # Check convergence.
      sub_units0 = sub_group_df[
          (sub_group_df['group_id'] == 0)]['unit_ids'].values
      sub_units1 = sub_group_df1[
          (sub_group_df1['group_id'] == 0)]['unit_ids'].values

      intersected_units = np.intersect1d(sub_units0, sub_units1)
      match_ratio = len(intersected_units) / len(sub_units0)
      print('Match ratio:', match_ratio)

      sub_group_df = sub_group_df1

      if match_ratio > 0.99:
        break

    return sub_group_df, psth_dict
