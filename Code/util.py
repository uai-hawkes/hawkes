from absl import logging
import h5py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle
import seaborn
import scipy
import scipy.interpolate
from scipy import signal

"""
Total areas
'APN', 'LP', 'MB', 'DG', 'CA1', 'VISrl', nan, 'TH', 'LGd', 'CA3', 'VIS', 'CA2',
'ProS', 'VISp', 'POL', 'VISpm', 'PPT', 'OP', 'NOT', 'HPF', 'SUB', 'VISam', 'ZI',
'LGv', 'VISal', 'VISl', 'SGN', 'SCig', 'MGm', 'MGv', 'VPM', 'grey', 'Eth',
'VPL', 'IGL', 'PP', 'PIL', 'PO', 'VISmma', 'POST', 'SCop', 'SCsg', 'SCzo',
'SCiw', 'IntG', 'MGd', 'MRN', 'LD', 'VISmmp', 'CP', 'VISli', 'PRE', 'RPF', 'LT',
'PF', 'PoT', 'VL', 'RT'

Areas names from data:
units = cache.get_units()
units['ecephys_structure_acronym'].unique()


MB: Midbrain.
APN: Anterior Pretectal area. The pretectal area, or pretectum, is a midbrain
    structure composed of seven nuclei and comprises part of the subcortical
    visual system. 
PPT: Posterior pretectal nucleus
NOT: Nucleus of the optic tract.
AT: Anterior tegmental nucleus.
DT: Dorsal terminal nucleus of the accessory optic tract.
LT: Lateral terminal nucleus of the accessory optic tract.
SC: Superior colliculus
SCig: Superior colliculus, motor related, intermediate gray layer.
SCop: Superior colliculus, optic layer.
SCsg: Superior colliculus, superficial gray layer.
SCzo: Superior colliculus, zonal layer.
SCiw: Superior colliculus, motor related, intermediate white layer.
MRN: Midbrain reticular nucleus.
RPF: Retroparafascicular nucleus.
OP: Olivary pretectal nucleus.


HPF: Hippocampal formation.
CA CA1 CA2 CA3
DG: The dentate gyrus (DG), is part of the hippocampal formation in the temporal
    lobe of the brain that includes the hippocampus, and the subiculum.
ProS: Prosubiculum. Hippocapal formation.
SUB: Subiculum. 
POST: Postsubiculum.
PRE: Presubiculum.


VIS: 
VISam: Anteromedial.
VISpm: Posteromedial.
VISp: Primary.
VISl: Lateral.
VISal: Anterolateral.
VISrl: rostrolateral.
VISli: Laterointermediate area.
VISmmp: Mediomedial posterior visual area.
VISmma: Mediomedial anterior visual area.


TH: Thalamus.
LGd: The dorsolateral geniculate nucleus is the main division of the lateral
    geniculate body. The majority of input to the dLGN comes from the retina. It
    is laminated and shows retinotopic organization
LGv: The ventrolateral geniculate nucleus has been found to be relatively large
    in several species such as lizards, rodents, cows, cats, and primates
IGL: Intergeniculate leaflet of the lateral geniculate complex. A distinctive
    subdivision of the lateral geniculate complex in some rodents that
    participates in the regulation of circadian rhythm.
POL: Posterior limiting nucleus of the thalamus.
PO: Posterior complex of the thalamus.
SGN: Suprageniculate nucleus.
MGm: Medial geniculate complex, medial part.
MGv: Medial geniculate complex, ventral part.
MGd: Medial geniculate complex, dorsal part.
VPM: Ventral posteromedial nucleus of the thalamus.
Eth: Ethmoid nucleus of the thalamus.
VPL: Ventral posterolateral nucleus of the thalamus.
PP: Peripeduncular nucleus.
PIL: Posterior intralaminar thalamic nucleus.
IntG: Intermediate geniculate nucleus.
LD: Lateral dorsal nucleus of thalamus.
RT: Reticular nucleus of the thalamus.
PF: Parafascicular nucleus.
PoT: Posterior triangular thalamic nucleus.
LP: Lateral posterior nucleus of the thalamus.


Grey matter.
grey: Grey matter.
ZI: The zona incerta is a horizontally elongated region of gray matter in the
    subthalamus below the thalamus.

Striutum.
CP: Caudoputamen.

Ventricle.
VL: lateral ventricle.

The brain region names come from
Allen Institute 2017 - Allen Mouse Common Coordinate Framework v3.pdf

Also check: https://allensdk.readthedocs.io/en/v2.1.0/_static/
    examples/nb/ecephys_data_access.html

name: the probe name is assigned based on the location of the probe on the 
recording rig. This is useful to keep in mind because probes with the same name 
are always targeted to the same cortical region and enter the brain from the 
same angle (probeA = AM, probeB = PM, probeC = V1, probeD = LM, probeE = AL, 
probeF = RL). However, the targeting is not always accurate, so the actual 
recorded region may be different.
"""
# Orange.
MIDBRAIN = ['APN', 'MB', 'AT', 'DT', 'PPT', 'NOT', 'LT', 'OP',
            'SC', 'SCig', 'SCiw', 'SCzo', 'SCsg', 'SCop', 'MRN', 'RPF']

# Blue.
HIPPOCAMPUS_AREA = ['HPF', 'CA', 'DG', 'CA1', 'CA2', 'CA3', 'ProS', 'SUB',
                    'POST', 'PRE']

# Red
THALAMUS_AREA = ['TH', 'LGd', 'LGv', 'LP', 'IGL', 'PO', 'POL', 'SGN',
                 'MGv', 'MGm', 'MGd', 'VPM', 'Eth', 'VPL', 'PP', 'PIL', 'IntG',
                 'LD', 'RT', 'PF', 'PoT']

# Green.
VISUAL_AREA = ['VIS', 'VISam', 'VISpm', 'VISp', 'VISl', 'VISal', 'VISrl',
               'VISmmp', 'VISmma', 'VISli']


def load_nwb_file(file_path):
  """Loads nwb files.

  H5 data variable can be saved, but not loaded.
  
  Args:
    file_path:

  Returns:
    The h5 type variable.
  """
  return h5py.File(file_path)


def save_variable(file_path, data, verbose=True):
  """Saves variable to pickle file.

  H5 data variable can be saved, but not loaded. To overcome the OverflowError:
  cannot serialize a bytes object larger than 4 GiB. `protocol=4` is added.

  Args:
    file_path: The output file path.
    data: Python variables. If you want to save multiple variables, put them in
        a `tuple` or `collections.namedtuple`.
  """
  with open(file_path, 'wb') as f:
    pickle.dump(data, f, protocol=4)
  if verbose:
    print('util.save_variable, save variable to: ', file_path)
    logging.info('util.save_variable, save variable to: %s', file_path)


def save_as_mat(
    file_path,
    data,
    variable_name='data',
    verbose=False):
  """Save data in mat format for Matlab input."""
  adict = {'data':data}
  scipy.io.savemat(file_path, adict)
  if verbose:
    print('Save mat file:', file_path)


def load_as_mat(
    file_path,
    verbose=False):
  """Load mat file."""
  data = scipy.io.loadmat(file_path)
  return data


def load_variable(file_path, verbose=False):
  """Loads variable from a pickle file.

  H5 data can be saved, but not loaded.

  Args:
    file_path:
  """
  with open(file_path, 'rb') as f:
    output_variable = pickle.load(f)
    if verbose:
      print('Load variabe from:', file_path)
  return output_variable


def bin_spike_times(
    spike_times,
    bin_width,
    len_trial):
  """Convert spike times to spike bins, spike times list to spike bins matrix.

  # spike times outside the time range will not be counted in the bin at the
  # end. A time bin is left-closed right-open [t, t+delta).
  # e.g. t = [0,1,2,3,4], y = [0, 0.1, 0.2, 1.1, 5, 6]
  # output: [3, 1, 0, 0]

  Args:
    spike_times: The format can be list, np.ndarray.
  """
  bins = np.arange(0, len_trial+bin_width, bin_width)

  if len(spike_times) == 0:
    return np.zeros(len(bins)-1), bins[:-1]

  # multiple spike_times.
  elif isinstance(spike_times[0], list) or isinstance(spike_times[0], np.ndarray):
    num_trials = len(spike_times)
    num_bins = len(bins) - 1
    spike_hist = np.zeros((num_trials, num_bins))
    for r in range(num_trials):
      spike_hist[r], _ = np.histogram(spike_times[r], bins)

  # single spike_times.
  else:
    spike_hist, _ = np.histogram(spike_times, bins)
  return spike_hist, bins[:-1]


def color_by_brain_area(ccf_structure, colortype='normal'):
  """Assign a color for a brain area."""
  if ccf_structure in VISUAL_AREA:
    color = 'tab:green'
    if colortype == 'dark':
      color = 'darkgreen'
    elif colortype =='light':
      color = 'lime'
    elif colortype =='rgby':
      color = [0.30196078, 0.68627451, 0.29019608, 1.]
  elif ccf_structure in HIPPOCAMPUS_AREA:
    color = 'tab:blue'
    if colortype == 'dark':
      color = 'darkblue'
    elif colortype =='light':
      color = 'lightblue'
    elif colortype =='rgby':
      color = [0.21568627, 0.49411765, 0.72156863, 1.]
  elif ccf_structure in THALAMUS_AREA:
    color = 'tab:red'
    if colortype == 'dark':
      color = 'darkred'
    elif colortype =='light':
      color = 'lightcoral'
    elif colortype =='rgby':
      color = [0.89411765, 0.10196078, 0.10980392, 1.]
  elif ccf_structure in MIDBRAIN:
    color = 'tab:orange'
    if colortype == 'dark':
      color = 'darkorange'
    elif colortype =='light':
      color = 'gold'
    elif colortype =='rgby':
      color = [1., 0.49803922, 0.,1.]
  else:
    color = 'tab:gray'
    if colortype == 'dark':
      color = 'dimgray'
    elif colortype =='light':
      color = 'lightgray'
    elif colortype =='rgby':
      color = [.5, .5, .5, 1.0]
  return color


def cross_pearson_corr(
    y1,
    y2,
    index_range):
  assert len(y1) == len(y2)
  num_bins = len(y1)
  lag_index = np.arange(index_range[0], index_range[1]).astype(int)
  num_lags = len(lag_index)
  corr = np.zeros(num_lags)

  for i, lag in enumerate(lag_index):
    if lag == 0:
      r, pval = scipy.stats.pearsonr(y1, y2)
    elif lag > 0:
      r, pval = scipy.stats.pearsonr(y1[lag:], y2[:-lag])
    elif lag < 0:
      lag = abs(lag)
      r, pval = scipy.stats.pearsonr(y1[:-lag], y2[lag:])

    corr[i] = r

  return corr, lag_index


def cross_corr(
    y1,
    y2,
    index_range=None,
    type='max',
    debias=True):
  """Calculates the cross correlation and lags with normalization.

  The definition of the discrete cross correlation is in:
  https://www.mathworks.com/help/matlab/ref/xcorr.html
  The `y1` takes the first place, and the `y2` takes the second place. So when
  lag is negtive, it means the `log_lmbd` is on the left of `spike_trains`.

  Args:
    index_range: two entries list. [min_index, max_index]. If the index_range is
        beyond the range of the array, it will
        automatically be clipped to the bounds.
    type:
        'max': single max value.
        'full': get the whole correlation and corresponding lags.

  Returns:
    max_corr: Maximum correlation without normalization.
    lag: The lag in terms of the index.
  """

  # plt.figure()
  # plt.plot(y1, '.')
  # plt.plot(y2, '.')
  # plt.show()

  if len(y1) != len(y2):
    raise ValueError('The lengths of the inputs should be the same.')

  y1_auto_corr = np.dot(y1, y1) / len(y1)
  y2_auto_corr = np.dot(y2, y2) / len(y1)
  corr = signal.correlate(y1, y2, mode='same')
  # The unbiased sample size is N - lag.
  if debias:
    sample_size = signal.correlate(
        np.ones(len(y1)), np.ones(len(y1)), mode='same')
  else:
    sample_size = len(y1)

  if y1_auto_corr != 0 and y2_auto_corr != 0:
    corr = corr / sample_size / np.sqrt(y1_auto_corr * y2_auto_corr)
  shift = len(y1) // 2

  if index_range is None and type == 'max':
    max_corr = np.max(corr)
    argmax_corr = np.argmax(corr)
    return max_corr, argmax_corr - shift
  elif index_range is None and type == 'full':
    return corr, np.arange(len(corr)) - shift

  index_range = np.array(index_range).astype(int)
  shifted_index_range = index_range + shift
  if index_range[0] + shift < 0:
    index_range[0] -= index_range[0] + shift
    shifted_index_range[0] = 0
  if index_range[1] + shift >= len(y1):
    index_range[1] -= index_range[1] + shift - len(y1) + 1
    shifted_index_range[1] = len(y1) - 1

  index_range_mask = np.array(
      range(index_range[0], index_range[1] + 1))
  shifted_index_range_mask = np.array(
      range(shifted_index_range[0], shifted_index_range[1] + 1))

  if type == 'max':
    max_corr = np.max(corr[shifted_index_range_mask])
    argmax_corr = np.argmax(corr[shifted_index_range_mask])
    lag = index_range_mask[argmax_corr]
    return max_corr, lag
  elif type == 'full':
    return corr[shifted_index_range_mask], index_range_mask


def cross_prod(
    y1,
    y2,
    index_range=None):
  """Calculates the cross correlation and lags without normalization.

  Args:
    index_range: two entries list. [min_index, max_index]. If the index_range is
        beyond the range of the array, it will
        automatically be clipped to the bounds.
  """
  if y1.shape != y2.shape:
    raise ValueError('The lengths of the inputs should be the same.')
  if len(y1.shape) == 1:
    num_bins = len(y1)
  elif len(y1.shape) == 2:
    num_bins = y1.shape[1]

  corr = scipy.signal.correlate(y1, y2, mode='same')
  # The unbiased sample size is N - lag.
  unbiased_sample_size = scipy.signal.correlate(
      np.ones(num_bins), np.ones(num_bins), mode='same')
  corr = corr / unbiased_sample_size
  shift = num_bins // 2

  if index_range is None:
    return corr, np.arange(num_bins) - shift

  index_range = np.array(index_range).astype(int)
  shifted_index_range = index_range + shift
  index_range_mask = np.array(
      range(index_range[0], index_range[1] + 1))
  shifted_index_range_mask = np.array(
      range(shifted_index_range[0], shifted_index_range[1] + 1))
  return corr[shifted_index_range_mask], index_range_mask


def array_shift(x, shift, zero_pad=False):
  """Shift the array.

  Args:
    shift: Negtive to shift left, positive to shift right.
  """
  x = np.array(x)
  shift = np.array(shift)
  if len(x.shape) > 2:
    raise ValueError('x can only be an array of a matrix.')

  # If `shift` is a scalar.
  if len(shift.shape) == 0:
    # If x is 1D array, shift along axis=0, if x is 2D matrix, shift along rows.
    x = np.roll(x, shift, axis=len(x.shape)-1)
    # pad zeros to the new positions.
    if zero_pad and len(x.shape) == 1 and shift > 0:
      x[:shift] = 0
    elif zero_pad and len(x.shape) == 1 and shift < 0:
      x[shift:] = 0
    elif zero_pad and len(x.shape) > 1 and shift > 0:
      x[:, :shift] = 0
    elif zero_pad and len(x.shape) > 1 and shift > 0:
      x[:, shift:] = 0
  # Shift matrix rows independently.
  elif len(shift.shape) == 1: 
    if len(shift) != x.shape[0]:
      raise ValueError('length of shift should be equal to rows of x.')
    for row, s in enumerate(shift):
      x[row] = np.roll(x[row], s)
      # pad zeros to the new positions.
      if zero_pad and s > 0:
        x[row, :s] = 0
      elif zero_pad and s < 0:
        x[row, s:] = 0
  else:
    raise ValueError('shift can be a scalar or a vector for each row in x.')
  return x


def construct_b_spline_basis(
    spline_order,
    knots,
    dx,
    add_constant_basis=True,
    show_plot=False):
  """Constructs B-spline basis."""
  num_basis = len(knots) - spline_order - 1
  num_rows = int(np.round((knots[-1] - knots[0]) / dx)) + 1
  x = np.linspace(knots[0], knots[-1], num_rows)
  basis_matrix = np.zeros((len(x), num_basis))
  interpolate_token=[0, 0, spline_order]
  interpolate_token[0] = np.array(knots)

  for i in range(num_basis):
    basis_coefficients = [0] * num_basis
    basis_coefficients[i] = 1.0 
    interpolate_token[1] = basis_coefficients
    y = scipy.interpolate.splev(x, interpolate_token)
    basis_matrix[:, i] = y

  if add_constant_basis:
    basis_matrix = np.hstack((np.ones((len(x), 1)), basis_matrix))

  if show_plot:
    plt.figure()
    plt.plot(x, basis_matrix, '.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

  return x, basis_matrix


def construct_b_spline_basis_even_knots(
    spline_order,
    num_knots,
    x_range,
    dx,
    add_constant_basis=True,
    show_plot=False):
  """Constructs B-spline basis with knots equal distance.

  Args:
    x_range: [left_end, right_end].
  """
  # construct_b_spline_basis
  knots = np.linspace(x_range[0], x_range[1], num_knots)
  knots = np.hstack((np.ones(spline_order) * x_range[0],
                     knots,
                     np.ones(spline_order) * x_range[1]))

  return construct_b_spline_basis(
      spline_order, knots, dx, add_constant_basis, show_plot)


def butter_bandpass(lowcut, highcut, fs, order=5):
  """Design of the Butterworth bandpass filter."""
  nyq = 0.5 * fs
  low = lowcut / nyq
  high = highcut / nyq
  b, a = signal.butter(order, [low, high], btype='band')
  return b, a


def butterworth_bandpass_filter(x, lowcut, highcut, fs, order=4):
  """Butter bandpass filter.

  Args:
    x: Input signal.
    fs: Sampling frequency.
    order: The order of the Butterworth filter.
  """
  b, a = butter_bandpass(lowcut, highcut, fs, order=order)
  y = signal.lfilter(b, a, x)
  return y


def get_power_spectrum(
    x,
    fs,
    output_figure_path=None,
    show_figure=False):
  """Gets the power spectrum."""
  num_per_segment = 2 ** 12
  f, Pxx_den = signal.welch(x, fs, nperseg=1024)

  plt.figure()
  # plt.semilogy(f, Pxx_den)
  plt.plot(f, Pxx_den)
  plt.xlabel('frequency [Hz]')
  plt.ylabel('PSD [V**2/Hz]')

  if output_figure_path:
    plt.savefig(output_figure_path)
    print('Save figure to: ', output_figure_path)
  if show_figure:
    plt.show()
  plt.close()

  return f, Pxx_den


def get_spectrogram(
    x,
    fs,
    time_offset=0,
    output_figure_path=None,
    show_figure=True):
  """Get the spectrum along time.

  Args:
    x: Input signal.
    fs: Sampling frequency.
  """
  # `nfft` > `nperseg` means apply zero padding to make the spectrum look
  # smoother, but it does not provide extra informaiton. `noverlap` is the
  # overlap between adjacent sliding windows, the larger, the more overlap.
  # num_per_segment = 2 ** 8
  num_per_segment = 250
  f, t, Sxx = signal.spectrogram(
      x, fs,
      nperseg=num_per_segment,
      noverlap=num_per_segment // 50 * 49,
      nfft=num_per_segment * 8)
  t = np.array(t) + time_offset
  # Used to debug the positions of the sliding window.
  # print(np.array(t))

  plt.figure(figsize=(10, 8))
  # plt.pcolormesh(t, f, np.log(Sxx))  # The log scale plot.
  # plt.pcolormesh(t, f, Sxx, vmax=np.max(Sxx) / 10)
  plt.pcolormesh(t, f, Sxx, vmax=200)
  plt.ylim(0, 100)
  plt.colorbar()
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')

  if output_figure_path:
    plt.savefig(output_figure_path)
    print('Save figure to: ', output_figure_path)
  if show_figure:
    plt.show()
  plt.close()


def fisher_transform(rho):
  """Fisher transformation for correlation.

  z = 0.5 * log((1 + rho) / (1 - rho))
  """
  rho = np.array(rho)
  z = 0.5 * np.log((1 + rho) / (1 - rho))
  return z


def marginal_corr_from_cov(cov):
  """Calculates marginal correlation matrix from covariance matrix.

  Args:
    cov: N x N matrix.
  """
  cov_diag_sqrt = np.sqrt(np.diag(cov))
  corr = cov / np.outer(cov_diag_sqrt, cov_diag_sqrt)

  return corr


def partial_corr_from_cov(cov):
  """Calculates partial correlation matrix from covariance matrix.

  Args:
    cov: N x N matrix.
  """
  theta = np.linalg.inv(cov)
  theta_diag_sqrt = np.sqrt(np.diag(theta))
  corr = - theta / np.outer(theta_diag_sqrt, theta_diag_sqrt)

  return corr


def xcorr(x, y,verbose=False):
  """Cross correlation coefficient.

  The lag centers at 0 if two arrays have equal length.

  References:
  https://www.mathworks.com/help/signal/ug/
    confidence-intervals-for-sample-autocorrelation.html
  """
  length = len(x)
  x = x - np.mean(x)
  y = y - np.mean(y)
  sigma = np.sqrt(np.dot(x, x) * np.dot(y, y))
  xcorr = np.correlate(x, y, mode='same') / sigma
  lag = np.arange(length) - length // 2

  # 95% CI, 0.025 on each side.
  alpha = scipy.stats.norm.ppf(0.975)
  CI_level = alpha / np.sqrt(length)

  if verbose:
    plt.figure()
    plt.plot(lag, xcorr)
    plt.axhline(y=CI_level, ls=':')
    plt.axhline(y=-CI_level, ls=':')

  return lag, xcorr, CI_level


def plot_networkx_graph(G):
  """Plot networkx graph."""
  if nx.is_directed(G):
    print('Directed')
    directed = True
  else:
    print('Un-directed')
    directed = False
    # cliques = nx.find_cliques(G)
    # print(list(cliques))

  print(f'num_nodes {G.number_of_nodes()}  num_edges {G.number_of_edges()}')
  plt.figure(figsize=[11, 4])
  plt.subplot(121)
  # pos = nx.spring_layout(graph, scale=2)
  # pos = nx.drawing.random_layout(graph)
  pos=nx.circular_layout(G)
  # nx.draw(G, pos=pos)

  if len(nx.get_node_attributes(G, 'weight')) > 0:
    edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())
    nx.draw(G, pos, node_color='b', edgelist=edges, edge_color=weights,
            width=2, edge_cmap=plt.cm.jet)
  else:
    nx.draw(G, pos, node_color='b', width=2, edge_cmap=plt.cm.jet)
  plt.subplot(122)
  adj_mat = nx.to_numpy_matrix(G)
  seaborn.heatmap(adj_mat)
  plt.show()


def plot_networkx_adj(G):
  """Plot networkx graph."""
  if nx.is_directed(G):
    print('Directed')
    directed = True
  else:
    print('Un-directed')
    directed = False
    cliques = nx.find_cliques(G)

  print(f'num_nodes {G.number_of_nodes()}  num_edges {G.number_of_edges()}')
  plt.figure(figsize=[5, 4])
  adj_mat = nx.to_numpy_matrix(G)
  seaborn.heatmap(adj_mat)
  plt.show()
