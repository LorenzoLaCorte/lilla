

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cycler

palette = sns.color_palette("colorblind")
plt.rcParams['axes.prop_cycle'] = cycler(color=palette)

def plot_algorithm(pmf, fid_func, axs=None, t_trunc=None, legend=None, legend_fid=None):
    """Plot waiting-time PMF/CDF and fidelity curves.

    Parameters
    ----------
    pmf : array-like
        Probability mass function over waiting times.
    fid_func : array-like
        Fidelity values per waiting time. Can be 1D (shape: [T]) or 2D (shape: [N, T] or [T, N]).
    axs : 2D array of matplotlib Axes, optional
        If provided, plots will be drawn on these axes. Expected shape (2, 2).
        If None, a new figure with a (2, 2) grid of subplots is created.
    t_trunc : int, optional
        Truncation time. If None, choose smallest t such that CDF >= 0.997 (fallback full length).
    legend : list[str] | tuple[str] | None
        Uniform legend entries applied to PMF/CDF (kept for backward compatibility).
    legend_fid : list[str] | tuple[str] | None
        Legend entries specifically for the fidelity plot. If fid_func is 2D, this should have
        length equal to the number of fidelity series (N). If 1D, provide a single entry.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure containing the plots.
    axs : numpy.ndarray
        The 2x2 array of axes used for plotting (bottom-right may be unused currently).
    """
    # Convert to numpy arrays (copy to avoid mutating caller data; ensure float for NaN)
    pmf = np.asarray(pmf, dtype=float)
    fid_arr = np.asarray(fid_func, dtype=float)

    # Determine truncation point from PMF CDF
    cdf = np.cumsum(pmf)
    if t_trunc is None:
        try:
            t_trunc = int(np.min(np.where(cdf >= 0.997)))
        except ValueError:
            t_trunc = len(pmf)

    # Truncate PMF
    pmf = pmf[:t_trunc]

    # Create axes if not supplied
    if axs is None:
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    else:
        # Derive figure from provided axes
        fig = axs[0][0].figure

    # Plot PMF
    axs[0][0].plot(np.arange(t_trunc), pmf)
    axs[0][0].set_xlabel("Waiting time $T$")
    axs[0][0].set_ylabel("Probability")
    axs[0][0].set_title("PMF")

    # Plot CDF
    axs[0][1].plot(np.arange(t_trunc), np.cumsum(pmf))
    axs[0][1].set_title("CDF")
    axs[0][1].set_xlabel("Waiting time $T$")
    axs[0][1].set_ylabel("Probability")

    # Prepare and plot Fidelity
    t_axis = np.arange(t_trunc)
    ax_fid = axs[1][0]

    if fid_arr.ndim == 1:
        y = fid_arr[:t_trunc]
        if y.size > 0:
            y[0] = np.nan  # avoid misleading point at T=0
        if legend_fid is not None and len(legend_fid) >= 1:
            ax_fid.plot(t_axis, y, label=legend_fid[0])
            ax_fid.legend()
        else:
            ax_fid.plot(t_axis, y)
    elif fid_arr.ndim == 2:
        series_first = fid_arr.shape[1] >= fid_arr.shape[0]
        arr = fid_arr if series_first else fid_arr.T
        arr = arr[:, :t_trunc]
        if arr.shape[1] > 0:
            arr[:, 0] = np.nan

        n_series = arr.shape[0]

        # Detect if series are very close/overlapping
        def series_are_close(a):
            # Pairwise max and mean absolute differences (ignoring NaN)
            if a.shape[0] < 2:
                return False
            diffs_max = []
            diffs_mean = []
            for i in range(a.shape[0]):
                for j in range(i + 1, a.shape[0]):
                    d = np.nan_to_num(a[i] - a[j], nan=0.0)
                    diffs_max.append(np.max(np.abs(d)))
                    diffs_mean.append(np.mean(np.abs(d)))
            # Thresholds: very small differences considered overlapping
            return (np.min(diffs_max) < 1e-6) or (np.min(diffs_mean) < 1e-3)

        close = series_are_close(arr)

        # If close, differentiate with markers and staggered markevery to avoid overlap
        markers = ['o', 's', 'D', '^']
        markevery_step = 100 if t_trunc >= 500 else max(10, t_trunc // 3 if t_trunc > 0 else 10)
        offsets = [0, markevery_step // 2, markevery_step // 4, 3 * markevery_step // 4]

        for i in range(n_series):
            plot_kwargs = {}
            if close:
                plot_kwargs["marker"] = markers[i % len(markers)]
                # Stagger start so points don't overlap
                start_offset = offsets[i % len(offsets)]
                plot_kwargs["markevery"] = (start_offset, markevery_step)
                plot_kwargs["linewidth"] = 1.5

            if legend_fid is not None and i < len(legend_fid):
                plot_kwargs["label"] = legend_fid[i]

            ax_fid.plot(t_axis, arr[i], **plot_kwargs)

        if legend_fid is not None:
            ax_fid.legend()
    else:
        raise ValueError("fid_func must be a 1D or 2D array-like")

    ax_fid.set_xlabel("Waiting time $T$")
    ax_fid.set_ylabel("Fidelity")
    ax_fid.set_title("Fidelity")

    # Unused bottom-right axis for now
    axs[1][1].axis("off")

    # Optional uniform legends for PMF/CDF
    if legend is not None:
        axs[0][0].legend(legend)
        axs[0][1].legend(legend)

    plt.tight_layout()
    return fig, axs