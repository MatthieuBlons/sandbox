from sksurv.nonparametric import kaplan_meier_estimator
import matplotlib.pyplot as plt


from sksurv.nonparametric import kaplan_meier_estimator
import matplotlib.pyplot as plt
import numpy as np


def survival_curve(
    df,
    metric,
    event,
    stratif=None,
    variability=False,
    time_pt=None,
    time_unit="days",
    ax=None,
    show=True,
):
    summary = {}
    if not ax:
        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1, fig=fig)

    def plot_km_curve(time, survival_prob, conf_int, label):
        ax.step(time, survival_prob, label=label, linewidth=1.5, where="post")
        if variability:
            ax.fill_between(time, conf_int[0], conf_int[1], alpha=0.2, step="post")
        ax.set_ylim(bottom=0, top=1)

    tmp = df.dropna(axis=0, how="any", subset=[metric, event], inplace=False)
    if stratif is not None:  # Stratified survival curves
        # if stratif is list
        # new_var = results_table.apply(make_new_var(on_vars), axis=1)
        # stratif = "_".join(on_vars)
        # df[stratif] = new_var
        tmp = tmp.dropna(axis=0, subset=stratif, inplace=False)
        for group in tmp[stratif].unique():
            summary[group] = {}
            mask = tmp.loc[tmp[stratif] == group]
            time, survival_prob, conf_int = kaplan_meier_estimator(
                mask[event].astype("bool"),
                mask[metric],
                conf_level=0.95,
                conf_type="log-log",
            )
            summary[group]["unit"] = time_unit
            summary[group]["time"] = time
            summary[group]["proba"] = survival_prob
            summary[group]["confidence"] = conf_int
            # Median PFS
            cross = survival_prob <= 0.5
            summary[group]["median"] = time[cross][0] if cross.any() else np.nan
            # plot
            mask = [survival_prob[i] != 0 for i in range(len(survival_prob))]
            plot_km_curve(
                time[mask], survival_prob[mask], conf_int[:, mask], f"{group}"
            )

        ax.set_title(f"Kaplan Meier Estimates for {metric} with respect to {stratif}")

    else:  # Single survival curve
        summary["unit"] = time_unit
        time, survival_prob, conf_int = kaplan_meier_estimator(
            tmp[event].astype("bool"), tmp[metric], conf_level=0.95, conf_type="log-log"
        )
        summary["time"] = time
        summary["proba"] = survival_prob
        summary["confidence"] = conf_int
        # Median PFS
        cross = survival_prob <= 0.5
        summary["median"] = time[cross][0] if cross.any() else np.nan
        plot_km_curve(time, survival_prob, conf_int, metric)
        ax.set_title(f"{metric}")
    if time_pt:
        ax.vlines(
            x=time_pt,
            colors="red",
            linestyles="dashed",
            ymin=0,
            ymax=1,
            label=f"{time_pt} {time_unit}",
        )
    ax.set_xlabel(f"{time_unit}")
    ax.set_ylabel("Fraction of patients")
    ax.legend()

    if show:
        plt.show()

    return tmp, summary


def km_at_time(time, proba, conf, thr):
    """
    Return the Kaplan-Meier estimate and 95% CI at time t.

    Parameters
    ----------
    time : ndarray
        Event times.
    proba : ndarray
        KM survival probabilities.
    conf : ndarray
        Shape (2, n_times): lower and upper confidence limits.
    thr : float
        Time of interest.

    Returns
    -------
    metric : float
    ci_low : float
    ci_high : float
    """
    idx = np.searchsorted(time, thr, side="right") - 1

    if idx < 0:
        # before the first event
        return 1.0, 1.0, 1.0

    return (
        proba[idx],
        conf[0, idx],
        conf[1, idx],
    )

