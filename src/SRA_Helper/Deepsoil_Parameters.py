# -*- coding: utf-8 -*-
"""
Fit DEEPSOIL MKZ + MRDF-Darendeli parameters for every unique Darendeli
soil type of a discretized pystrata profile and lay them out in the same
column order as DEEPSOIL's "Advanced Table View", ready to paste in.

Targets are the pystrata-generated MRD curves (which already use the
Dmin scaled to kappa = 0.036), so the fitted DEEPSOIL model reproduces
the same curves used in the pystrata analysis (RHS_Stafford.py).

Fitting machinery comes from Darendeli_fit.py:
  - MKZ backbone      G/Gmax = 1 / (1 + beta*(gamma/gamma_r)**s)
  - MRDF-Darendeli    F(g)   = P1 * (G/Gmax)**P2
                      D      = F * D_Masing + Dmin

Damping is fitted over the full strain range of the target curve,
including the constant plateau pystrata clips beyond ~1.3 %. This
matches DEEPSOIL's own curve fitting and keeps the fitted damping from
dipping at large strains (the MRDF factor P1*(G/Gmax)**P2 otherwise
drags it down once the fit is unconstrained past the last kept point).

Used by RHS_Stafford.py (in-memory); can also be run standalone, in
which case it reads the previously exported profile/curve workbooks.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid

FIT_STRAIN_MAX = np.inf       # % ; fit damping over the full target range
REFERENCE_STRESS_MPA = 0.18   # DEEPSOIL default; b = 0 -> not used


def mkz_ggmax(gamma, gamma_r, beta,s):
    """MKZ modulus reduction: G/Gmax = 1 / (1 + beta*(gamma/gamma_r)**s)."""
    return 1.0 / (1.0 + beta * (gamma / gamma_r) ** s)


def non_masing_damping(gamma, GGmax):
    """
    Hysteretic damping (percent) of a backbone tau = G0*GGmax*gamma under
    Masing rules:  D = (4/pi) * [ int(tau dg)/ (tau_m*g_m) - 1/2 ].
    Computed numerically so it works for any backbone shape.
    """
    tau = GGmax * gamma                          # G0 = 1 (cancels)
    W = cumulative_trapezoid(tau, gamma, initial=0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        D = (400.0 / np.pi) * (W / (tau * gamma) - 0.5)
    D[0] = 0.0
    return np.clip(D, 0.0, None)



def fit_soil_type(gamma_t, GG_t, D_t, Dmin):
    """MR fit (MKZ) then MRDF-Darendeli damping fit against target points.

    gamma_t : target strains (%), GG_t : target G/Gmax,
    D_t : target total damping (%), Dmin : small-strain damping (%).
    """
    # --- MR fit over the full strain range ---------------------------
    s = 0.915
    (gr, beta), _ = curve_fit(
        lambda g, gr, beta: mkz_ggmax(g, gr, beta, s),
        gamma_t, GG_t, p0=[0.05, 1.0],
        bounds=([1e-4, 0.1], [10.0, 10.0]),
        maxfev=20000)

    # --- Masing damping of the fitted backbone on a dense grid -------
    # (dense grid keeps the trapezoid integration in non_masing_damping
    # accurate; the ~20 target points alone are too coarse)
    g_dense = np.logspace(-5, np.log10(max(gamma_t) * 1.01), 400)
    GG_dense = mkz_ggmax(g_dense, gr, beta, s)
    Dmas_dense = non_masing_damping(g_dense, GG_dense)

    keep = gamma_t <= FIT_STRAIN_MAX
    g_fit = gamma_t[keep]
    GG_fit = mkz_ggmax(g_fit, gr, beta, s)
    Dmas_fit = np.interp(g_fit, g_dense, Dmas_dense)

    def model_D(gamma_dummy, P1, P2):
        F = P1 * GG_fit ** P2
        return F * Dmas_fit + Dmin

    (P1, P2), _ = curve_fit(model_D, g_fit, D_t[keep],
                            p0=[0.62, 0.1],
                            bounds=([0.05, 0.0], [1.5, 2.0]),
                            maxfev=20000)

    rms_gg = np.sqrt(np.mean((mkz_ggmax(gamma_t, gr, beta, s) - GG_t) ** 2))
    rms_d = np.sqrt(np.mean((model_D(g_fit, P1, P2) - D_t[keep]) ** 2))

    fit_curves = dict(gamma=g_dense, GG=GG_dense,
                      D=P1 * GG_dense ** P2 * Dmas_dense + Dmin)
    return dict(gamma_r=gr, beta=beta, s=s, P1=P1, P2=P2,
                rms_gg=rms_gg, rms_d=rms_d), fit_curves


def fit_profile_parameters(profile_df, curve_dfs, verbose=True,
                           collect_curves=None):
    """Fit each unique soil type of a discretized profile.

    profile_df : dataframe from profile_to_dataframe() in RHS_Stafford.py
    curve_dfs  : {layer number (int) -> dataframe with 'Strain (%)',
                  'G/Gmax', 'Damping (%)'} target curves per layer
    collect_curves : optional dict; filled with
                     {soil type -> (target_df, fitted_curves)} for plotting

    Returns one row per unique soil type with the fitted parameters.
    """
    soil_rows = profile_df[~profile_df['Is Halfspace']]
    firsts = soil_rows.drop_duplicates('Soil Type')

    rows = []
    for _, lay in firsts.iterrows():
        name = lay['Soil Type']
        Dmin = lay['Damping Min (%)']
        if int(lay['Layer']) not in curve_dfs:
            # Linear soil type (no MRD curves) -> nothing to fit
            if verbose:
                print(f"{name:50s} linear (no MRD curves), skipped")
            continue
        tgt = curve_dfs[int(lay['Layer'])]

        pars, fc = fit_soil_type(tgt['Strain (%)'].to_numpy(),
                                 tgt['G/Gmax'].to_numpy(),
                                 tgt['Damping (%)'].to_numpy(),
                                 Dmin)
        if collect_curves is not None:
            collect_curves[name] = (tgt, fc)

        rows.append({'Soil Type': name,
                     'Dmin (%)': Dmin,
                     'Ref. Strain (%)': pars['gamma_r'],
                     'β': pars['beta'],
                     's': pars['s'],
                     'P1': pars['P1'],
                     'P2': pars['P2'],
                     'RMS G/Gmax (-)': pars['rms_gg'],
                     'RMS Damping (%)': pars['rms_d']})
        if verbose:
            print(f"{name:50s} gr={pars['gamma_r']:.4g}%  "
                  f"beta={pars['beta']:.4g}  s={pars['s']:.4g}  "
                  f"P1={pars['P1']:.4g}  P2={pars['P2']:.4g}  "
                  f"rmsD={pars['rms_d']:.3f}%")
    return pd.DataFrame(rows)


def make_deepsoil_table(profile_df, params_df):
    """Combine profile and fitted parameters into DEEPSOIL's
    Advanced Table View layout (halfspace excluded — it is defined
    separately in DEEPSOIL)."""
    soil = profile_df[~profile_df['Is Halfspace']]
    merged = soil.merge(
        params_df[['Soil Type', 'Dmin (%)', 'Ref. Strain (%)',
                   'β', 's', 'P1', 'P2']],
        on='Soil Type', how='left')

    n = len(merged)
    blank = [None] * n
    # Soil types without fitted parameters are linear elastic
    linear = merged['Ref. Strain (%)'].isna().to_numpy()
    merged['Dmin (%)'] = merged['Dmin (%)'].fillna(merged['Damping Min (%)'])
    return pd.DataFrame({
        'Layer Number': merged['Layer'].to_numpy(),
        'Layer Name': [f"Layer {i}" for i in merged['Layer']],
        'Thickness (m)': merged['Thickness (m)'].to_numpy(),
        'Unit Weight (KN/m^3)': merged['Unit Weight (kN/m3)'].to_numpy(),
        'Shear Wave Velocity (m/s)': merged['Vs (m/s)'].to_numpy(),
        'Shear Strength (kPa)': blank,
        'Soil Model': np.where(linear, 'Linear', 'MKZ'),
        'Dmin (%)': merged['Dmin (%)'].to_numpy(),
        'Ref. Strain (%)': merged['Ref. Strain (%)'].to_numpy(),
        'Reference Stress (MPa)': np.where(linear, None,
                                           REFERENCE_STRESS_MPA),
        'β': merged['β'].to_numpy(),
        's': merged['s'].to_numpy(),
        'b': np.where(linear, None, 0),
        'd': np.where(linear, None, 0),
        'Θ1': blank, 'Θ2': blank, 'Θ3': blank, 'Θ4': blank, 'Θ5': blank,
        'A': blank, 'γ1': blank,
        'Reduction Factor Formulation': np.where(linear, None,
                                                 'MRDF-Darendeli'),
        'P1': merged['P1'].to_numpy(),
        'P2': merged['P2'].to_numpy(),
        'P3': blank,
    })


def write_parameter_workbook(path, profile_df, curve_dfs, verbose=True,
                             collect_curves=None):
    """Fit all soil types and write the DEEPSOIL parameter workbook."""
    params_df = fit_profile_parameters(profile_df, curve_dfs,
                                       verbose=verbose,
                                       collect_curves=collect_curves)
    table_df = make_deepsoil_table(profile_df, params_df)
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        table_df.to_excel(writer, sheet_name='DEEPSOIL_Advanced_Table',
                          index=False)
        params_df.to_excel(writer, sheet_name='Soil_Type_Parameters',
                           index=False)
    if verbose:
        print(f"\nSaved DEEPSOIL parameters to {path}")
    return params_df, table_df


def plot_fits(collected, path):
    """Verification plot: target points vs fitted MKZ/MRDF curves."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(collected)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.2 * ncols, 3.2 * nrows))
    for ax, (name, (tgt, fc)) in zip(np.ravel(axes), collected.items()):
        in_rng = tgt['Strain (%)'] <= FIT_STRAIN_MAX
        ax.semilogx(tgt['Strain (%)'][in_rng], tgt['G/Gmax'][in_rng],
                    'ko', ms=4)
        ax.semilogx(fc['gamma'], fc['GG'], 'r-', lw=1.2)
        ax2 = ax.twinx()
        ax2.semilogx(tgt['Strain (%)'][in_rng], tgt['Damping (%)'][in_rng],
                     'bs', ms=4)
        ax2.semilogx(fc['gamma'], fc['D'], 'b-', lw=1.2)
        ax.set_xlim(1e-4, 100)
        ax.set_title(name.replace('Darendeli ', ''), fontsize=8)
        ax.set_ylabel('G/Gmax', fontsize=8)
        ax2.set_ylabel('D (%)', color='b', fontsize=8)
        ax.tick_params(labelsize=7)
        ax2.tick_params(labelsize=7, colors='b')
    for ax in np.ravel(axes)[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Verification plot saved to {path}")


# ----------------------------------------------------------------------
# Standalone use: read the previously exported profile / curve workbooks
# ----------------------------------------------------------------------
# if __name__ == '__main__':
#     profile_df = pd.read_excel('RHS/RHS_Discretized_Profile.xlsx')
#     curves_xl = pd.ExcelFile('RHS/RHS_DEEPSOIL_Discretized_MRD_Curves.xlsx')
#     curve_dfs = {int(str(name).split('_')[-1]): curves_xl.parse(name)
#                  for name in curves_xl.sheet_names}

#     collected = {}
#     write_parameter_workbook('RHS/RHS_DEEPSOIL_Parameters.xlsx',
#                              profile_df, curve_dfs,
#                              collect_curves=collected)
#     plot_fits(collected, 'RHS/RHS_Deepsoil_Parameter_Fits.png')
