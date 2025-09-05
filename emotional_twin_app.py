"""
emotional_twin_app.py
----------------------

This Streamlit application implements a simplified proof‑of‑concept of the
emotional digital twin described in the provided architecture and
technical documentation.  The goal of this app is not to replicate the
full production system but to demonstrate how the seven conceptual
modules—risk classification, Ornstein–Uhlenbeck dynamics, Monte Carlo
forecasting, Bayesian inference, SHAP‑style explanations, adherence
modelling and the Γ integration operator—can work together in a single
interactive application.  The implementation avoids external
dependencies (such as scikit‑learn or shap) that are unavailable in
the execution environment and instead relies on numpy and pandas for
numerical work and Streamlit for the user interface.

Users can enter self‑reported scores (0–100) for seven risk domains—
feeding/eating concerns, suicidal behaviour, panic attacks, sleep
difficulties, emotional dysregulation, non‑suicidal self‑injury and
need to reduce substance use—along with their current adherence to
recommended interventions and an overall mood rating.  The app then
computes a risk profile (Module 1), simulates stochastic trajectories
using an Ornstein–Uhlenbeck process (Module 2), performs a Monte
Carlo forecast to estimate the probability of exceeding a wellness
threshold (Module 3), updates a simple Beta–Binomial model to
illustrate Bayesian parameter updates (Module 4), approximates SHAP
explanations via linear contributions (Module 5), tracks adherence
over time (Module 6) and finally combines all of these quantities
through a dynamic weighting scheme to produce a single Wellness Score
S(t) (Module 7).

This file is intended to be run with the ``streamlit run`` command.
Because Streamlit is not installed in the current environment, the
application cannot be executed here, but the code has been written to
be self‑contained and ready for deployment in a compatible Python
environment with Streamlit available.
"""

import base64
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    import streamlit as st  # type: ignore
except ImportError:
    # Streamlit is not available in this environment; define a stub for
    # linters and type checkers.  When the app is run in a proper
    # environment with Streamlit installed, this import will succeed.
    class _StreamlitStub:  # pragma: no cover - stub only used when streamlit is missing
        def __getattr__(self, name):
            raise ImportError("Streamlit is required to run this application")

    st = _StreamlitStub()  # type: ignore


# Constants and configuration -------------------------------------------------

# Names of the clinical domains/risk factors used throughout the app.
CATEGORY_NAMES = [
    "Preocupaciones alimentarias",  # eating concerns
    "Comportamiento suicida",       # suicidal behaviour
    "Ataques de pánico",            # panic attacks
    "Dificultades de sueño",        # sleep difficulties
    "Desregulación emocional",      # emotional dysregulation
    "Autolesiones no suicidas",     # non‑suicidal self‑injury
    "Reducción del uso de sustancias"  # need to reduce substance use
]

# Baseline values for each risk domain used in the SHAP approximation and
# Ornstein–Uhlenbeck processes.  These represent typical “healthy” levels
# around which the OU process mean‑reverts.
BASELINES = np.array([50.0] * len(CATEGORY_NAMES))

# Time horizon for Monte Carlo simulations (in days) and number of
# trajectories.  A smaller number of simulations is used here for
# performance in demonstration; in practice this can be much larger.
SIMULATION_DAYS = 30
N_SIMULATIONS = 200

# Threshold above which the Wellness Score is considered critical.  This
# value is used when computing the exceedance probability in the Monte
# Carlo module and updating the Bayesian Beta–Binomial model.  In a
# clinical setting this would be set by domain experts.
WELLNESS_THRESHOLD = 80.0


def risk_classifier(
    inputs: np.ndarray
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Compute a risk profile from self‑reported inputs.

    Module 1 in the architecture learns a multidimensional risk profile
    across several clinical domains.  Because we cannot train a
    machine‑learning model in this context, this function implements a
    simple, interpretable mapping: it normalises the user inputs into
    weights on the probability simplex and then uses those weights
    both as an interpretable risk profile and to compute an aggregated
    risk score.

    Parameters
    ----------
    inputs : np.ndarray
        A one‑dimensional array of length equal to the number of risk
        categories.  Each entry should be a score between 0 and 100.

    Returns
    -------
    weights : np.ndarray
        Normalised weights representing the relative contribution of
        each risk factor to the overall risk.  These lie on the
        probability simplex (non‑negative and summing to one).
    aggregated_risk : float
        A single risk number computed as a weighted sum of the inputs.
        It ranges between 0 and 100.
    raw_risk : np.ndarray
        The raw risk values (just the inputs in this simplified model).
    """
    # Protect against division by zero if the user enters all zeros.
    total = float(inputs.sum())
    if total <= 0.0:
        # Uniform weights if no risk is reported.
        weights = np.ones_like(inputs) / len(inputs)
    else:
        weights = inputs / total
    # Aggregated risk is the weighted sum of the raw scores.  Because
    # weights sum to one and inputs are in [0, 100], the result will
    # also lie in [0, 100].
    aggregated_risk = float(np.dot(weights, inputs))
    return weights, aggregated_risk, inputs.copy()


def ou_simulation(
    x0: float,
    theta: float = 0.5,
    mu: float = 50.0,
    sigma: float = 10.0,
    dt: float = 1.0,
    horizon: int = SIMULATION_DAYS,
    n_sim: int = N_SIMULATIONS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate Ornstein–Uhlenbeck trajectories and compute summary statistics.

    The Ornstein–Uhlenbeck (OU) process is a mean‑reverting stochastic
    differential equation of the form

        dX_t = θ(μ − X_t) dt + σ dW_t

    where θ>0 controls the rate of reversion to the mean μ, σ>0
    determines the volatility and W_t is standard Brownian motion.  The
    exact solution for discrete time increments can be derived, but
    here we implement the Euler–Maruyama scheme which suffices for
    demonstration purposes.

    Parameters
    ----------
    x0 : float
        The initial state of the process at time t=0.
    theta : float, optional
        Rate of mean reversion.
    mu : float, optional
        Long‑term mean to which the process reverts.
    sigma : float, optional
        Volatility parameter controlling random fluctuations.
    dt : float, optional
        Time step (in days) for simulation.
    horizon : int, optional
        Number of time steps to simulate.
    n_sim : int, optional
        Number of trajectories to generate.

    Returns
    -------
    trajectories : np.ndarray
        Array of shape (n_sim, horizon+1) containing simulated
        trajectories.  The first column contains the initial state x0.
    mean_traj : np.ndarray
        Array of shape (horizon+1,) containing the pointwise means of
        the trajectories.
    std_traj : np.ndarray
        Array of shape (horizon+1,) containing the pointwise standard
        deviations of the trajectories.
    """
    # Preallocate trajectory array
    trajectories = np.zeros((n_sim, horizon + 1), dtype=float)
    trajectories[:, 0] = x0
    # Precompute constants for efficiency
    sqrt_dt_sigma = sigma * np.sqrt(dt)
    for t in range(1, horizon + 1):
        # Generate random normal increments for all simulations at once
        dW = np.random.normal(loc=0.0, scale=1.0, size=n_sim)
        # Euler–Maruyama update
        prev = trajectories[:, t - 1]
        trajectories[:, t] = prev + theta * (mu - prev) * dt + sqrt_dt_sigma * dW
    mean_traj = trajectories.mean(axis=0)
    std_traj = trajectories.std(axis=0)
    return trajectories, mean_traj, std_traj


def monte_carlo_forecast(
    traj: np.ndarray,
    threshold: float = WELLNESS_THRESHOLD
) -> Tuple[float, float]:
    """Estimate exceedance probability and expected final value from trajectories.

    Given a set of simulated trajectories (for example, of the
    aggregated risk or Wellness Score), compute the probability that
    the process exceeds a specified threshold at any point in the
    future horizon and the expected value at the final time step.

    Parameters
    ----------
    traj : np.ndarray
        Array of shape (n_sim, horizon+1) containing simulated
        trajectories.  Each row is a sample path.
    threshold : float, optional
        The critical value for determining risk events.

    Returns
    -------
    exceed_prob : float
        The estimated probability (0–1) that a trajectory exceeds the
        threshold at least once during the horizon.
    expected_final : float
        The expected final value of the process at the end of the
        horizon.
    """
    # Determine for each trajectory whether it exceeds the threshold at
    # any time.  Using axis=1 gives a boolean for each trajectory.
    exceed_flags = (traj >= threshold).any(axis=1)
    exceed_prob = float(exceed_flags.mean())
    expected_final = float(traj[:, -1].mean())
    return exceed_prob, expected_final


def bayesian_beta_binomial_update(
    prior_alpha: float,
    prior_beta: float,
    successes: int,
    trials: int
) -> Tuple[float, float, float]:
    """Perform a Bayesian update of a Beta–Binomial model.

    Module 4 of the architecture continuously updates its beliefs about
    latent parameters as new observations arrive.  A Beta prior on a
    Bernoulli proportion updated with Binomial observations yields a
    Beta posterior with parameters α' = α + k and β' = β + n − k,
    where k is the number of successes and n the number of trials.

    In this simplified example, “success” corresponds to the event
    that the Wellness Score exceeds the threshold within the prediction
    horizon.  Over repeated interactions with the app, alpha and beta
    accumulate evidence about how frequently high‑risk events occur.

    Parameters
    ----------
    prior_alpha : float
        Prior α parameter of the Beta distribution.
    prior_beta : float
        Prior β parameter of the Beta distribution.
    successes : int
        Number of observed exceedances.
    trials : int
        Number of total observations (here equal to the number of
        simulations).

    Returns
    -------
    post_alpha : float
        Updated α parameter.
    post_beta : float
        Updated β parameter.
    posterior_mean : float
        The posterior mean α'/(α' + β').
    """
    post_alpha = prior_alpha + successes
    post_beta = prior_beta + (trials - successes)
    posterior_mean = post_alpha / (post_alpha + post_beta)
    return post_alpha, post_beta, posterior_mean


def shap_like_contributions(
    inputs: np.ndarray,
    baselines: np.ndarray,
    weights: np.ndarray
) -> np.ndarray:
    """Compute simple SHAP‑like feature contributions for a linear model.

    True SHAP values (Shapley Additive Explanations) require a model
    capable of evaluating outputs on all subsets of features, which is
    not feasible here without external libraries.  For an additive
    model with weights w and baseline b, however, the contribution of
    feature i to the difference between the current prediction and the
    baseline prediction can be approximated by

        ϕ_i = w_i * (x_i − b_i)

    Such contributions sum to the difference between the prediction
    and the baseline and provide intuitive, if rough, feature
    attributions.

    Parameters
    ----------
    inputs : np.ndarray
        Current feature values.
    baselines : np.ndarray
        Baseline values for each feature.
    weights : np.ndarray
        Weights used in the risk classifier.

    Returns
    -------
    contributions : np.ndarray
        Contribution of each feature to the difference between the
        current aggregated risk and the baseline aggregated risk.
    """
    diff = inputs - baselines
    contributions = weights * diff
    return contributions


def adherence_update(prev_adherence: float, current: float, alpha: float = 0.5) -> float:
    """Update adherence metric using a simple exponentially weighted average.

    Module 6 monitors engagement with interventions.  Here we update
    adherence based on the most recent self‑reported adherence and the
    previous adherence estimate.  The parameter α determines how
    quickly the new measurement overrides the historical average.

    Parameters
    ----------
    prev_adherence : float
        The adherence estimate from the previous iteration (0–100).
    current : float
        The newly reported adherence (0–100).
    alpha : float, optional
        Smoothing factor between 0 and 1.  Higher values put more
        weight on the current observation.

    Returns
    -------
    updated : float
        The updated adherence estimate.
    """
    return alpha * current + (1.0 - alpha) * prev_adherence


def dynamic_gamma_integration(
    z: List[float],
    variances: List[float],
    epsilon: float = 1e-6
) -> Tuple[np.ndarray, float]:
    """Dynamically weight module outputs to produce a unified Wellness Score.

    Module 7 combines the outputs of Modules 1–6 into a single scalar
    measure S(t).  In lieu of a learned weighting mechanism, we
    compute the reliability of each component as the inverse of its
    variance (a high variance suggests greater uncertainty and thus
    lower reliability).  We also ensure non‑negativity and normalise
    the weights so that they lie on the probability simplex.  The
    resulting weights are then used to form a weighted sum of the
    components.  Finally, the score is clipped to the range [0, 100].

    Parameters
    ----------
    z : list of float
        Raw outputs from the seven modules (aggregated risk, OU mean,
        MC exceedance probability, Bayesian posterior mean,
        max SHAP contribution, adherence estimate, mood).
    variances : list of float
        Estimated variances for each component.  A larger variance
        yields a smaller weight.
    epsilon : float, optional
        Small constant to prevent division by zero.

    Returns
    -------
    weights : np.ndarray
        Normalised weights for combining the module outputs.
    score : float
        The resulting Wellness Score in [0, 100].
    """
    reliabilities = np.array([1.0 / (v + epsilon) for v in variances], dtype=float)
    # If all variances are extremely large or equal, reliabilities may be
    # uniform.  Ensure non‑negativity.
    reliabilities = np.maximum(reliabilities, 0.0)
    if reliabilities.sum() <= 0.0:
        weights = np.ones(len(z)) / len(z)
    else:
        weights = reliabilities / reliabilities.sum()
    # Weighted sum of module outputs.  Map probability outputs to the
    # 0–100 scale to keep units consistent.  For example, the exceedance
    # probability and posterior mean are probabilities in [0, 1], so
    # multiply them by 100.
    z_array = np.array(z, dtype=float)
    # Scale probability‑like measures (indices 2 and 3) to [0, 100].
    z_array[2] *= 100.0  # MC exceedance probability
    z_array[3] *= 100.0  # Bayesian posterior mean
    # SHAP contribution can be negative; centre it on 50 for display
    z_array[4] = 50.0 + z_array[4]
    score = float(np.dot(weights, z_array))
    # Clip to [0, 100]
    score = max(0.0, min(100.0, score))
    return weights, score


def main() -> None:
    """Entry point for the Streamlit app."""
    st.set_page_config(page_title="Gemelo Digital Emocional", layout="wide")

    # Title and introduction
    st.title("Gemelo Digital Emocional")
    st.write(
        "Esta aplicación interactiva demuestra una implementación simplificada "
        "de un **gemelo digital emocional**, basada en la arquitectura de "
        "siete módulos descrita en la documentación técnica. Introduzca sus "
        "datos a continuación para generar un perfil de riesgo, simular "
        "trayectorias emocionales y obtener una puntuación de bienestar en "
        "tiempo real."
    )

    # Show the architecture diagram if available
    try:
        with open("arquitectura_digital_twin.png", "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.image(f"data:image/png;base64,{encoded}", caption="Arquitectura del gemelo digital", use_column_width=True)
    except FileNotFoundError:
        st.warning("No se encontró el diagrama de arquitectura.")

    st.header("1. Ingreso de datos de riesgo")
    st.write(
        "Proporcione sus valores de autoevaluación (0–100) para cada una de "
        "las siguientes áreas. Estos números representan cómo se siente en "
        "este momento respecto a cada factor de riesgo."
    )

    # Collect user inputs for the seven risk domains
    inputs = []  # type: List[float]
    cols = st.columns(2)
    for idx, name in enumerate(CATEGORY_NAMES):
        with cols[idx % 2]:
            value = st.slider(name, min_value=0, max_value=100, value=50, step=1)
            inputs.append(float(value))
    inputs_array = np.array(inputs, dtype=float)

    st.header("2. Adherencia y estado de ánimo")
    st.write(
        "Además del perfil de riesgo, también necesitamos conocer su "
        "adherencia a las recomendaciones y su estado de ánimo general."
    )
    adherence_input = st.slider("Adherencia a las recomendaciones (0–100)", 0, 100, 70, 1)
    mood_input = st.slider("Estado de ánimo actual (0–100)", 0, 100, 60, 1)

    # Initialize or update stateful variables in the session
    if "prev_adherence" not in st.session_state:
        st.session_state.prev_adherence = adherence_input
    if "beta_prior_alpha" not in st.session_state:
        # Weakly informative Beta prior
        st.session_state.beta_prior_alpha = 2.0
        st.session_state.beta_prior_beta = 2.0

    # Process inputs when the user clicks the button
    if st.button("Calcular puntaje de bienestar"):
        # Module 1: Risk classifier
        weights, aggregated_risk, raw_risk = risk_classifier(inputs_array)
        st.subheader("Resultado del módulo 1: Clasificador de riesgo")
        st.write(f"Puntaje de riesgo agregado: **{aggregated_risk:.2f}** (0–100)")
        # Display risk profile as a bar chart
        risk_df = pd.DataFrame(
            {
                "Dominio": CATEGORY_NAMES,
                "Puntaje": raw_risk,
                "Peso": weights,
            }
        )
        st.bar_chart(risk_df.set_index("Dominio")["Puntaje"])

        # Module 2: OU simulation on aggregated risk
        traj, mean_traj, std_traj = ou_simulation(
            x0=aggregated_risk,
            theta=0.5,
            mu=50.0,
            sigma=15.0,
            dt=1.0,
            horizon=SIMULATION_DAYS,
            n_sim=N_SIMULATIONS,
        )
        st.subheader("Resultado del módulo 2: Proceso de Ornstein–Uhlenbeck")
        ou_df = pd.DataFrame({"Día": np.arange(len(mean_traj)), "Media": mean_traj, "Desv": std_traj})
        st.line_chart(ou_df.set_index("Día")["Media"], height=250)
        st.write(
            f"Media final de la simulación: **{mean_traj[-1]:.2f}** ± {std_traj[-1]:.2f} (desviación estándar)"
        )

        # Module 3: Monte Carlo forecast for exceedance probability
        exceed_prob, expected_final = monte_carlo_forecast(traj, threshold=WELLNESS_THRESHOLD)
        st.subheader("Resultado del módulo 3: Simulación Monte Carlo")
        st.write(f"Probabilidad de exceder {WELLNESS_THRESHOLD} en los próximos {SIMULATION_DAYS} días: "
                 f"**{exceed_prob*100:.2f}%**")
        st.write(f"Valor esperado al final del horizonte: **{expected_final:.2f}**")

        # Module 4: Bayesian inference update on exceedance events
        successes = int(exceed_prob * N_SIMULATIONS)
        trials = N_SIMULATIONS
        post_alpha, post_beta, posterior_mean = bayesian_beta_binomial_update(
            st.session_state.beta_prior_alpha,
            st.session_state.beta_prior_beta,
            successes,
            trials,
        )
        # Update session state for next iteration
        st.session_state.beta_prior_alpha = post_alpha
        st.session_state.beta_prior_beta = post_beta
        st.subheader("Resultado del módulo 4: Inferencia bayesiana")
        st.write(
            f"Parámetros posteriores (α, β): ({post_alpha:.1f}, {post_beta:.1f}) \n"
            f"Media posterior de excedencia: **{posterior_mean*100:.2f}%**"
        )

        # Module 5: SHAP‑like explanations
        contributions = shap_like_contributions(raw_risk, BASELINES, weights)
        # Sort contributions for nicer display
        contrib_df = pd.DataFrame(
            {
                "Dominio": CATEGORY_NAMES,
                "Contribución": contributions,
            }
        ).sort_values("Contribución", ascending=False)
        st.subheader("Resultado del módulo 5: Explicaciones (estilo SHAP)")
        st.write(
            "Las siguientes contribuciones indican cuánto influye cada dominio en "
            "la diferencia entre su puntuación actual y la línea base (50). "
            "Valores positivos aumentan el riesgo y valores negativos lo reducen."
        )
        st.bar_chart(contrib_df.set_index("Dominio")["Contribución"])
        max_contribution = float(contributions.max())

        # Module 6: Adherence modelling
        updated_adherence = adherence_update(st.session_state.prev_adherence, adherence_input)
        st.session_state.prev_adherence = updated_adherence
        st.subheader("Resultado del módulo 6: Modelado de adherencia")
        st.write(
            f"Adherencia previa: {st.session_state.prev_adherence:.1f}, "
            f"adherencia actual: {adherence_input:.1f}, "
            f"adherencia actualizada: **{updated_adherence:.1f}**"
        )

        # Module 7: Integration via Γ operator
        z = [
            aggregated_risk,        # risk score
            mean_traj[-1],          # OU mean
            exceed_prob,            # MC exceedance probability (0–1)
            posterior_mean,         # Bayesian posterior mean (0–1)
            max_contribution,       # max SHAP contribution (can be ±)
            updated_adherence,      # adherence (0–100)
            float(mood_input),      # mood (0–100)
        ]
        # Variances correspond to our uncertainty about each module.  For
        # aggregated risk we use 1 (high reliability); for OU we use the
        # variance of the final OU value; for MC we use the sample
        # variance of exceedance indicator; for Bayes we use the posterior
        # variance of a Beta distribution (αβ/((α+β)^2(α+β+1))); for SHAP
        # contributions we use the variance of contributions; for
        # adherence we use a fixed small variance; and for mood we also
        # use a small variance, assuming these are reliable self‑reports.
        beta_var = (post_alpha * post_beta) / (((post_alpha + post_beta) ** 2) * (post_alpha + post_beta + 1))
        variances = [
            1.0,                        # aggregated risk
            float(std_traj[-1] ** 2),   # OU variance at horizon
            exceed_prob * (1 - exceed_prob),  # Bernoulli variance for exceedance
            beta_var,                   # Beta posterior variance
            float(contributions.var()) if len(contributions) > 0 else 1.0,  # SHAP variance
            4.0,                        # adherence variance (fixed small)
            4.0,                        # mood variance (fixed small)
        ]
        weights_gamma, wellness_score = dynamic_gamma_integration(z, variances)
        st.subheader("Resultado del módulo 7: Operador Γ e integración global")
        weights_df = pd.DataFrame(
            {
                "Módulo": ["Riesgo", "OU", "Monte Carlo", "Bayes", "SHAP", "Adherencia", "Ánimo"],
                "Peso": weights_gamma,
            }
        )
        st.write(f"Puntuación de bienestar S(t): **{wellness_score:.2f}**")
        st.bar_chart(weights_df.set_index("Módulo")["Peso"])
        st.write(
            "Los pesos anteriores muestran la confianza relativa en cada módulo. "
            "Los componentes con menor varianza (mayor fiabilidad) reciben un peso "
            "más alto en el cálculo final."
        )

        # Summarise key takeaways
        st.markdown("---")
        st.header("Resumen e interpretación")
        st.write(
            "**Conclusiones principales:**\n"
            "- Su puntuación agregada de riesgo es de {:.2f}. "
            "Los factores con mayor peso fueron: {}.\n"
            "- Según el proceso de OU, se espera que su riesgo tienda hacia {:.2f} en "
            "los próximos {} días.\n"
            "- La simulación de Monte Carlo estima un {:.2f}% de probabilidad de que su "
            "puntuación supere {}.\n"
            "- El modelo bayesiano actualiza la probabilidad de excedencia a {:.2f}%.\n"
            "- La adherencia actualizada es de {:.1f} y su estado de ánimo reportado es de {:.1f}.\n"
            "- La puntuación global de bienestar es {:.2f} sobre 100, lo que {}."
        .format(
            aggregated_risk,
            ", ".join(risk_df.sort_values("Peso", ascending=False)["Dominio"].head(3)),
            mean_traj[-1],
            SIMULATION_DAYS,
            exceed_prob * 100.0,
            WELLNESS_THRESHOLD,
            posterior_mean * 100.0,
            updated_adherence,
            mood_input,
            wellness_score,
            "indica una situación de riesgo elevado" if wellness_score > 70 else "sugiere un estado relativamente estable"
        ))


if __name__ == "__main__":  # pragma: no cover
    main()