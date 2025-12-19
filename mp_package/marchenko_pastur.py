# The calculations are based on article: https://arxiv.org/pdf/cond-mat/0508341

import numpy as np
from scipy.optimize import root_scalar

distributions = {
    "normal": lambda shape, var: np.random.normal(loc=0, scale=np.sqrt(var), size=shape),
    "uniform": lambda shape, var: np.random.uniform(low=-np.sqrt(3*var), high=np.sqrt(3*var), size=shape),
    "exponential": lambda shape, var: np.random.exponential(scale=np.sqrt(var), size=shape),
    "cauchy": lambda shape, var: np.random.standard_cauchy(size=shape) * np.sqrt(var),
    "log-normal": lambda shape, var: np.random.lognormal(mean=0, sigma=np.sqrt(np.log(var + 1)), size=shape),
    "bernoulli": lambda shape, var: np.random.choice([-1, 1], size=shape) * np.sqrt(var),
    "t-Student": lambda shape, var: np.random.standard_t(df=3, size=shape) * np.sqrt(var / 3),
}

def generate_random(dist_name, shape, var):
    return distributions[dist_name](shape, var)

def generate_X(N_list: list[int], T: int, sigmas_squared: list[float], dist_name: str) -> np.ndarray:
    """
    Generuje macierz danych X z wieloma grupami o różnych wariancjach.

    :param N_list: lista liczby wierszy dla kolejnych grup
    :param sigmas_squared: lista wariancji dla kolejnych grup 
    :return: macierz wymiaru NxT
    """
    N_total = sum(N_list)
    X = np.zeros((N_total, T))
    start_row = 0
    for n_rows, sigma_sq in zip(N_list, sigmas_squared):
        end_row = start_row + n_rows
        # X[start_row:end_row, :] = np.random.normal(0, np.sqrt(sigma_sq), size=(n_rows, T))
        X[start_row:end_row, :] = generate_random(dist_name, shape=(n_rows, T), var=sigma_sq)
        start_row = end_row
    return X

def generate_eigenvalues(N_list: list[int], T: int, sigmas_squared: list[float], num_trials: int, dist_name: str = "normal") -> tuple[np.ndarray, list[float]]:
    """
    Generuje wartości własne.
    N_list[i] - liczba wierszy dla grupy i
    T - liczba obserwacji 
    sigmas_squared[i] - wariancja dla grupy i
    num_trials - liczba wywołań
    """
    N_total = sum(N_list)
    all_eigenvalues = np.empty(num_trials * N_total)

    for i in range(num_trials):
        X = generate_X(N_list, T, sigmas_squared, dist_name)
        C = (1 / T) * X @ X.T
        all_eigenvalues[i * N_total : (i + 1) * N_total] = np.linalg.eigvalsh(C)
    mean = np.mean(all_eigenvalues)
    var = np.var(all_eigenvalues)
    stats_dict = {
        "mean": mean,
        "variance": var,
        "skewness": np.mean((all_eigenvalues - mean)**3) / var**1.5, 
        "kurtosis": np.mean((all_eigenvalues - mean)**4) / var**2, 
        "min": np.min(all_eigenvalues), 
        "max": np.max(all_eigenvalues)
    }

    return all_eigenvalues, stats_dict

def equation_7(Y: float, X: float, weights: list[float], sigmas_squared: list[float], r: float) -> float:
    """
    Równanie (7) z artykułu dla danego punktu Z = X + iY.
    """
    result = 0
    for k in range(len(weights)):
        result += weights[k] * sigmas_squared[k]**2 / ((X - sigmas_squared[k])**2 + Y**2)
    return result - 1/r

def find_critical_horizon(X_values: np.ndarray, weights: list[float], sigmas_squared: list[float], r: float) -> np.ndarray:
    """
    Rozwiazuje rownanie (7).
    """
    Y_values = []
    for X in X_values:
        def eq7_for_X(Y: float) -> float:
            """
            Rozwiazuje rownanie (7) dla danego X.
            """
            return equation_7(Y, X, weights, sigmas_squared, r)
        if eq7_for_X(0) <= 0:
            Y_values.append(0)
            continue
        result = root_scalar(eq7_for_X, bracket=[1e-10, 1000], method='brentq')
        Y_values.append(result.root)
    return np.array(Y_values)

def calculate_x_and_rho(X_values: np.ndarray, Y_values: np.ndarray, weights: list[float], sigmas_squared: list[float], r: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Oblicza x(Z) i rho_c(Z) wedlug rownan (8) i (9).
    """
    x_values = []
    rho_values = []
    
    for i, (X, Y) in enumerate(zip(X_values, Y_values)):
        if np.isnan(Y):
            x_values.append(np.nan)
            rho_values.append(np.nan)
            continue
        term1 = X
        term2 = 0
        term3 = 0
        
        for k in range(len(weights)):
            term2 += weights[k] * sigmas_squared[k]
            term3 += weights[k] * sigmas_squared[k]**2 * (X - sigmas_squared[k]) / ((X - sigmas_squared[k])**2 + Y**2)
        term2 *= r 
        term3 *= r
        x = term1 + term2 + term3
        x_values.append(x)

        if Y > 0 and x > 0:
            rho_sum = 0
            for k in range(len(weights)):
                rho_sum += weights[k] * sigmas_squared[k] / ((X - sigmas_squared[k])**2 + Y**2)
            # Usunieto znak minus przed Y, aby gestosc byla dodatnia - inna galaz rownania
            rho = Y * rho_sum / (np.pi * x)
            rho_values.append(rho)
        else:
            rho_values.append(0)
    
    return np.array(x_values), np.array(rho_values)



###########


def estimate_spectrum_range(sigmas_squared: list[float], r: float) -> tuple[float, float]:
    """
    Szacuje zakres widma na podstawie wartosci wlasnych i parametru r.
    """
    sorted_lambdas = sorted(sigmas_squared)
    min_lambda = sorted_lambdas[0]
    max_lambda = sorted_lambdas[-1]

    X_min = min_lambda * (1 - np.sqrt(r))**2
    X_max = max_lambda * (1 + np.sqrt(r))**2
    return X_min, X_max

def theoretical_eigenvalue_distribution(N_list: list[int], T: int, sigmas_squared: list[float], num_points: int=1000) -> tuple[np.ndarray, np.ndarray]:
    """
    Oblicza teoretyczny rozklad wartosci wlasnych dla macierzy korelacji.
    """
    N_total = sum(N_list)
    weights = [n / N_total for n in N_list]
    r = N_total / T

    X_min, X_max = estimate_spectrum_range(sigmas_squared, r)
    X_values = np.linspace(X_min, X_max, num_points)
    Y_values = find_critical_horizon(X_values, weights, sigmas_squared, r)
    x_values, rho_values = calculate_x_and_rho(X_values, Y_values, weights, sigmas_squared, r)
    
    sorted_indices = np.argsort(x_values)
    x_sorted = x_values[sorted_indices]
    rho_sorted = rho_values[sorted_indices]
    
    mask = np.abs(rho_sorted) > 1e-10
    mean = np.mean(rho_sorted[mask])
    var  = np.var(rho_sorted[mask])
    stats_dict = {
        "mean": mean,
        "variance": var,
        "skewness": np.mean((rho_sorted - mean)**3) / var**1.5, 
        "kurtosis": np.mean((rho_sorted - mean)**4) / var**2, 
        "min": X_min, 
        "max": X_max
    }
    return x_sorted, rho_sorted, stats_dict