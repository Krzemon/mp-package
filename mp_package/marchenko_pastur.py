# The calculations are based on article: https://arxiv.org/pdf/cond-mat/0508341

import numpy as np
from scipy.optimize import root_scalar

def generate_X(N: int, T: int, N1: int, sigmas_squared: list[float]) -> np.ndarray:
    """
    Generuje macierz danych X z dwoma grupami o roznych wariancjach.
    """
    N2 = N - N1
    X = np.zeros((N, T))
    X[:N1, :] = np.random.normal(0, np.sqrt(sigmas_squared[0]), (N1, T))
    X[N1:, :] = np.random.normal(0, np.sqrt(sigmas_squared[1]), (N2, T))
    return X

def equation_7(Y: float, X: float, weights: list[float], sigmas_squared: list[float], r: float) -> float:
    """
    Rownanie (7) z artykulu dla danego punktu Z = X + iY.
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
    # print(f"Zakres widma: X_min = {X_min:.4f}, X_max = {X_max:.4f}")
    return X_min, X_max

def theoretical_eigenvalue_distribution(N: int, T: int, N1: int, sigmas_squared: list[float], num_points: int=1000) -> tuple[np.ndarray, np.ndarray]:
    """
    Oblicza teoretyczny rozklad wartosci wlasnych dla macierzy korelacji.
    """
    N2 = N - N1
    weights = [N1/N, N2/N]
    r = N/T
    
    X_min, X_max = estimate_spectrum_range(sigmas_squared, r)
    X_values = np.linspace(X_min, X_max, num_points)
    Y_values = find_critical_horizon(X_values, weights, sigmas_squared, r)
    x_values, rho_values = calculate_x_and_rho(X_values, Y_values, weights, sigmas_squared, r)
    
    sorted_indices = np.argsort(x_values)
    x_sorted = x_values[sorted_indices]
    rho_sorted = rho_values[sorted_indices]
    
    return x_sorted, rho_sorted

def generate_eigenvalues_batch(N: int, T: int, N1: int, sigmas_squared: list[float], batch_size: int) -> list:
    """
    Generuje wartosci wlasne w partiach, aby efektywnie zarzadzac pamiecia.
    """
    eigenvalues = []
    for _ in range(batch_size):
        X = generate_X(N, T, N1, sigmas_squared)
        C = (1/T) * X @ X.T
        eigenvalues.extend(np.linalg.eigvalsh(C))
    return eigenvalues



#############   input

# N1 = 10
# N2 = 10
# N = N1 + N2
# sigmas_squared = [1.0, 4.0]
# num_trials = 100000 # liczba wywolan
# batch_size = 10000 # rozmiar batchu
# bins=100 # liczba binow 
# T_list = [40, 100, 400, 2000]


###############   obliczenia 


# fig, axs = plt.subplots(2, 2, figsize=(16, 12))
# axs = axs.flatten()

# for idx, T in enumerate(T_list):
#     print(f"\n==== T = {T} ====")

#     x_theo, rho_theo = theoretical_eigenvalue_distribution(N, T, N1, sigmas_squared, num_points=1000)
#     if len(x_theo) == 0 or len(rho_theo) == 0:
#         print("BŁĄD: Nie udało się obliczyć teoretycznego rozkładu!")
#         continue

#     all_eigenvalues = []
#     num_batches = num_trials // batch_size
#     start_time = time.time()
#     for i in tqdm(range(num_batches), desc=f"T={T}"):
#         batch_eigenvalues = generate_eigenvalues_batch(N, T, N1, sigmas_squared, batch_size)
#         all_eigenvalues.extend(batch_eigenvalues)

#     elapsed_time = (time.time() - start_time) / 60
#     print(f"Wygenerowano {len(all_eigenvalues)} wartości własnych w czasie {elapsed_time:.1f} min")

#     ax = axs[idx]
#     ax.hist(all_eigenvalues, bins=bins, density=True, alpha=0.6, color='skyblue', label=f'Histogram dla {num_trials} prób')
#     ax.plot(x_theo, rho_theo, 'r-', linewidth=3, label='Teoretyczny rozkład')

#     line1 = ax.axvline(x=sigmas_squared[0], color='green', linewidth=2, linestyle='--', alpha=0.8)
#     line2 = ax.axvline(x=sigmas_squared[1], color='green', linewidth=2, linestyle='--', alpha=0.8)

#     ax.set_title(f'N={N}, T={T}, r={N/T:.4f}', fontsize=14)
#     ax.set_xlabel('Wartość własna', fontsize=12)
#     ax.set_ylabel('Gęstość prawdopodobieństwa', fontsize=12)
#     ax.legend(fontsize=10)
#     ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()