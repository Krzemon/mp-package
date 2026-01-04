# The calculations are based on article: https://arxiv.org/pdf/cond-mat/0508341

import numpy as np
from scipy.optimize import root_scalar

def equation_7(Y: float, X: float, weights: list[float], sigmas_squared: list[float], r: float) -> float:
    """
    Oblicza równanie (7) z artykułu dla danego punktu Z = X + iY.

    :param Y: Wartość Im{Z} horyzontu krytycznego
    :param X: Wartość Re{Z} horyzontu krytycznego
    :param weights: lista wag zawartości kolejnych grup 
    :param sigmas_squared: lista wariancji kolejnych grup 
    :param r: stosunek N/T (liczba_wierszy / liczba kolumn; macierzy X)
    :return: residuum równania (7)
    """
    result = 0
    for k in range(len(weights)):
        result += weights[k] * sigmas_squared[k]**2 / ((X - sigmas_squared[k])**2 + Y**2)
    return result - 1/r

def find_critical_horizon(X_values: np.ndarray, weights: list[float], sigmas_squared: list[float], r: float) -> np.ndarray:
    """
    Rozwiązuje równanie (7); obliczenie horyzontu krytycznego.

    :param X_values: tablica (np.ndarray) wartosci X w zakresie teoretycznym widma [lambda_min, lambda_max] 
    :param weights: lista wag zawartości kolejnych grup 
    :param sigmas_squared: lista wariancji kolejnych grup 
    :param r: stosunek N/T (liczba_wierszy / liczba kolumn; macierzy X)
    :return: tablica (np.array) Y = Y(X_values).
    """
    Y_values = []
    for X in X_values:
        def eq7_for_X(Y: float) -> float:
            """
            Funkcja pomocnicza dla danego x = X_values[i]

            :param Y: stosunek N/T (liczba_wierszy / liczba kolumn; macierzy X)
            :return: tablica (np.array) Y = Y(X_values).
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
    Oblicza x(Z) i rho_c(Z) według równań (8) i (9).

    :param X_values: tablica (np.ndarray) wartosci X w zakresie teoretycznym widma [lambda_min, lambda_max] 
    :param Y_values: tablica (np.ndarray) wartosci Y będące rozwiązaniem równania (8) w funkcji wartości X
    :param weights: lista wag zawartości kolejnych grup 
    :param sigmas_squared: lista wariancji kolejnych grup 
    :param r: stosunek N/T (liczba_wierszy / liczba kolumn; macierzy X)
    :return: krotka (tuple) danych: (np.ndarray X, np.ndarray RHO) 
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
            # Usunięto znak minus przed Y, aby gęstość była dodatnia - inna gałąź równania
            rho = Y * rho_sum / (np.pi * x)
            rho_values.append(rho)
        else:
            rho_values.append(0)
    
    return np.array(x_values), np.array(rho_values)

def estimate_spectrum_range(sigmas_squared: list[float], r: float) -> tuple[float, float]:
    """
    Szacuje zakres widma na podstawie wariancji i parametru r.

    :param sigmas_squared: lista wariancji kolejnych grup 
    :param r: stosunek N/T (liczba_wierszy / liczba kolumn; macierzy X)
    :return: krotka (tuple) danych: (lambda_min, lambda_max) 
    """

    sorted_lambdas = sorted(sigmas_squared)
    min_lambda = sorted_lambdas[0]
    max_lambda = sorted_lambdas[-1]

    X_min = min_lambda * (1 - np.sqrt(r))**2
    X_max = max_lambda * (1 + np.sqrt(r))**2
    return X_min, X_max

def theoretical_eigenvalue_distribution(N_list: list[int], T: int, sigmas_squared: list[float], num_points: int=1000) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Oblicza teoretyczne widmo wartości własnych dla macierzy k.

    :param N_list: lista ilości stopni swobody (wierszy) dla grup o tej samej wariancji
    :param T: liczba danych (kolumn) dla każdego stopnia swobody
    :param sigmas_squared: lista wariancji dla każdej grupy
    :param num_points: liczba punktów rysowania krzywej teoretycznej
    :return: krotka (tuple) danych: (posortowana np.ndarray X, posortowana np.ndarray PDF, słownik zawierający statystyki) 
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