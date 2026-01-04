import numpy as np

distributions = {
    "normal": lambda shape, var: np.random.normal(loc=0, scale=np.sqrt(var), size=shape),
    "uniform": lambda shape, var: np.random.uniform(low=-np.sqrt(3*var), high=np.sqrt(3*var), size=shape),
    "exponential": lambda shape, var: np.random.exponential(scale=np.sqrt(var/2), size=shape) * np.random.choice([-1, 1], size=shape),
    "bernoulli": lambda shape, var: np.random.choice([-1, 1], size=shape) * np.sqrt(var),
}

def generate_random(dist_name, shape, var):
    return distributions[dist_name](shape, var)

def generate_eigenvalues(N_list: list[int], T: int, sigmas_squared: list[float], num_trials: int, dist_name: str = "normal") -> tuple[np.ndarray, list[float]]:
    """
    Oblicza wartości własne i statystyki rozkładu.

    :param N_list: lista ilości stopni swobody (wierszy) dla grup o tej samej wariancji
    :param T: liczba danych (kolumn) dla każdego stopnia swobody
    :param sigmas_squared: lista wariancji dla każdej grupy
    :param num_trials: liczba wywołań pętli do wygenerowania wartości własnych
    :param dist_name: nazwa rozkładu (centrowanego) z jakiego losowane są dane
    :return: krotka (tuple) danych: (wszystkie wartości własne, słownik zawierający statystyki) 
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

def generate_X(N_list: list[int], T: int, sigmas_squared: list[float], dist_name: str) -> np.ndarray:
    """
    Generuje macierz danych X z grupami o różnych wariancjach.

    :param N_list: lista ilości stopni swobody (wierszy) dla grup o tej samej wariancji
    :param T: liczba danych (kolumn) dla każdego stopnia swobody
    :param sigmas_squared: lista wariancji dla każdej grupy
    :param dist_name: nazwa rozkładu (centrowanego) z jakiego losowane są dane
    :return: macierz (np.ndarray) wymiaru NxT
    """
    N_total = sum(N_list)
    X = np.zeros((N_total, T))
    start_row = 0
    for n_rows, sigma_sq in zip(N_list, sigmas_squared):
        end_row = start_row + n_rows
        X[start_row:end_row, :] = generate_random(dist_name, shape=(n_rows, T), var=sigma_sq)
        start_row = end_row
    return X