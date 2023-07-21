import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List


def generate_population_data(loc: float, scale: float, size: int, 
                             skewness: float = None, skewness_type: str = None) -> np.ndarray:
    """
    Generate population data from a normal or skewed normal distribution.

    Args:
    - loc (float)           : mean of the distribution.
    - scale (float)         : standard deviation of the distribution.
    - size (int)            : number of data points to generate.
    - skewness (float)      : the degree of skewness of the distribution (default=None).
    - skewness_type (str)   : if 'n' would be negatifly skewed, 'p' positifly skewed, 
                              otherwise error.

    Returns:
    - data_pop (numpy.ndarray): generated population data.
    """
    if skewness is None:
        # Generate population data from a normal distribution
        data_pop = np.random.normal(loc=loc, scale=scale, size=size)
    else:
        # Generate population data from a skewed normal distribution
        # a = skewness / np.sqrt(1 + skewness**2)
        # data_pop = loc - scale * stats.skewnorm.rvs(a, size=size)
        if skewness_type == 'p':
            data_pop = loc - scale*stats.skewnorm.rvs(a=10, scale=scale, size=10000)
        elif skewness_type == 'n':
            data_pop = loc - scale*stats.skewnorm.rvs(a=-10, scale=scale, size=10000)
        else:
            raise('It should be p or n')

    return data_pop


def generate_sample(data_pop: pd.DataFrame, n_sampling: int, sample_size: int) -> List:
    """
    Generate random sampling from a given population.

    Args:
    - data_pop (DataFrame): population data.
    - n_sampling (int)    : number of sampling performed.
    - sample_size (int)   : number of sample taken from population in each sampling.

    Returns:
    - mean_sample_list (list): list of mean sample in every sampling.
    """
    mean_sample_list = []

    for i in range(n_sampling):
        mean_sample_i = data_pop.sample(n=sample_size, replace=False)["data"].mean()
        mean_sample_list.append(mean_sample_i)

    return mean_sample_list


def plot_population_data(data_pop: pd.DataFrame, col_name: str) -> None:
    """
    Plot histogram of population data.

    Args:
    - data_pop (DataFrame): population data.
    - col_name (str)      : name of column in the dataframs.

    Returns:
    - None
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
    sns.histplot(data_pop[col_name], stat="density", ax=ax)
    ax.set_xlabel("data", fontsize=16)
    ax.set_ylabel("density", fontsize=16)
    plt.show()


if __name__ == '__main__':
    data_pop = generate_population_data(loc=90, scale=10, size=10000, skewness=180, skewness_type='p')

    df = pd.DataFrame(data = data_pop,
                            columns = ["data"])
    
    samples = generate_sample(df, 3000, 30)

    samples_df = pd.DataFrame(data = samples,
                            columns = ["data"])
    
    print(f'rata-data: {samples_df.mean()}')
    print(f'std: {samples_df.std()}')

    plot_population_data(samples_df, 'data')