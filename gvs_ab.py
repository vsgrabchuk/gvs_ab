import pandas as pd
import numpy as np
from scipy import stats as ss
import statsmodels.api as sm

import matplotlib.pyplot as plt

from tqdm.auto import tqdm


def aa_test(
    s1,
    s2,
    ss_percent=10,
    n_simulations=10000,
    alpha=0.05,
    test='t',
    print_info=True,
    bins=20
):
    '''
    Функция, реализующая A/A тест для двух выборок
    
    Parameters
    ----------
    s1: pandas.Series
        Выборка 1 (sample)
    s2: pandas.Series
        Выборка 2
    ss_percent: float, default 10
        Процент от выборки (min размера) для составления подвыборки (subsample)
    n_simulations: int, default 10000
        Количество симуляций
    test: str, default 't'
        Статистический тест
        't' - t-тест
        'u' - тест Манна-Уитни
    alpha: float, default 0.05
        Уровень значимости
    print_info: bool, default True
        Флаг отображения информации (текст+графики)
    bins: int, default 20
        Количество bin'ов для отображения гистограммы
    
    Returns
    -------
    fpr: float
        FPR
    '''
    n_s_min = min(len(s1), len(s2))  # Минимальный размер из выборок
    n_ss = round(n_s_min * ss_percent / 100)  # Количество элементов в подвыборке
    
    p_vals = []  # Список с p-values
    
    # Цикл симуляций
    my_range = range(n_simulations)
    if print_info:
        my_range = tqdm(my_range)
    for i in my_range:
        ss1 = s1.sample(n_ss, replace=False)
        ss2 = s2.sample(n_ss, replace=False)
        # Сравнение подвыборок
        if test == 't':  # t-тест с поправкой Уэлча
            test_res = ss.ttest_ind(ss1, ss2, equal_var=False)
        elif test == 'u':  # U-тест
            test_res = ss.mannwhitneyu(ss1, ss2)
            
        p_vals.append(test_res[1]) # Сохраняем p-value
        
    # FPR
    fpr = sum(np.array(p_vals) < alpha) / n_simulations
    
    # Визулаилзация распределения p-values
    if print_info:
        print('min sample size:', n_s_min)
        print('synthetic subsample size:', n_ss)
        plt.style.use('ggplot')
        _, _, bars = plt.hist(p_vals, bins=bins)
        for bar in bars:
            bar.set_edgecolor('black')
            if bar.get_x() < alpha:
                # Статзначимая разница
                bar.set_facecolor('#f74a64')
            else:
                bar.set_facecolor('grey')
        plt.xlabel('p-values')
        plt.ylabel('frequency')
        plt.title(f"FPR: {fpr}") 
        
        sm.qqplot(np.array(p_vals), dist=ss.uniform, line="45")

        plt.show()

    return fpr


def bootstrap_test(
    s1,
    s2,
    n_simulations=5000,
    statistic_func=np.mean,
    alpha=0.05,
    bins=100,
    p_val_evaluation=False,
    print_info=True
):
    '''
    Функция для проверки гипотез с помощью bootstrap

    Parameters
    ----------
    s1: pandas.Series
        Выборка 1
    s2: pandas.Series
        Выборка 2
    n_simulations: int, default 5000
        Количество симуляций
    statistic_func: callable, default np.mean
        Функция для вычисления интересующей статистики
    alpha: float, default 0.05
        Уровень значимости
    bins: int, default 100
        Количество столбиков для гистограммы
    p_val_evaluation: bool, default False
        Оценка p-value с помощью нормального распределения
    print_info: bool, default True
        Вывод на экран информации (в т.ч. графика)

    Returns
    -------
    is_0_in_ci: bool
        Находится ли 0 в ДИ
    boot_data: pandas.Series
        Расрпределение разницы метрик бутстрап выборок
    quantiles: pandas.Series
        Значения квантилей, ограничивающих ДИ
    '''
    # Размер бутстрап подвыборок
    ss1_size = len(s1)
    ss2_size = len(s2)

    # Распределение разницы статистик подвыборок
    boot_data = []
    my_range = range(n_simulations)
    if print_info:
        my_range = tqdm(my_range)
    for i in my_range:  # извлекаем подвыборки
        ss1 = s1.sample(ss1_size, replace=True).values
        ss2 = s2.sample(ss2_size, replace=True).values
        boot_data.append(statistic_func(ss1)-statistic_func(ss2))

    boot_data = pd.Series(boot_data)

    # Вычисление квантилей
    left_quant = alpha / 2
    right_quant = 1 - left_quant
    quantiles = boot_data.quantile([left_quant, right_quant])
    quantiles.columns = ['value']

    # Статзначимость отличий выборок
    if quantiles.iloc[0] <= 0 <= quantiles.iloc[1]:
        is_0_in_ci = True
    else:
        is_0_in_ci = False

    # Вычисление p-value
    # (из предположения, что распределение разницы статистик подвыборок - нормальное)
    if p_val_evaluation:
        p_1 = ss.norm.cdf(
            x=0,
            loc=boot_data.mean(),
            scale=boot_data.std()
        )
        p_2 = ss.norm.cdf(
            x=0,
            loc=-boot_data.mean(),
            scale=boot_data.std()
        )
        p_value = min(p_1, p_2) * 2  # Двустороння вероятность нулевых отличий распределений

    # Визуализация
    if print_info:
        hist_ys = []
        _, _, bars = plt.hist(boot_data, bins=bins)
        for bar in bars:
            hist_ys.append(bar.get_height())
            bar.set_edgecolor('black')
            if quantiles.iloc[0] <= bar.get_x() <= quantiles.iloc[1]:
                # Столбик в ДИ
                bar.set_facecolor('grey')
            else:
                bar.set_facecolor('#f74a64')
        plt.style.use('ggplot')
        plt.vlines(quantiles, ymin=0, ymax=max(hist_ys), linestyle='--', colors='black')  # Отображение квантилей
        if quantiles.iloc[0] <= 0 <= quantiles.iloc[1]:  # Подсветка нуля
            plt.vlines(0, ymin=0, ymax=max(hist_ys), colors='#40d140')  # H0
        else:
            plt.vlines(0, ymin=0, ymax=max(hist_ys), colors='#f74a64')  # H1
        plt.xlabel('metric difference')
        plt.ylabel('frequency')
        plt.title("Subsamples metric difference")
        plt.show()
        if p_val_evaluation:
            print(f'p-value: {p_value}')

    return is_0_in_ci, boot_data, quantiles


def poisson_bootstrap_ctr(
    likes_1, 
    views_1, 
    likes_2, 
    views_2, 
    n_simulations=5000,
    alpha=0.05,
    bins=100,
    print_info=True
):
    """
    Функция реализует пуассоновский бутстрап
    
    Parameters:
    -----------
    likes_1: array-like
        Вектор, элементы которого - число лайков каждого пользователя выборки 1
    likes_2: array-like
        Вектор, элементы которого - число лайков каждого пользователя выборки 2
    views_1: array-like
        Вектор, элементы которого - число просмотров каждого пользователя выборки 1
    views_2: array-like
        Вектор, элементы которого - число просмотров каждого пользователя выборки 2
    n_simulations: int, default 5000
        Количество симуляций
    alpha: float, default 0.05
        Уровень значимости
    bins: int, default 100
        Количество столбиков для гистограммы
    print_info: bool, default True
        Вывод на экран информации (в т.ч. графика)
        
    Returns:
    --------
    is_0_in_ci: bool
        Находится ли 0 в ДИ
    boot_data: pandas.Series
        Расрпределение разницы метрик бутстрап выборок
    quantiles: pandas.Series
        Значения квантилей, ограничивающих ДИ
    """
    likes_1 = np.asarray(likes_1)
    likes_2 = np.asarray(likes_2)
    views_1 = np.asarray(views_1)
    views_2 = np.asarray(views_2)
    
    # Непосредственно вычисления
    weights_1 = stats.poisson(1).rvs(
        (n_simulations, len(likes_1)))
    weights_2 = stats.poisson(1).rvs(
        (n_simulations, len(likes_2)))

    CTR_1 = (weights_1@likes_1) / (weights_1@views_1)
    CTR_2 = (weights_2@likes_2) / (weights_2@views_2)
    
    boot_data = pd.Series(CTR_1 - CTR_2)
    
    # Вычисление квантилей
    left_quant = alpha / 2
    right_quant = 1 - left_quant
    quantiles = boot_data.quantile([left_quant, right_quant])
    quantiles.columns = ['value']

    # Статзначимость отличий выборок
    if quantiles.iloc[0] <= 0 <= quantiles.iloc[1]:
        is_0_in_ci = True
    else:
        is_0_in_ci = False

    # Визуализация
    if print_info:
        hist_ys = []
        _, _, bars = plt.hist(boot_data, bins=bins)
        for bar in bars:
            hist_ys.append(bar.get_height())
            bar.set_edgecolor('black')
            if quantiles.iloc[0] <= bar.get_x() <= quantiles.iloc[1]:
                # Столбик в ДИ
                bar.set_facecolor('grey')
            else:
                bar.set_facecolor('#f74a64')
        plt.style.use('ggplot')
        plt.vlines(quantiles, ymin=0, ymax=max(hist_ys), linestyle='--', colors='black')  # Отображение квантилей
        if quantiles.iloc[0] <= 0 <= quantiles.iloc[1]:  # Подсветка нуля
            plt.vlines(0, ymin=0, ymax=max(hist_ys), colors='#40d140')  # H0
        else:
            plt.vlines(0, ymin=0, ymax=max(hist_ys), colors='#f74a64')  # H1
        plt.xlabel('metric difference')
        plt.ylabel('frequency')
        plt.title("Subsamples metric difference")
        plt.show()

    return is_0_in_ci, boot_data, quantiles