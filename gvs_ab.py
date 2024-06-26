"""
This module helps to execute ab test and more.

Created by vsgrabchuk.
"""


import pandas as pd
import numpy as np
from scipy import stats as ss
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

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
    # Обобщить применение теста


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
        boot_data.append(statistic_func(ss2)-statistic_func(ss1))

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
    # Вместо is_0_in_ci использовать is_h0, как аналог p-value
    # Добавить возможность выбрать статтест для сравнения подвыборок
    # Загнать визуализацию в отдельную функцию?


def poisson_bootstrap_ctr(
    likes_1, 
    views_1, 
    likes_2, 
    views_2, 
    n_simulations=5000,
    boost=True,
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
    boost: bool, default True
        Ускорение вычислений
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
    if boost:  # Использование фишек numpy для ускорения
        weights_1 = ss.poisson(1).rvs(
            (n_simulations, len(likes_1)))
        weights_2 = ss.poisson(1).rvs(
            (n_simulations, len(likes_2)))
        CTR_1 = (weights_1@likes_1) / (weights_1@views_1)
        CTR_2 = (weights_2@likes_2) / (weights_2@views_2)
    else:  # Работает медленней (в 2 раза?), но экономит память
        CTR_1 = []
        CTR_2 = []
        n_1 = len(likes_1)
        n_2 = len(likes_2)
        for _ in range(n_simulations):
            weights_1 = ss.poisson(1).rvs(n_1)
            weights_2 = ss.poisson(1).rvs(n_2)
            CTR_1.append( (weights_1@likes_1) / (weights_1@views_1) )
            CTR_2.append( (weights_2@likes_2) / (weights_2@views_2) )
        CTR_1 = np.asarray(CTR_1)
        CTR_2 = np.asarray(CTR_2)

    boot_data = pd.Series(CTR_2 - CTR_1)
    
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
    # Вместо is_0_in_ci использовать is_h0, как аналог p-value
    # Можно ли сделать универсальней? (не только для CTR)
    # Загнать визуализацию в отдельную функцию?


def bucketization(
    df,
    column,
    buckets,
    print_info=True
):
    """
    Функция добавляет в df колонку с бакетом, используя хеш от column

    Parameters:
    -----------
    df: pandas.DataFrame
        Таблица, которю необходимо модифицировать
    buckets: int
        Количество бакетов
    column: str
        Колонка, в соответствии с которой формируется бакет (считается хеш)
        Если указано None, то бакет считается рандомно
    print_info: bool, default True
        Отображение инфы по бакетам

    Returns:
    --------
    new_df: pandas.DataFrame
        Новый df с дополнительной колонкой bucket
    """
    new_df = pd.DataFrame(df)
    if column is None:
        new_df['bucket'] = ss.randint.rvs(low=0, high=buckets, size=new_df.shape[0])
    else:
        new_df['bucket'] = df[column].swifter.apply(lambda x: hash(x)%buckets)
    
    # Проверка корректности распределения наблюдений по бакетам
    if print_info:
        print(f'bucket_size ~ {round(new_df.shape[0] / buckets)}')
        sns.histplot(new_df.bucket, bins=buckets)
        plt.xlabel('bucket')
        plt.ylabel('freq')

    return new_df
    # Сделать ice-bucket test?


def laplace_smoothing(x, y, gamma=5.):
    """
    Функция реализует сглаживание Лапласа для поэлементной метрики отношения x/y
    
    Parameters:
    -----------
    x: array-like
        Числитель для вычисления поэлементной метрики
    y: array-like
        Знаменатель для вычисления поэлементной метрики
    gamma: float, default 5.
        Гиперпараметр для сглаживания
    
    Returns:
    --------
        ratios: np.array
            Сглаженная поэлементная метрика
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    global_ratio = x.sum() / y.sum()
    
    ratios = (x+gamma*global_ratio) / (y+gamma)
    
    return ratios


def linearization_ratio(x_0, y_0, x_1, y_1):
    """
    Функция линеаризует числители для метрики отношения x/y 
    (для контрольной и тестовой выборок)
    
    Parameters:
    -----------
    x_0: array-like
        Вектор измерений, попадающих в числитель формулы для вычисления метрики отношения
        Контрольная группа
    y_0: array-like
        Вектор измерений, попадающих в знаменатель формулы для вычисления метрики отношения
        Контрольная группа
    x_1: array-like
        Вектор измерений, попадающих в числитель формулы для вычисления метрики отношения
        Тестовая группа
    y_1: array-like
        Вектор измерений, попадающих в знаменатель формулы для вычисления метрики отношения
        Тестовая группа
        
    Returns:
    --------
    x_0_lin: array-like
        Вектор линеаризованных измерений числителя
        Контрольная группа
    x_1_lin: array-like
        Вектор линеаризованных измерений числителя
        Тестовая группа
    """
    x_0 = np.asarray(x_0)
    x_1 = np.asarray(x_1)
    y_0 = np.asarray(y_0)
    y_1 = np.asarray(y_1)
    
    k = x_0.sum() / y_0.sum()
    
    x_0_lin = x_0 - k * y_0
    x_1_lin = x_1 - k * y_1
    
    return x_0_lin, x_1_lin


def get_ttest_sample_size(eps, std_1, std_2, alpha=0.05, beta=0.2):
    """
    Функция для расчёта размера выборки при использовании t-test
    
    Parameters:
    -----------
    eps: float
        MDE
    std_1: float
        std 1-й выборки
    std_2: float
        std 2-й выборки
    alpha: float, default 0.05
        Допустимый FPR
    beta: float, default 0.2
        Допустимый FNR
        
    Returns:
    --------
    sample_size: int
        Требуемый размер выборки для детектирования заданного MDE
        с учётом допустимых вероятностей ошибок
    """
    ppf_alpha = ss.norm.ppf(1 - alpha/2, loc=0, scale=1)
    ppf_beta = ss.norm.ppf(1 - beta, loc=0, scale=1)
    
    z_scores_sum_squared = (ppf_alpha + ppf_beta) ** 2
    
    sample_size = int(
        np.ceil(
            z_scores_sum_squared * (std_1**2 + std_2**2) / (eps**2)
        )
    )
    
    return sample_size


def func_for_mtx_rows(func, *args, cur_func_res_idx=None, cur_func_print_info=False, **kwargs):
    '''
    Функция позволяет использовать матрицы в качестве входных данных для обычных функций
    
    Parameters:
    -----------
    func: callable
        Используемая функция
    args: nd.array[nxm]
        Матрицы, строки которых - аргументы для func
    cur_func_res_idx: int, default None
        Индекс элемента кортежа, возвращаемого func. Элемент будет возвращён текущей функцией
    cur_func_print_info=False: bool, default
        Отображение статусбара
    kwargs
        Именованные аргументы для func
    
    Returns:
    --------
    np.array
        Набор возвращаемых значений func


    Examples:
    ---------
    func_for_mtx_rows(scipy.stats.ttest_ind, mtx_a, mtx_b)
    '''
    if cur_func_print_info:
        data = tqdm(zip(*args))
    else:
        data = zip(*args)


    if cur_func_res_idx is None:
        return np.asarray([func(*func_args, **kwargs) for func_args in data])
    else:
        return np.asarray([func(*func_args, **kwargs)[cur_func_res_idx] for func_args in data])



def ttest_vect(a, b, equal_var=True):
    """
    Реализация векторизованного t-test'а
    *Почему-то не ускоряет вычисления :(
    
    Parameters:
    -----------
    a: np.array[nxm]
        Массив выборок "a"
    b: np.array[nxm]
        Массив выборок "b"
    equal_var: bool, default True
    
    Returns:
    --------
    t: np.array
        Значения t-статистики
    pvals: np.array
        Значения p-value
    """
    
    n_a = a.shape[1]
    n_b = b.shape[1]

    mean_a = a.mean(axis=1)
    mean_b = b.mean(axis=1)
    
    # Долго считается дисперсия
    var_a = a.var(axis=1, ddof=1)
    var_b = b.var(axis=1, ddof=1)

    if equal_var:
        df = n_a + n_b - 2
        se = np.sqrt(
            ((n_a-1)*var_a + (n_b-1)*var_b) * (1/n_a + 1/n_b) / df
        )
    else:
        df = (var_a/n_a + var_b/n_b)**2 / ( (var_a/n_a)**2/(n_a-1) + (var_b/n_b)**2/(n_b-1) )
        se = np.sqrt(var_a/n_a + var_b/n_b)
    
    t = (mean_a - mean_b) / se
    
    t_dist = ss.t(df)
    pvals = 2 * t_dist.cdf(-np.abs(t))
    
    return t, pvals