from dataclasses import dataclass, field
from scipy.spatial import distance
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.random import PCG64, Generator


@dataclass
class CLARANSConfig:
    """
    Это класс для задания конфигурации класса CLARANS
    ---
        Параметры:
    ===
        Конфигурация алгоритма
    ---
        num_k (int, default=2): желаемое количество кластеров 
    ---
        maxneighbour (int, default=10): Критерий остановки одной итерации спуска
        Сколько попыток нахождения лучшего соседа предпримет алгоритм перед переходом
        на следующую итерацию
    ---
        numlocal (int, default=10): Сколько итераций нахождения локального минимума предпримет алгоритм
        перед остановкой
    ---
        random_state (float, optional): случайное состояние, следует заполнять для детерминированности
        результата
    ===
        Предобработка
    ---
        y_col (str, optional): Сюда следует указать название target колонки, если она будет присутсвовать
        в изначальных данных
    ===
        Трансляция - Инициализация через существующее решение
    --- 
        mincost (float, optional): первый параметр, возвращаемый из get_translation - обозначает
        текущее лучшее решение
    ---
        best_node: (dict, optional): Словарь нодов текущего лучшего решения
    ===
        Режимы отладки
    ---
        verbose (Bool, optional): лёгкий режим отладки с комментариями по ключевым моментам алгоритма
    ---
        full_debug (Bool, optional): полное комментирование дейсвий алгоритма
    """
    # Конфигурация алгоритма
    num_k: int = field(default=2)
    maxneighbour: int = field(default=10)
    numlocal: int = field(default=10)
    random_state: float = field(default=np.random.random())
    # Предобработка данных
    y_col: str = field(default=None)
    # Трансляция
    mincost: float = field(default=np.inf)
    best_node: dict = field(default_factory=dict)
    # Отладка
    verbose: bool = field(default=False)
    full_debug: bool = field(default=False)


class CLARANS:
    def __init__(self, 
                 distance_function = distance.euclidean, 
                 params: CLARANSConfig = CLARANSConfig()):
        """Конструктор объекта класса CLARANS

        Параметры:
        ===
            distance_function (function(coords1, coords2)) - функция расстояния
            Получает на вход координаты, возвращает одно число - расстояние между объектами
            По умолчанию берётся функция евклидова расстояния библиотеки scipy
        ---
            params: (CLARANSConfig) - класс конфигурации, см описание класса CLARANSConfig
        """
        # search config:
        self._numlocal: int = params.numlocal
        self._maxneighbour: int = params.maxneighbour
        self._num_k: int = params.num_k
        self._d = distance_function
        # data preprocessing:
        self._y_col: str = params.y_col
        # best result:
        self._mincost: float = params.mincost
        self._best_node: dict = params.best_node
        # random generator:
        self._random_generator: Generator = Generator(PCG64(params.random_state).spawn(1)[0])
        # misc
        self._verbose = params.verbose
        self._debug = params.full_debug

    def train(self, data: pd.DataFrame):
        """Функция для тренировки на заданном массиве данных

        Args:
            data (pd.DataFrame): Датафрейм данных

        Returns:
            Словарь: Содержит лучшие найденные алгоритмом медойды
        """
        data = self._preprocessing(data)
        for _ in tqdm(range(self._numlocal)):
            if self._verbose or self._debug:
                print(f'--- cycle {_+1} ---')
            # select random node
            values = (data.sample(self._num_k, random_state=self._rs_next())
                      .reset_index()
                      .values)
            if self._debug:
                print('--- randomly generated nodes as np array ---')
                [print(row) for row in values]
            node = {int(obj[0]): obj[1:] for obj in values}
            descent_node = self._graph_descent(data, node)            
            descent_std = self._calc_std(data, descent_node)
            if self._debug:
                print(f'comparing {descent_std} to {self._mincost}')
            if descent_std < self._mincost:
                if self._verbose or self._debug:
                    print(f'! new std {descent_std:.3} is better !')
                    print('Changing best node to:')
                    [print(f'{key} - {value}') for key, value in descent_node.items()]
                self._mincost = descent_std
                self._best_node = descent_node
        return self._best_node
        
    # return original dataset
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Функция, совершающая кластеризацию передаваемых данных

        Args:
            data (pd.DataFrame): Входные данные

        Returns:
            pd.DataFrame: Размаркированные данные
        """
        data = self._preprocessing(data)
        return self._label_data(data, self._best_node)

    def get_solution(self) -> pd.DataFrame:
        """Функция, выводащая текущий лучший результат

        Returns:
            pd.DataFrame: Лучшие точки с их индексами
        """
        solution = pd.DataFrame(self._best_node.values())
        solution['index'] = self._best_node.keys()
        return solution
    
    def get_translation(self) -> tuple[float, dict]:
        """Функция, возвращающая данные для трансляции решения

        Returns:
            tuple[float, dict]: Кортеж значений, представляющих собой текущее решение
            в порядке: (цена, набор ночек)
        """
        return (self._mincost, self._best_node)
    
    # descending from given node
    def _graph_descent(self, data: pd.DataFrame, node: dict) -> dict:
        j = 0
        while j < self._maxneighbour:
            iop = self._rs_iop(data, node)
            iom = self._random_generator.choice(list(node.keys()))
            change_cost = self._calc_cost(data, iom, iop, node)
            if change_cost < 0:
                if self._debug:
                    print('- changing node in descent -')
                    print('was:')
                    [print(f'{key} - {value}') for key, value in node.items()]
                del node[iom]
                if self._debug:
                    print(f'chosen node: {data.loc[iop].values}')
                node[iop] = data.loc[iop].values
                j = 0
                if self._debug:
                    print('became:')
                    [print(f'{key} - {value}') for key, value in node.items()]
            else:
                j += 1
        if self._verbose or self._debug:
            print('- descent_finished -')
            if self._debug:
                print('with solution: ')
                [print(f'{key} - {value}') for key, value in node.items()]
        return node

    # calculating cost based on data and given iom and iop
    def _calc_cost(self, 
                  data: pd.DataFrame, 
                  iom: int,
                  iop: int,
                  node: dict) -> float:
        
        cost = 0
        om = node[iom]
        op = data.loc[iop].values
        # if self.verbose:
            # print('--- calculating cost for ---')
            # print(f'om: {om}')
            # print(f'op: {op}')
        for _, oj in data.iterrows():
            oj_m = self._d(oj, om)
            oj_p = self._d(oj, op)
            oj_2 = np.min([self._d(oj, coords) for _, coords in node.items() if _ != iom])
            # switchcase for poor
            # case 1 and 2
            if oj_m < oj_2:
                # belongs to m class
                # case 1
                if oj_p >= oj_2:
                    cost += oj_2 - oj_m
                # case 2
                elif oj_p < oj_2:
                    cost += oj_p - oj_m
            # case 3 and 4
            elif oj_m >= oj_2:
                # case 3
                if oj_p >= oj_2:
                    cost += 0
                # case 4
                elif oj_p < oj_2:
                    cost += oj_p - oj_2
        if self._debug:
            print(f'- resulting cost: {cost} -')
        return cost
    
    def _rs_next(self) -> float:
        return int(self._random_generator.random() * 100000)
    
    def _rs_iop(self, data: pd.DataFrame, node: dict) -> np.array:
        iop = self._random_generator.choice(data.index)
        while iop in node.keys():
            iop += 1
        return iop
        
    # preprocessed data only
    def _label_data(self, data: pd.DataFrame, node: dict) -> pd.DataFrame:
        labels = []
        if self._debug:
            print('- label data -')
            [print(f'{key} - {value}') for key, value in node.items()]
        for _, row in data.iterrows():
            labels.append(
                np.argmin([self._d(row.values, coords) for coords in node.values()])
            )
        marked_df = data.copy()
        marked_df['label'] = labels
        return marked_df
    
    # calculating disp for given data and solution node
    def _calc_std_old(self, data: pd.DataFrame, node: dict):
        marked_df = self._label_data(data, node)
        dispersions = []
        for label in marked_df['label'].unique():
            temp_df = marked_df[marked_df['label'] == label]
            dispersions.append(np.std(temp_df.values))
        return np.mean(dispersions)
    
    def _calc_std(self, data: pd.DataFrame, node: dict):
        distances = []
        for _, row in data.iterrows():
            distances.append(
                np.min([self._d(row.values, coords) for coords in node.values()])
            )
        return np.mean(distances)
                
    
    def _preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self._drop_y(data)
        data = data.reset_index(drop=True)
        return data

    def _drop_y(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.drop(columns=self._y_col) if self._y_col else data