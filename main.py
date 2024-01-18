import wave
import numpy as np

# Функция активации для внутреннего слоя
def relu(x):
    return (x > 0) * x

# Производная функции активации для обратного распространения ошибки 
def relu2deriv(output):
    return output>0

# шаг корекции весов
alpha = 0.2
# размер внутреннего слоя
hidden_size = 10

# создание матриц весов со значением от 0 до 1
weights_0_1 = 2*np.random.random((1, hidden_size)) - 1
weights_1_2 = 2*np.random.random((hidden_size, 1)) - 1

# алгоритм обучения 
for itaration in range(60):

    # переменная выводимой ошибки 
    layer_2_error = 0

    # пропуск датасета через нейросеть
    for i in range (1, 8):

        # открытие файлов из датасета
        with wave.open(f'dataset/{str(i)}.wav', 'rb') as file:

            # ожидаемое значение на выходе 
            true = i

            # входные данные
            input = np.array([[file.getnframes()]])
            
            # прямое распространение 
            layer_0 = input
            layer_1 = relu(np.dot(layer_0, weights_0_1))
            layer_2 = np.dot(layer_1, weights_1_2)

            # подсчет выводимой ошибки / возвод в квадрат, чтобы увеличить значение > 1 и уменьшить значение < 1
            layer_2_error += np.sum((layer_2 - true)**2)

            # вичисление применяемой ошибки для весов между 2 и 3 слоем и 1 и 2 / применение производной функции активации для последних
            layer_2_delta = (true - layer_2)
            layer_1_delta = layer_2_delta.dot(weights_1_2.T)*relu2deriv(layer_1)

            # корекция весов сети на значение ошибки для конкретного нейрона * шаг корекции 
            weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
            weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)
    
    # вывод ошибки в консоль
    if (itaration % 10 == 9):
        print(f'Error: {layer_2_error}')


