
# Rosenbrock method

Реализация метода Розенброка.

## Параметры поиска
В структуре *Options* описываются параметры поиска:
1. `mHInit` – исходный массив значений шагов по каждому направлению.
2. `mInc` – исходное значение коэффициента увеличения шага.
3. `mDec` – исходное значение коэффициента уменьшения шага.
4. `mEps` – параметр для остановки алгоритма.
5. `maxUnsuccessStepsNumber` – максимальное число неудачных серий шагов по всем направлениям на одной итерации.
6. `maxStepsNumber` – ограничение на число итераций.

## Реализация алгоритма - метод *search*
### Инициализация переменных

`FT xOld[n]` – предыдущая рассматриваемая точка, инициализируем значением текущей точки.
```c++
FT xOld[n]; 
snowgoose::VecUtils::vecCopy(n, x, xOld);
```

`FT fcur` – значение функции в текущей точке (текущее приближение экстремума);
`FT fOld` – значение функции в предыдущей точке (предыдущее приближение экстремума);

`std::vector<FT> sft` – вектор значений шагов по каждому направлению.
```c++
std::vector<FT> sft(mOptions.mHInit, mOptions.mHInit + n);
```

`FT stepLenghts[n]` – массив длин шагов, пройденных по каждому направлению на текущей итерации. Используется при повороте базиса в процессе ортогонализации. 
```c++
snowgoose::VecUtils::vecSet(n, 0., stepLenghts);
```

`FT * dirs` – матрица, состоящая из базисных векторов – направлений поиска. Исходный базис совпадает с координатными направлениями. 
```c++
FT * dirs = new FT[n_sqr];
snowgoose::VecUtils::vecSet(n * n, 0., dirs);
for (int i = 0; i < n; i++) {
    dirs[i * n + i] = 1;
}
```

`FT * a, b, d` – матрицы, используемые в процессе ортогонализации. 
```c++
FT * a = new FT[n_sqr];
FT * b = new FT[n_sqr];
FT * d = new FT[n_sqr];
```

### Функция `step()`
Попытка продвинуться по каждому направлению с учетом текущей величины шага по данному направлению. Вернет true, если хотя бы один шаг по какому-либо из направлений 
оказался удачным. 

`bool isStepSuccessful` – возвращаемое значение. 
```c++
bool isStepSuccessful = false;
```

`FT xn[n]` – локальная копия текущей точки. 
```c++
snowgoose::VecUtils::vecCopy(n, x, xn);
```

`FT fn` – значение функции в xn. 
```c++
FT fn = fcur;
```
###### Проход по всем направлениям в цикле
`FT h` – величина шага по i-тому напрвлению. 
```c++
const FT h = sft[i];
```

`FT xtmp[n]` – точка при движении по i-тому направлению. 
```c++
FT xtmp[n]; 
snowgoose::VecUtils::vecSaxpy(n, xn, &(dirs[i * n]), h, xtmp);
```

`FT ftmp` – значение функции в xtmp. 
```c++
FT ftmp = obj->func(xtmp);
```

В случае удачи (`ftmp < fn`)
1. Увеличиваем пройденное расстояние по i-тому направлению
```c++
stepLenghts[i] += h;
```
2. Увеличиваем величину шага по i-тому направлению
```c++
sft[i] = inc(h);
```
3. Запоминаем точку
```c++
snowgoose::VecUtils::vecCopy(n, xtmp, xn);
fcur = ftmp;
```

В случае неудачи уменьшаем величину шага по i-тому направлению
```c++
sft[i] = dec(h);
```

### Функция `ortogonalize()`
Поворот базиса *dirs* с помощью процедуры Грама-Шмидта:

<a href="https://www.codecogs.com/eqnedit.php?latex=a&space;_i&space;=&space;\begin{cases}&space;&&space;d&space;_i&space;,\qquad&space;\lambda&space;_i=0&space;\\&space;&&space;\sum_{j=1}^n&space;\lambda&space;_i&space;d&space;_j&space;,\qquad&space;\lambda&space;_i&space;\neq&space;0&space;\end{cases}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?a&space;_i&space;=&space;\begin{cases}&space;&&space;d&space;_i&space;,\qquad&space;\lambda&space;_i=0&space;\\&space;&&space;\sum_{j=1}^n&space;\lambda&space;_i&space;d&space;_j&space;,\qquad&space;\lambda&space;_i&space;\neq&space;0&space;\end{cases}" title="a _i = \begin{cases} & d _i ,\qquad \lambda _i=0 \\ & \sum_{j=1}^n \lambda _i d _j ,\qquad \lambda _i \neq 0 \end{cases}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=b&space;_i&space;=&space;\begin{cases}&space;&&space;a&space;_i&space;,\qquad&space;i=1&space;\\&space;&&space;a&space;_i&space;-&space;\sum_{j=1}^{i-1}&space;(a&space;_i&space;^T&space;\bar{d&space;_j})&space;\bar{d&space;_j}&space;,\qquad&space;i&space;\geq&space;2&space;\end{cases}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?b&space;_i&space;=&space;\begin{cases}&space;&&space;a&space;_i&space;,\qquad&space;i=1&space;\\&space;&&space;a&space;_i&space;-&space;\sum_{j=1}^{i-1}&space;(a&space;_i&space;^T&space;\bar{d&space;_j})&space;\bar{d&space;_j}&space;,\qquad&space;i&space;\geq&space;2&space;\end{cases}" title="b _i = \begin{cases} & a _i ,\qquad i=1 \\ & a _i - \sum_{j=1}^{i-1} (a _i ^T \bar{d _j}) \bar{d _j} ,\qquad i \geq 2 \end{cases}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\qquad&space;\bar{d&space;_i}&space;=&space;\frac{b&space;_i}{\left&space;\|&space;b&space;_i&space;\right&space;\|}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\qquad&space;\bar{d&space;_i}&space;=&space;\frac{b&space;_i}{\left&space;\|&space;b&space;_i&space;\right&space;\|}" title="\qquad \bar{d _i} = \frac{b _i}{\left \| b _i \right \|}" /></a>

, где `dirs[i]` обозначено через *d<sub>i</sub>*.


### Основной цикл поиска
Делаем вызов функции `step()`
```c++
bool success = step();
```

Пересчитываем количество последовательных неудачных шагов
```c++
unsuccessSteps = success ? 0 : unsuccessSteps + 1;
```

#### Итерация неудачна (`success == false`): 
* Если после последнего поворота базиса текущее значение функции изменилось (`fcur < fOld`), либо число неудачных итераций превысило максимум (`unsuccessSteps >= mOptions.maxUnsuccessStepsNumber`), считаем расстояние между текущей и предыдущей точками
```c++
FT dist = snowgoose::VecUtils::vecDist(n, xOld, x); 
```
Если расстояние меньше границы (`dist < mOptions.mEps`), заканчиваем вычисления, иначе поворачиваем базис и ставим значениям шагов по каждому направлению исходное значение:
```c++
ortogonalize();
sft.assign(mOptions.mHInit, mOptions.mHInit + n);
snowgoose::VecUtils::vecSet(n, 0., stepLenghts);
```

* Если после последнего поворота базиса текущее значение функции не изменилось (`fcur == fOld`), и значения шагов по всем направлениям меньше &epsilon;, заканчиваем вычисления.
