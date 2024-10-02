import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Cargar los archivos CSV
data_gls= pd.read_csv(r'C:\Users\ADMIN\Desktop\labmeta\GitKraken\Metaheuristica-TSP\Results\Small(Parametrization)\tsp_Small_GLS_results.csv')
data_11 = pd.read_csv(r'C:\Users\ADMIN\Desktop\labmeta\GitKraken\Metaheuristica-TSP\Results\Small(Parametrization)\tsp_Small_results_11.csv')
data_31 = pd.read_csv(r'C:\Users\ADMIN\Desktop\labmeta\GitKraken\Metaheuristica-TSP\Results\Small(Parametrization)\tsp_Small_results_31.csv')
data_51 = pd.read_csv(r'C:\Users\ADMIN\Desktop\labmeta\GitKraken\Metaheuristica-TSP\Results\Small(Parametrization)\tsp_Small_results_51.csv')

# Añadir una columna a cada dataset para indicar su origen (para identificar después en los gráficos)
data_gls['Dataset'] = 'GLS'
data_11['Dataset'] = 'TS_11'
data_31['Dataset'] = 'TS_31'
data_51['Dataset'] = 'TS_51'

# Combinar los datasets en un solo DataFrame
data_combined = pd.concat([data_gls, data_11, data_31, data_51])


# Comparar varias variables
variables = ['ObjectiveFunction']

for var in variables:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Dataset', y=var, data=data_combined)
    plt.title(f'Comparación de {var} entre GLS y TS_51')
    plt.xlabel('Dataset')
    plt.ylabel(var)
    plt.show()

# Calcular las medias de las variables de interés
mean_values = data_combined.drop(columns=['Results']).groupby('Dataset').mean()[['ObjectiveFunction']]

# Crear gráfico de barras
mean_values.plot(kind='bar', figsize=(10, 6))
plt.title('Comparación de medias de ObjectiveFunction, Optimal y Calls entre GLS y TS con 11, 31 y 51 iteraciones')
plt.ylabel('Valor medio')
plt.show()