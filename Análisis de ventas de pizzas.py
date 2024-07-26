"""
Descripción
Un conjunto de datos sintéticos que describe las ventas de pizza en una pizzería en algún lugar de EE. UU. Si bien el contenido es artificial, 
los ingredientes utilizados para hacer las pizzas distan mucho de serlo. Hay 32 pizzas diferentes que se dividen en 4 categorías diferentes: 
clásica (pizzas clásicas: '¡Probablemente comiste una como esta antes, pero nunca como ésta!'), pollo (pizzas con pollo como ingrediente 
principal: 'Prueba la Southwest Chicken Pizza'). ! ¡Te encantará!'), suprema (pizzas que se esfuerzan un poco más: '¡Mi pizza Soppressata usa 
solo el mejor salami de mi salumista personal!'), y vegetariana (pizzas sin carne alguna: 'Mis Cinco Quesos'). ¡La pizza tiene tantos quesos que 
sólo puedo ofrecerla en tamaño grande!').

variables
Un tibble con 49574 filas y 7 variables:

id: El ID del pedido, que consta de una o más pizzas en una fecha y hora determinadas.
fecha: una representación de caracteres de la fecha del pedido, expresada en el formato de fecha ISO 8601 (AAAA-MM-DD)
hora: una representación de caracteres de la hora del pedido, expresada como hora de 24 horas en el formato de hora extendido ISO 8601 (hh:mm:ss)
nombre: el nombre corto de la pizza
tamaño: El tamaño de la pizza, que puede ser S, M, L, XL (¡poco común!) o XXL (¡aún más raro!); la mayoría de las pizzas están disponibles en 
tamaños S, M y L, pero se aplican excepciones.
tipo: La categoría o tipo de pizza, que puede ser clásica, de pollo, suprema o vegetariana.
precio: El precio de la pizza y el precio por el que se vendió (en USD)

Detalles: cada pizza del conjunto de datos se identifica con un nombre corto. Los siguientes listados proporcionan los nombres completos de cada 
pizza y sus ingredientes principales.

Pizzas clásicas:
classic_dlx: La pizza clásica de lujo (pepperoni, champiñones, cebollas rojas, pimientos rojos y tocino)
big_meat: La gran pizza de carne (tocino, pepperoni, salchicha italiana, chorizo)
pepperoni: La pizza de pepperoni (queso mozzarella, pepperoni)
hawaiano: La pizza hawaiana (jamón en lonchas, piña, queso mozzarella)
pep_msh_pep: Pizza de pepperoni, champiñones y pimientos (pepperoni, champiñones y pimientos verdes)
ital_cpcllo: La pizza italiana Capocollo (Capocollo, pimientos rojos, tomates, queso de cabra, ajo, orégano)
napolitana: La Pizza Napolitana (Tomates, Anchoas, Aceitunas Verdes, Cebollas Moradas, Ajo)
the_greek: La pizza griega (aceitunas Kalamata, queso feta, tomates, ajo, carne asada, cebollas moradas)

Pizzas de Pollo:
thai_ckn: La pizza tailandesa de pollo (pollo, piña, tomates, pimientos rojos, salsa tailandesa de chile dulce)
bbq_ckn: Pizza de pollo a la barbacoa (pollo a la barbacoa, pimientos rojos, pimientos verdes, tomates, cebollas moradas, salsa barbacoa)
Southw_ckn: The Southwest Chicken Pizza (pollo, tomates, pimientos rojos, cebollas moradas, chiles jalapeños, maíz, cilantro, salsa chipotle)
cali_ckn: The California Chicken Pizza (pollo, alcachofa, espinacas, ajo, chiles jalapeños, queso fontina, queso gouda)
ckn_pesto: La pizza de pollo al pesto (pollo, tomates, pimientos rojos, espinacas, ajo, salsa pesto)
ckn_alfredo: La Pizza Alfredo de Pollo (Pollo, Cebolla Morada, Pimientos Rojos, Champiñones, Queso Asiago, Salsa Alfredo)

Pizzas Supremas:
brie_carre: La pizza Brie Carré (queso Brie Carré, jamón serrano, cebolla caramelizada, peras, tomillo, ajo)
calabrese: La pizza calabrese ('Nduja Salami, panceta, tomates, cebollas rojas, pimientos friggitello, ajo)
soppressata: La pizza soppressata (salami soppressata, queso fontina, queso mozzarella, champiñones, ajo)
siciliano: La pizza siciliana (salami siciliano grueso, tomates, aceitunas verdes, salchicha luganega, cebolla, ajo)
ital_supr: La pizza suprema italiana (salami calabrese, capocollo, tomates, cebollas moradas, aceitunas verdes, ajo)
peppr_salami: Pizza de salami a la pimienta (salami de Génova, capocollo, pepperoni, tomates, queso asiago, ajo)
prsc_argla: La pizza de prosciutto y rúcula (Prosciutto di San Daniele, rúcula, queso mozzarella)
spinach_supr: La Pizza Suprema de Espinacas (Espinacas, Cebolla Morada, Pepperoni, Tomates, Alcachofas, Aceitunas Kalamata, Ajo, Queso Asiago)
picante_ital: La pizza italiana picante (Capocollo, tomates, queso de cabra, alcachofas, peperoncini verdi, ajo)

Pizzas Vegetales
mexicana: La Pizza Mexicana (tomates, pimientos rojos, chiles jalapeños, cebollas moradas, cilantro, maíz, salsa chipotle, ajo)
four_cheese: La pizza de los cuatro quesos (queso ricotta, queso gorgonzola piccante, queso mozzarella, queso parmigiano reggiano, ajo)
five_cheese: La pizza de los cinco quesos (queso mozzarella, queso provolone, queso gouda ahumado, queso romano, queso azul, ajo)
spin_pesto: La pizza de espinacas al pesto (espinacas, alcachofas, tomates, tomates secos, ajo, salsa pesto)
veggie_veg: La pizza de verduras + verduras (champiñones, tomates, pimientos rojos, pimientos verdes, cebollas moradas, calabacines, espinacas, 
ajo)
green_garden: The Green Garden Pizza (espinacas, champiñones, tomates, aceitunas verdes, queso feta)
mediterraneo: La Pizza Mediterránea (Espinacas, Alcachofas, Aceitunas Kalamata, Tomates Secos, Queso Feta, Tomates Ciruela, Cebollas Rojas)
spinach_fet: La pizza de espinacas y queso feta (espinacas, champiñones, cebolla morada, queso feta, ajo)
ital_veggie: La pizza italiana de verduras (berenjenas, alcachofas, tomates, calabacines, pimientos rojos, ajo, salsa pesto)"""

import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
plt.style.use('dark_background')
from datetime import datetime
from IPython.core.display import display, HTML


#Explorando el conjunto de datos
df = pd.read_csv("A_year_of_pizza_sales_from_a_pizza_place_872_68.csv")
print(df.shape)

print(df.head())

#Caída de las tres primeras mesas por falta de necesidad
df.drop(df.columns[:3].tolist(), axis=1, inplace=True)

#Explorando datos NULL
df.isna().sum()

#Progresión de precios a lo largo del tiempo en 2015
fig = px.line(df, x="date", y="price", template = "plotly_dark")
fig.show()

#Pizzas más populares ordenadas
fig, axes = plt.subplots()
count = pd.DataFrame(df["name"].value_counts())
sns.barplot(x=count.index[:10], y=count.iloc[:10, 0], ax=axes, color = "fuchsia", edgecolor = "lightpink")
for container in axes.containers:
    axes.bar_label(container)
axes.set_xticklabels(axes.get_xticklabels(), rotation=45)
axes.set_yticklabels(())
axes.set_title("La pizza cuenta\n", fontsize = '16', fontweight = 'bold')
plt.show()

#Tipos de pizza más populares
fig, axes = plt.subplots(ncols=2, figsize=(15, 6))
for i, j in enumerate(["size", "type"]):
    count = df[j].value_counts()
    count.plot(kind="bar", ax=axes[i])
    for container in axes[i].containers:
        axes[i].bar_label(container)
    axes[i].set_yticklabels(())
    axes[i].set_xlabel("")
    axes[i].set_title(j.capitalize())

plt.tight_layout()
plt.show()

months = ["Enero", "Febrero", "Marzo",
         "Abril", "Mayo", "Junio",
         "Julio", "Agosto", "Septiembre",
         "Octubre", "Noviembre", "Diciembre"]

def convert_date(x):
    date = datetime.strptime(x, "%Y-%m-%d")
    return date.month

#Extraer meses de fechas
df["months"] = df["date"].apply(convert_date)
#Precio medio por pizza
grouped = df.groupby("months")
fig, axes = plt.subplots()
mean = pd.DataFrame(grouped["price"].mean())
dd = {"month": [], "value": []}
for i in range(mean.shape[0]):
    dd["month"] += [months[mean.index[i]-1]]
    dd["value"] += [mean.iloc[i, 0]]
dd = pd.DataFrame(dd)

sns.barplot(x=dd.iloc[:, 0], y=dd.iloc[:, 1], ax=axes, color = "cyan")
for container in axes.containers:
    axes.bar_label(container, label_type="center", rotation=90)
axes.set_xticklabels(axes.get_xticklabels(), rotation=0)
axes.set_yticklabels(())
axes.set_ylabel("")
axes.set_xlabel("")
axes.set_title("Meses\n")
plt.show()

#Top 10 de las pizzas más caras de media
grouped = df.groupby("name")
fig, axes = plt.subplots()
mean = pd.DataFrame(grouped["price"].mean())
mean = mean.sort_values("price", ascending=False)

sns.barplot(x=mean.index[:10], y=mean.iloc[:10, 0], ax=axes, color = "orange")
for container in axes.containers:
    axes.bar_label(container, label_type="center", rotation=90)
axes.set_xticklabels(axes.get_xticklabels(), rotation=45)
axes.set_yticklabels(())
axes.set_ylabel("")
axes.set_xlabel("")
axes.set_title("Nombres de pizza\n", fontsize = '16', fontweight = 'bold')
plt.show()

#Precios medios para cada tipo de pizza en general
fig, axes = plt.subplots(ncols=2, figsize=(15, 6))
for i, j in enumerate(["size", "type"]):
    grouped = df.groupby(j)
    mean = grouped["price"].mean()
    sns.barplot(x=mean.index, y=mean, ax=axes[i], color = "deeppink", edgecolor = "pink")
    for container in axes[i].containers:
        axes[i].bar_label(container)
    axes[i].set_yticklabels(())
    axes[i].set_xlabel("")
    axes[i].set_title(j.capitalize())
    axes[i].set_ylabel("")
    axes[i].set_xlabel("")
plt.tight_layout()
plt.show()

#Precios medios de diferentes tipos de pizza para cada mes en 2015
for l in sorted(df["months"].unique()):
    display(HTML("<h2>Precios medios por cada tipo de pizza en {} 2015</h2>".format(months[l-1])))
    temp_df = df[df["months"] == l]
    fig, axes = plt.subplots(ncols=2, figsize=(15, 6))
    for i, j in enumerate(["size", "type"]):
        grouped = temp_df.groupby(j)
        mean = grouped["price"].mean()
        sns.barplot(x=mean.index, y=mean, ax=axes[i])
        for container in axes[i].containers:
            axes[i].bar_label(container)
        axes[i].set_yticklabels(())
        axes[i].set_xlabel("")
        axes[i].set_title(j.capitalize())
        axes[i].set_ylabel("")
        axes[i].set_xlabel("")
    plt.tight_layout()
    plt.show()
    

# Cargar el conjunto de datos
file_path = 'A_year_of_pizza_sales_from_a_pizza_place_872_68.csv'
pizza_sales = pd.read_csv(file_path)

# Mostrar las primeras filas del conjunto de datos
print(pizza_sales.head())

# Generar estadísticas resumidas
print(pizza_sales.describe())

print(pizza_sales.info())

# Convertir la columna 'fecha' al formato de fecha y hora
pizza_sales['date'] = pd.to_datetime(pizza_sales['date'], format='%Y-%m-%d')

# Convertir la columna 'hora' al formato de fecha y hora
pizza_sales['time'] = pd.to_datetime(pizza_sales['time'], format='%H:%M:%S').dt.time

# Mostrar las primeras filas del conjunto de datos limpio
print(pizza_sales.head())

# Distribución de ventas por tamaño de pizza
size_distribution = pizza_sales['size'].value_counts()

# Graficado
plt.figure(figsize=(8, 6))
sns.barplot(x=size_distribution.index, y=size_distribution.values, palette="inferno")
plt.title('Distribución de ventas por tamaño de pizza\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Tamaño de la pizza\n')
plt.ylabel('Número de ventas\n')
plt.show()

# Ventas totales de cada tipo de pizza
total_sales_by_type = pizza_sales['type'].value_counts()

# Precio de venta promedio para cada tipo de pizza
average_sales_by_type = pizza_sales.groupby('type')['price'].mean()

# Trazar las ventas totales
plt.figure(figsize=(8, 6))
sns.barplot(x=total_sales_by_type.index, y=total_sales_by_type.values, palette="inferno")
plt.title('Ventas totales por tipo de pizza\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Tipo de pizza\n')
plt.ylabel('Número de ventas\n')
plt.show()

# Trazar el precio de venta promedio
plt.figure(figsize=(8, 6))
sns.barplot(x=average_sales_by_type.index, y=average_sales_by_type.values, palette="inferno")
plt.title('Precio de venta promedio por tipo de pizza\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Tipo de pizza\n')
plt.ylabel('Precio Promedio (USD)\n')
plt.show()

# Ventas e ingresos totales por nombre de pizza
total_sales_by_pizza = pizza_sales['name'].value_counts().sort_values(ascending=False).head(10)
total_revenue_by_pizza = pizza_sales.groupby('name')['price'].sum().sort_values(ascending=False).head(10)

# Trazar las ventas totales
plt.figure(figsize=(12, 6))
sns.barplot(x=total_sales_by_pizza.values, y=total_sales_by_pizza.index, palette="inferno")
plt.title('Las 10 mejores pizzas por ventas totales\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Número de ventas\n')
plt.ylabel('Nombre de la pizza\n')
plt.show()

# Trazar los ingresos totales
plt.figure(figsize=(12, 6))
sns.barplot(x=total_revenue_by_pizza.values, y=total_revenue_by_pizza.index, palette="inferno")
plt.title('Las 10 mejores pizzas por ingresos totales\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Ingresos totales (USD)\n')
plt.ylabel('Nombre de la pizza\n')
plt.show()

# Ingresos generados por cada tipo de pizza
revenue_by_type = pizza_sales.groupby('type')['price'].sum()

# Graficado
plt.figure(figsize=(8, 6))
sns.barplot(x=revenue_by_type.index, y=revenue_by_type.values, palette="inferno")
plt.title('Ingresos generados por cada tipo de pizza\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Tipo de pizza\n')
plt.ylabel('Ingresos totales (USD)\n')
plt.show()

# Tendencia de ventas diaria
daily_sales = pizza_sales.groupby('date').size()

# Tendencia de ventas semanal
pizza_sales['week'] = pizza_sales['date'].dt.isocalendar().week
weekly_sales = pizza_sales.groupby('week').size()

# Tendencia de ventas mensual
pizza_sales['month'] = pizza_sales['date'].dt.month
monthly_sales = pizza_sales.groupby('month').size()

# Trazar la tendencia de ventas diaria
plt.figure(figsize=(12, 6))
daily_sales.plot()
plt.title('Tendencia de ventas diarias\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Fecha\n')
plt.ylabel('Número de ventas\n')
plt.show()

# Trazar la tendencia de ventas semanal
plt.figure(figsize=(12, 6))
weekly_sales.plot()
plt.title('Tendencia de ventas semanales\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Semana\n')
plt.ylabel('Número de ventas\n')
plt.show()

# Trazar la tendencia de ventas mensual
plt.figure(figsize=(12, 6))
monthly_sales.plot()
plt.title('Tendencia de ventas mensuales\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Mes\n')
plt.ylabel('Número de ventas\n')
plt.show()

# Hora pico del día para pedidos de pizza
pizza_sales['hour'] = pizza_sales['time'].apply(lambda x: x.hour)
peak_time_of_day = pizza_sales['hour'].value_counts().sort_index()

# Graficado
plt.figure(figsize=(12, 6))
sns.lineplot(x=peak_time_of_day.index, y=peak_time_of_day.values, marker='o', palette="blue")
plt.title('Hora pico del día para pedidos de pizza\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Hora del día\n')
plt.ylabel('Número de ventas\n')
plt.xticks(range(24))
plt.grid(True)
plt.show()

# Tendencias de ventas por día de la semana
pizza_sales['day_of_week'] = pizza_sales['date'].dt.dayofweek
sales_by_day_of_week = pizza_sales['day_of_week'].value_counts().sort_index()

# Graficado
plt.figure(figsize=(8, 6))
sns.barplot(x=sales_by_day_of_week.index, y=sales_by_day_of_week.values, palette="inferno")
plt.title('Tendencias de ventas por día de la semana\n')
plt.xlabel('Día de la semana\n')
plt.ylabel('Número de ventas\n')
plt.xticks(ticks=sales_by_day_of_week.index, labels=['Lun', 'Mar', 'Mie', 'Jue', 'Vie', 'Sab', 'Dom'])
plt.show()

# Tamaño medio del pedido (número de pizzas por pedido)
average_order_size = pizza_sales.groupby('id').size().mean()
print(f'Tamaño promedio del pedido: {average_order_size:.2f}')

# Distribución de tamaños de pedidos
order_size_distribution = pizza_sales.groupby('id').size().value_counts().sort_index()

# Graficado
plt.figure(figsize=(12, 6))
sns.barplot(x=order_size_distribution.index, y=order_size_distribution.values, palette="inferno")
plt.title('Distribución de tamaños de pedidos\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Número de pizzas por pedido\n')
plt.ylabel('Número de ordenes\n')
plt.show()

# Patrones en pizzas ordenadas juntas
#order_combinations = pizza_sales.groupby('id')['name'].apply(lambda x: list(combinations(x, 2)))
#flattened_combinations = [item for sublist in order_combinations for item in sublist]
#combination_counts = Counter(flattened_combinations)

# Combinaciones más comunes
#most_common_combinations = combination_counts.most_common(10)
#print(most_common_combinations)

# Pizzas más y menos rentables
profit_by_pizza = pizza_sales.groupby('name')['price'].sum().sort_values(ascending=False)

# Trazando la pizza más rentable
plt.figure(figsize=(8, 6))
sns.barplot(x=profit_by_pizza.head(10).values, y=profit_by_pizza.head(10).index, palette="inferno")
plt.title('Las 10 pizzas más rentables\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Ingresos totales (USD)\n')
plt.ylabel('Nombre de la pizza\n')
plt.show()

# Trazando la pizza menos rentable
plt.figure(figsize=(8, 6))
sns.barplot(x=profit_by_pizza.tail(10).values, y=profit_by_pizza.tail(10).index, palette="inferno")
plt.title('Las 10 pizzas menos rentables\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Ingresos totales (USD)\n')
plt.ylabel('Nombre de la pizza\n')
plt.show()

"""Informe de ventas de pizza: una porción de información
Resumen ejecutivo: este informe profundiza en el dinámico mundo de las ventas de pizza, aprovechando un conjunto de datos completo que captura 
la esencia de los patrones de consumo de pizza en varios tipos, tamaños y sabores. A través de análisis y visualización exhaustivos, descubrimos 
tendencias intrigantes, arrojamos luz sobre las preferencias de los consumidores y brindamos información útil para los vendedores de pizzas que 
buscan optimizar sus ofertas y mejorar la satisfacción del cliente.

Introducción: La pizza, una delicia culinaria apreciada en todo el mundo, es la pieza central de innumerables reuniones, celebraciones y 
experiencias gastronómicas informales. Comprender los matices de las ventas de pizza es fundamental para los restauranteros que buscan prosperar 
en un panorama de mercado cada vez más competitivo. Nuestro informe tiene como objetivo desentrañar los misterios que rodean los hábitos de 
consumo de pizza, a partir de un rico conjunto de datos que abarca diversos tipos, tamaños y métricas de ventas de pizza.

Análisis de datos exploratorios: el conjunto de datos consta de 49.574 entradas, cada una de las cuales representa un pedido de pizza único. 
Comenzamos examinando la distribución de los tipos de pizza, revelando una cautivadora variedad de opciones clásicas, de pollo, supremas y 
vegetarianas. En particular, las pizzas clásicas emergen como la opción más popular entre los consumidores, y su atractivo atemporal resuena en 
diversos paladares.

Además, exploramos la relación entre el tamaño de la pizza y el volumen de ventas, descubriendo una inclinación por las pizzas de tamaño mediano 
entre los clientes. A pesar de la rareza de las pizzas XL y XXL, su presencia subraya el deseo ocasional de indulgencia entre los consumidores.

Información obtenida de las visualizaciones: nuestras visualizaciones ofrecen un placer para los ojos y la mente, proporcionando una comprensión 
matizada de la dinámica de las ventas de pizza. Desde gráficos de barras que muestran las ventas totales por tipo de pizza hasta histogramas que 
representan la distribución de los precios de las pizzas, cada visualización descubre patrones y tendencias ocultos.

De particular interés es la distribución de ventas por tipo y tamaño de pizza, que revela patrones de consumo interesantes en diferentes segmentos
demográficos. Además, la tabla de las 10 pizzas más vendidas proporciona información valiosa sobre las preferencias de los consumidores, guiando 
a los proveedores a crear ofertas irresistibles adaptadas a su público objetivo.

Resultados clave:

Las pizzas clásicas dominan el mercado, enfatizando la perdurable popularidad de los sabores tradicionales.
Las pizzas de tamaño mediano son las reinas, logrando un equilibrio entre el tamaño de las porciones y la relación calidad-precio.
La sensibilidad al precio varía según el tipo de pizza, y las pizzas supremas tienen precios superiores debido a sus ingredientes gourmet.
El análisis de series de tiempo destaca las fluctuaciones en el volumen de ventas a lo largo del tiempo, lo que señala la importancia de las 
variaciones estacionales y las estrategias promocionales para impulsar la demanda.
Recomendaciones: Con estos conocimientos, los vendedores de pizzas pueden tomar medidas proactivas para mejorar la oferta de su menú y 
mejorar la experiencia del cliente. Las recomendaciones clave incluyen:

Diversificar el menú para atender las cambiantes preferencias de los consumidores y las restricciones dietéticas.
Implementar campañas de marketing específicas para promover pizzas especiales y capitalizar las tendencias emergentes.
Aprovechar las estrategias de precios basadas en datos para maximizar la rentabilidad y al mismo tiempo garantizar la asequibilidad para 
los clientes.
Conclusión: En conclusión, el conjunto de datos sobre ventas de pizza ofrece un tesoro de conocimientos para los proveedores que se 
esfuerzan por prosperar en el competitivo panorama culinario. Al aprovechar el poder del análisis y la visualización de datos, los dueños 
de restaurantes pueden desbloquear nuevas oportunidades de crecimiento, innovación y participación del cliente. A medida que el apetito 
por la pizza continúa aumentando, aquellos que adoptan la toma de decisiones basada en datos están preparados para lograr una porción del 
éxito en el dinámico mundo de las ventas de pizza."""


plt.figure(figsize=(10, 6))
sns.countplot(x='type', data=df, color = "fuchsia")
plt.title('Ventas totales por tipo de pizza\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Tipo de pizza\n')
plt.ylabel('Ventas totales\n')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='size', data=df, color = "mediumspringgreen")
plt.title('Ventas totales por tamaño de pizza\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Tamaño de la pizza\n')
plt.ylabel('Ventas totales\n')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=30, kde=True, color = "thistle", edgecolor = "violet")
plt.title('Distribución de precios de pizza\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Precio (dólares americanos)\n')
plt.ylabel('Frecuencia\n')
plt.show()


plt.figure(figsize=(10, 6))
sns.barplot(x='type', y='price', data=df, estimator=np.mean, color = "aqua")
plt.title('Precio Promedio de Pizzas por Tipo\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Tipo de pizza\n')
plt.ylabel('Precio Promedio (USD)\n')
plt.show()

df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
df['date'] = pd.to_datetime(df['date'])

plt.figure(figsize=(12, 6))
df.groupby('date')['id'].count().plot()
plt.title('Ventas totales a lo largo del tiempo\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Fecha\n')
plt.ylabel('Ventas totales\n')
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='type', hue='size', data=df, palette = "inferno")
plt.title('Distribución de ventas por tipo y tamaño de pizza\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Tipo de pizza\n')
plt.ylabel('Ventas totales\n')
plt.legend(title='Size')
plt.show()

top_pizzas = df['name'].value_counts().nlargest(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_pizzas.index, y=top_pizzas.values, color = "navy")
plt.title('Las 10 pizzas más vendidas\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Nombre de la pizza\n')
plt.ylabel('Ventas totales\n')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='type', y='price', data=df)
plt.title('Distribución de precios de Pizzas por tipo\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Tipo de pizza\n')
plt.ylabel('Precio (dólares americanos)\n')
plt.show()

