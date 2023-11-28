from fastapi import FastAPI
import pandas as pd
import uvicorn

app = FastAPI()

@app.get("/")
def index():
    return {"message": "Hello, world!"}

@app.get("/PlayTimeGenre/{genero}")
async def PlayTimeGenre(genero: str):
    """
    Recibe un género (str) y devuelve el año de lanzamiento con más horas jugadas
    * Un ejemplo de respuesta a esta solicitud sería:

        ```
        {
            "Año de lanzamiento con más horas jugadas para Acción": 2000
        }
    """
    # Leemos el archivo parquet
    df_genero = pd.read_parquet("Data/endpoint_1")

    # Filtrar el DataFrame por el género especificado
    df_genero = df_genero[df_genero["genres"] == genero]

    # Encontrar el año con más horas jugadas para el género
    año_con_mas_horas = list(df_genero[df_genero["playtime"] == df_genero["playtime"].max()]["release_year"])[0]

    return {f"Año de lanzamiento con más horas jugadas para {genero}": año_con_mas_horas}


@app.get("/UserForGenre/{genero}")
async def UserForGenre(genero: str):
    """
    Recibe un género y devuelve el usuario con más horas jugadas, junto al acumulado de horas por año.
        ```
    * La respuesta a esta solicitud sería:

        ```
        {
            "Usuario con más horas jugadas para Acción": 1234567890,
            "Horas jugadas": [
                {
                    "Año": 2000,
                    "Horas": 100
                },
                {
                    "Año": 2001,
                    "Horas": 200
                },
                ...
            ]
        }
        
    """
    # Leer el archivo Parquet
    df_endpoint_2 = pd.read_parquet("Data/endpoint_2")

    # Convertir la columna 'playtime' de minutos a horas
    df_endpoint_2['playtime'] = round(df_endpoint_2['playtime'] / 60, 2)

    # Filtrar el DataFrame por el género especificado
    df_genero_especifico = df_endpoint_2[df_endpoint_2['genres'] == genero]

    # Encontrar el usuario con más horas jugadas para el género dado
    usuario_con_mas_horas = df_genero_especifico.loc[df_genero_especifico['playtime'].idxmax()]['user_id']

    # Agrupar por año y sumar las horas jugadas para el usuario con más horas
    horas_por_año_usuario = df_genero_especifico[df_genero_especifico['user_id'] == usuario_con_mas_horas]
    horas_por_año_usuario = horas_por_año_usuario.groupby('release_year')['playtime'].sum().reset_index()
    horas_por_año_usuario = horas_por_año_usuario.rename(columns={'release_year': 'Año', 'playtime': 'Horas'})

    # Crear la lista de acumulación de horas jugadas por año
    lista_horas_por_año = horas_por_año_usuario.to_dict(orient='records')

    return {
        f"Usuario con más horas jugadas para {genero}": usuario_con_mas_horas,
        "Horas jugadas": lista_horas_por_año
    }


@app.get("/UsersRecommend/{year}")
async def UsersRecommend(year: int):
    """
    Recibe un año y devuelve los 3 juegos más recomendados por los usuarios australianos en el año especificado.

        La respuesta a esta solicitud sería:

            ```
            [
                {
                    "Puesto 1": "Elden Ring"
                },
                {
                    "Puesto 2": "God of War Ragnarök"
                },
                {
                    "Puesto 3": "Starfield"
                }
            ]
            
    """
    # Leemos el archivo de consulta.
    df = pd.read_parquet("Data/df_australian_user_reviews")

    # Filtramos las reseñas del año especificado.
    df_year = df[df["posted_year"] == year]

    # Filtramos las reseñas recomendadas y con comentarios positivos/neutrales.
    df_recommend = df_year[df_year["recommend"] == True]
    df_sentiment = df_recommend[df_recommend["sentiment_analysis"].isin([2, 1])]

    # Filtramos los juegos titulados "No especificado".
    df_sentiment = df_sentiment[df_sentiment["title"] != "No especificado"]

    # Convertimos los valores de la columna "recommend" a tipo int.
    df_sentiment["recommend"] = df_sentiment["recommend"].astype(int)

    # Agrupamos las reseñas por título y contamos el número de recomendaciones.
    recommendations = df_sentiment.groupby("title")["recommend"].sum()

    # Ordenamos las recomendaciones por número de recomendaciones.
    recommendations = recommendations.sort_values(ascending=False)

    # Obtenemos los nombres de los juegos para los top 3
    top_3_games = recommendations.head(3).index.tolist()

    if len(top_3_games) >= 3:
        return [{"Puesto 1": top_3_games[0]}, {"Puesto 2": top_3_games[1]}, {"Puesto 3": top_3_games[2]}]
    else:
        return "No hay suficientes datos para generar el top 3"

@app.get("/UsersWorstDeveloper/{year}")
async def UsersWorstDeveloper(year: int):
    """
    Recibe un año y devuelve las 3 desarrolladoras con más juegos con reseñas negativas en el año especificado.

        La respuesta a esta solicitud sería:
            ```
            [
                {
                    "Puesto 1": "Ubisoft"
                },
                {
                    "Puesto 2": "EA"
                },
                {
                    "Puesto 3": "Activision Blizzard"
                }
            ]
            
    """
    # Leemos el archivo de consulta
    df = pd.read_parquet("Data/df_australian_user_reviews")

    # Filtramos las reseñas del año especificado.
    df_year = df[df["posted_year"] == year]

    # Filtramos las reseñas NO recomendadas y con comentarios negativos.
    df_not_recommend = df_year[df_year["recommend"] == False]
    df_sentiment = df_not_recommend[df_not_recommend["sentiment_analysis"] == 0]

    # Filtramos los juegos titulados "No especificado".
    df_sentiment = df_sentiment[df_sentiment["developer"] != "Otro"]

    # Agrupamos las reseñas por desarrolladora y contamos el número de recomendaciones.
    recommendations = df_sentiment.groupby("developer")["recommend"].sum()

    # Ordenamos las recomendaciones por número de recomendaciones.
    recommendations = recommendations.sort_values(ascending=True)

    top_3_devs = recommendations.head(3).index.tolist()

    if len(top_3_devs) >= 3:
        return [{"Puesto 1": top_3_devs[0]}, {"Puesto 2": top_3_devs[1]}, {"Puesto 3": top_3_devs[2]}]
    else:
        return "No hay suficientes datos para generar el top 3"

@app.get("/sentiment_analysis/{empresa_desarrolladora}")
async def sentiment_analysis(empresa_desarrolladora: str):
    """
    Recibe un desarrollador y devuelve un resumen del análisis de sentimiento de los juegos de una desarrolladora específica.

        La respuesta a esta solicitud sería:
            ```
            {
                "Ubisoft": [
                    "Negative = 100",
                    "Neutral = 200",
                    "Positive = 300"
                ]
            }
            
    """
    # Cargar el DataFrame desde la ruta del archivo.
    df = pd.read_parquet("Data/df_australian_user_reviews")

    # Filtramos los registros con el developer especificado.
    df_developer = df[df["developer"] == empresa_desarrolladora]

    # Contar la cantidad de veces que aparecen los valores específicos para cada etiqueta.
    negative_count = (df_developer["sentiment_analysis"] == 0).sum()
    neutral_count = (df_developer["sentiment_analysis"] == 1).sum()
    positive_count = (df_developer["sentiment_analysis"] == 2).sum()

    # Crear el diccionario con el formato requerido.
    result_dict = {empresa_desarrolladora: [f"Negative = {negative_count}", f"Neutral = {neutral_count}", f"Positive = {positive_count}"]}

    return result_dict

@app.get("/recomendacion_juego/{id_producto}")
async def recomendacion_juego(id_producto: int):
    """
    Recibe un id de juego y devuelve las 5 recomendaciones más similares a un juego específico.

        La respuesta a esta solicitud sería:
            ```
            [
                "Recomendación 1: Elden Ring",
                "Recomendación 2: God of War Ragnarök",
                "Recomendación 3: Starfield",
                "Recomendación 4: Horizon Forbidden West",
                "Recomendación 5: Forza Horizon 5"
            ]
            
    """
    # Lectura de los archivos necesarios
    df_similitud_del_coseno = pd.read_parquet("Data/similitud_del_coseno")
    indice = pd.read_csv("Data/indices_modelo")
    filtrado = pd.read_parquet("Data/filtrado_modelo")

    if id_producto not in indice['item_id'].values:
        return f"El ID de producto {id_producto} no está en el archivo de índices."

    indc = indice.loc[indice['item_id'] == id_producto].index[0]
    puntajes_similares = list(enumerate(df_similitud_del_coseno[indc]))
    puntajes_similares = sorted(puntajes_similares, key=lambda x: x[1], reverse=True)
    puntajes_similares = puntajes_similares[1:6]
    juegos_indices = [int(i[0]) for i in puntajes_similares]

    recomendaciones = [
        f"Recomendación {i+1}: {filtrado['item_name'].iloc[juegos_indices[i]]}" 
        for i in range(len(juegos_indices))
    ]

    return recomendaciones
