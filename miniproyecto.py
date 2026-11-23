"""Miniproyecto 2: aprendizaje automático y profundo contra el mejor modelo clásico.

Se corre de principio a fin con

    python miniproyecto.py

y deja las figuras en figuras/ y las tablas en resultados/. Las partes
marcadas con TODO las completa el grupo; mientras falten, el guion las
reporta como pendientes y sigue. El informe, docs/informe.md, cita las cifras
y las figuras que produce este guion.
"""
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox  # noqa: E402
from statsmodels.tsa.arima.model import ARIMA  # noqa: E402
from statsmodels.tsa.seasonal import STL  # noqa: E402
from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: E402
from torch import nn  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from comun import (RAIZ, backtest, cargar_serie, estacional_ingenuo,  # noqa: E402
                   fechas_siguientes, rmse)

# Después de importar statsmodels, que al cargarse fija sus propios avisos.
warnings.filterwarnings("ignore")
torch.set_num_threads(1)
FIGURAS = RAIZ / "figuras"
RESULTADOS = RAIZ / "resultados"
FIGURAS.mkdir(exist_ok=True)
RESULTADOS.mkdir(exist_ok=True)

SEMILLA = 2026
N_PRUEBA = 24          # los meses del backtest, los mismos de comun.backtest

# El último modelo que ajustó cada función modelo(entrenamiento, h). Sirve
# para contar sus parámetros después del backtest.
ULTIMO = {}

# ============================================================
# 1. La serie y el modelo a superar
# ============================================================
# Los mismos datos del miniproyecto 1; su ficha está en datos/README.md.
ARCHIVO = "datos/comercio_minorista_colombia.csv"
FECHA = "observation_date"
VALOR = "COLSLRTTO01IXOBM"
PERIODO = 12

# Exógena del mejor modelo clásico: número de festivos de Colombia por mes.
# Se precalculó con el paquete holidays (como en el miniproyecto 1) para que
# el guion solo dependa de las librerías permitidas. Cubre la serie y los 12
# meses que siguen a su final.
ARCHIVO_FESTIVOS = "datos/festivos_colombia.csv"

# Órdenes elegidos por AIC en el miniproyecto 1.
ORDEN = (1, 1, 1)
ORDEN_ESTACIONAL = (1, 1, 1, PERIODO)


def preparar(serie):
    """La misma preparación del miniproyecto 1.

    La fuente trae los 135 meses completos, sin vacíos: la interpolación
    lineal queda solo como respaldo y no toca valores reales. Los máximos de
    diciembre son el pico estacional real del comercio minorista, no errores,
    así que se conservan.
    """
    return serie.interpolate(method="linear")


def exogenas():
    """Festivos por mes, conocidos de antemano, con índice de inicio de mes."""
    tabla = pd.read_csv(RAIZ / ARCHIVO_FESTIVOS, parse_dates=["mes"], index_col="mes")
    festivos = tabla["festivos"].astype(float).asfreq("MS")
    return festivos.to_frame()


def modelo_clasico(serie):
    """El mejor modelo clásico del miniproyecto 1: SARIMAX(1,1,1)(1,1,1,12)
    sobre el nivel de la serie, con los festivos del mes como exógena.

    Ajusta con x.loc[entrenamiento.index] y pronostica con los festivos de los
    h meses siguientes, que se conocen de antemano, así que no hay fuga.
    """
    x = exogenas()
    faltan = serie.index.difference(x.index)
    assert faltan.empty, f"festivos sin dato para {list(faltan)}"

    def modelo(entrenamiento, h):
        futuro = fechas_siguientes(entrenamiento, h)
        ajuste = ARIMA(entrenamiento, order=ORDEN, seasonal_order=ORDEN_ESTACIONAL,
                       exog=x.loc[entrenamiento.index]).fit()
        ULTIMO["modelo"] = ajuste
        return ajuste.forecast(h, exog=x.loc[futuro]).to_numpy()
    return modelo


NOMBRE_CLASICO = f"SARIMAX{ORDEN}{ORDEN_ESTACIONAL} + festivos"

# Meses que el residuo de la STL marca como choque: la caída por la COVID y el
# paro nacional de mayo de 2021. Se usan solo en el experimento de abajo; la
# serie que entra a los modelos los conserva.
BLOQUE_ATIPICO = list(pd.date_range("2020-03-01", "2020-09-01", freq="MS")) + [
    pd.Timestamp("2021-05-01")]


def atipicos_stl(serie, umbral=3.5):
    """Atípicos en el residuo de la STL robusta, con z robusto (mediana y MAD)."""
    residuo = STL(serie, period=PERIODO, robust=True).fit().resid
    mad = (residuo - residuo.median()).abs().median()
    z = 0.6745 * (residuo - residuo.median()) / mad
    marcados = z[z.abs() > umbral]
    print(f"  Atípicos en el residuo de la STL (|z robusto| > {umbral}, MAD {mad:.2f}): "
          f"{len(marcados)} meses")
    for fecha, valor in marcados.items():
        print(f"    {fecha:%Y-%m}: z = {valor:.1f}")
    return marcados


def ljung_box(serie, x):
    """Valores p del Ljung-Box (12 y 24 rezagos) del SARIMAX ajustado a toda la
    serie; se descartan los 13 primeros residuos, que dejan las diferencias."""
    ajuste = ARIMA(serie, order=ORDEN, seasonal_order=ORDEN_ESTACIONAL,
                   exog=x.loc[serie.index]).fit()
    return acorr_ljungbox(ajuste.resid.iloc[1 + PERIODO:], lags=[12, 24])["lb_pvalue"].tolist()


def experimento_imputacion(serie):
    """¿Corregir el Ljung-Box imputando el bloque atípico mejora el pronóstico?

    Compara la serie conservada con dos imputaciones del bloque por tendencia
    más estacional de la STL: una con la STL ajustada a toda la serie (ve los
    meses del backtest, hay fuga) y otra ajustada solo antes del backtest.
    """
    x = exogenas()
    versiones = {"conservar": serie}
    for nombre, base in [("imputar, STL de toda la serie (con fuga)", serie),
                         ("imputar, STL antes del backtest", serie.iloc[:-N_PRUEBA])]:
        stl = STL(base, period=PERIODO, robust=True).fit()
        imputada = serie.copy()
        imputada.loc[BLOQUE_ATIPICO] = (stl.trend + stl.seasonal).loc[BLOQUE_ATIPICO]
        versiones[nombre] = imputada
    print("  Experimento: conservar o imputar el bloque atípico")
    for nombre, version in versiones.items():
        _, clasico = backtest(version, modelo_clasico(version))
        _, base = backtest(version, lambda t, h: estacional_ingenuo(t, h, PERIODO))
        p12, p24 = ljung_box(version, x)
        print(f"    {nombre}: SARIMAX RMSE {clasico['RMSE']:.2f}, estacional ingenuo "
              f"RMSE {base['RMSE']:.2f}, Ljung-Box p(12) {p12:.3f}, p(24) {p24:.3f}")


def graficar_serie(serie):
    figura, eje = plt.subplots(figsize=(9, 3.4))
    eje.plot(serie.index, serie.to_numpy())
    eje.axvline(serie.index[-N_PRUEBA], color="gray", linestyle="--")
    eje.set_title(f"{VALOR}: serie mensual; a la derecha de la línea, los meses del backtest")
    eje.set_ylabel(VALOR)
    figura.tight_layout()
    figura.savefig(FIGURAS / "serie.png", dpi=120)
    plt.close(figura)


# ============================================================
# 2. La tabla supervisada
# ============================================================
# Lo que aprende cada modelo: "nivel" es la serie tal cual; "dif12" es la
# diferencia con el mismo mes del año anterior.
# Se aprende el nivel: con "dif12" cada fila predice el cambio contra el mismo
# mes del año anterior y el modelo tiene que reconstruir el nivel sumando
# y(t-12), que arrastra el error del año entero. Con la rejilla elegida para
# cada versión, el nivel gana en el backtest: XGBoost 11.93 contra 20.92 y
# Random Forest 17.18 contra 18.86 de RMSE. La comparación está en la sección 2.
OBJETIVO = "nivel"
REZAGOS = 12


def objetivo(y):
    return y.diff(PERIODO) if OBJETIVO == "dif12" else y


def a_nivel(y, z):
    """Del pronóstico del objetivo al nivel: y(t) = y(t-12) + z(t) si z es la diferencia."""
    return y.iloc[-PERIODO] + z if OBJETIVO == "dif12" else z


ESTADISTICAS = (3, PERIODO)


def tabla_supervisada(z):
    """Una fila por mes con los rezagos 1 a REZAGOS, el mes del año y
    estadísticas de ventana móvil.

    Ninguna columna de la fila t usa z(t): los rezagos entran con shift(k) y
    las ventanas se calculan sobre la serie ya corrida un mes, así que la fila
    t solo ve z(t-1) hacia atrás. El mes del año va en seno y coseno para que
    diciembre y enero queden vecinos y no en extremos opuestos de una escala.
    """
    corrida = z.shift(1)
    columnas = {f"rezago_{k}": z.shift(k) for k in range(1, REZAGOS + 1)}
    posicion = 2 * np.pi * (z.index.month.to_numpy(dtype=float) - 1) / PERIODO
    columnas["mes_sin"] = pd.Series(np.sin(posicion), index=z.index)
    columnas["mes_cos"] = pd.Series(np.cos(posicion), index=z.index)
    for ventana in ESTADISTICAS:
        columnas[f"media_{ventana}"] = corrida.rolling(ventana).mean()
    columnas["desv_3"] = corrida.rolling(3).std()
    return pd.DataFrame(columnas, index=z.index)


def comprobar_sin_fuga(serie):
    """Prueba que ninguna fila ve el mes que predice.

    Mueve el valor de cada mes, uno por uno, y exige que su propia fila de la
    tabla quede intacta. Si un rezago, una ventana o una estadística se
    colaran con el mes que se predice, esa fila cambiaría y la prueba falla.
    """
    tabla = tabla_supervisada(objetivo(serie))
    comparados = 0
    for posicion, fecha in enumerate(serie.index):
        movida = serie.copy()
        movida.iloc[posicion] = movida.iloc[posicion] + 50.0
        otra = tabla_supervisada(objetivo(movida))
        for columna in tabla.columns:
            antes, despues = tabla[columna].iloc[posicion], otra[columna].iloc[posicion]
            if pd.isna(antes) and pd.isna(despues):
                continue
            assert antes == despues, (
                f"fuga en {fecha:%Y-%m}: la columna {columna} de esa fila cambió "
                f"de {antes} a {despues} al mover el valor del mismo mes")
            comparados += 1
    print(f"  Sin fuga: {len(serie)} meses movidos uno a uno; la fila de cada mes "
          f"quedó intacta ({comparados} valores comparados)")


def filas_completas(y):
    z = objetivo(y)
    X = tabla_supervisada(z)
    completas = X.notna().all(axis=1) & z.notna()
    return X[completas], z[completas]


def recursivo(ajustado, y, h):
    """Pronóstico recursivo a h meses.

    Agrega el mes siguiente a y como NaN, arma la última fila de la tabla,
    pronostica con ajustado.predict, devuelve el pronóstico al nivel con
    a_nivel y lo escribe como si fuera dato antes de pasar al mes siguiente.
    """
    serie = y.copy()
    pronosticos = []
    for _ in range(h):
        mes = serie.index[-1] + pd.offsets.MonthBegin(1)
        previo = serie                    # sin el mes que se pronostica
        serie.loc[mes] = np.nan
        ultima = tabla_supervisada(objetivo(serie)).iloc[[-1]]
        assert ultima.notna().all(axis=1).iloc[0], (
            f"fila incompleta al pronosticar {mes:%Y-%m}: {list(ultima.columns[ultima.isna().iloc[0]])}")
        zh = float(np.asarray(ajustado.predict(ultima)).ravel()[0])
        pronostico = float(a_nivel(previo, zh))
        pronosticos.append(pronostico)
        serie.loc[mes] = pronostico
    return np.array(pronosticos)


def con_tabla(fabrica):
    """Modelo para el backtest: arma la tabla con el entrenamiento del origen,
    ajusta un modelo nuevo y pronostica h meses en modo recursivo. El
    escalador vive dentro del modelo, así que se ajusta en cada origen."""
    def modelo(entrenamiento, h):
        X, z = filas_completas(entrenamiento)
        ajustado = fabrica()
        ajustado.fit(X, z)
        ULTIMO["modelo"] = ajustado
        return recursivo(ajustado, entrenamiento, h)
    return modelo


# ============================================================
# 3. Aprendizaje automático
# ============================================================
# Las cuatro fábricas devuelven un modelo nuevo sin entrenar; con_tabla
# ajusta uno por origen. KNN y SVR llevan el escalador dentro del pipeline.

def knn(n_neighbors=5):
    return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=n_neighbors))


def svr(C=1.0, epsilon=0.1):
    return make_pipeline(StandardScaler(), SVR(kernel="rbf", C=C, epsilon=epsilon))


def bosque(max_depth=None, min_samples_leaf=1):
    return RandomForestRegressor(n_estimators=100, max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf, random_state=SEMILLA)


def xgboost(max_depth=3, learning_rate=0.1):
    return XGBRegressor(n_estimators=200, max_depth=max_depth, learning_rate=learning_rate,
                        random_state=SEMILLA)


# Dos a cuatro valores por hiperparámetro. KNN y SVR son make_pipeline, así que
# la rejilla nombra el paso y el hiperparámetro (paso__hiperparámetro).
REJILLAS = {
    "KNN": (knn, {"kneighborsregressor__n_neighbors": [3, 5, 8, 12]}),
    "SVR": (svr, {"svr__C": [1.0, 10.0, 100.0], "svr__epsilon": [0.1, 0.5]}),
    "Random Forest": (bosque, {"max_depth": [None, 5, 10], "min_samples_leaf": [1, 3]}),
    "XGBoost": (xgboost, {"max_depth": [2, 3, 5], "learning_rate": [0.05, 0.1]}),
}


def sin_prefijo(parametros):
    return {nombre.split("__")[-1]: valor for nombre, valor in parametros.items()}


def ajustar_hiperparametros(serie):
    """Para cada modelo de REJILLAS, la combinación con menor RMSE en
    validación temporal.

    La tabla se arma con serie.iloc[:-N_PRUEBA], para que ninguno de los meses
    del backtest decida, y la búsqueda usa TimeSeriesSplit, que valida siempre
    sobre meses posteriores a los de entrenamiento y nunca baraja.
    """
    entrenamiento = serie.iloc[:-N_PRUEBA]
    X, z = filas_completas(entrenamiento)
    turnos = TimeSeriesSplit(n_splits=3)
    elegidos = {}
    print(f"  Rejilla sobre {len(X)} filas, de {X.index[0]:%Y-%m} a {X.index[-1]:%Y-%m}; "
          f"los {N_PRUEBA} meses del backtest no deciden")
    for nombre, (fabrica, rejilla) in REJILLAS.items():
        busqueda = GridSearchCV(fabrica(), rejilla, cv=turnos,
                                scoring="neg_root_mean_squared_error")
        busqueda.fit(X, z)
        elegidos[nombre] = sin_prefijo(busqueda.best_params_)
        print(f"    {nombre}: {elegidos[nombre]}, RMSE de validación "
              f"{-busqueda.best_score_:.2f}")
    return elegidos


def graficar_importancia(serie, elegidos):
    """Importancia de las variables de Random Forest y XGBoost, ajustados con
    lo elegido; guarda figuras/importancia.png."""
    X, z = filas_completas(serie.iloc[:-N_PRUEBA])
    figura, ejes = plt.subplots(1, 2, figsize=(11, 4))
    for eje, nombre in zip(ejes, ("Random Forest", "XGBoost")):
        fabrica = REJILLAS[nombre][0]
        ajustado = fabrica(**elegidos[nombre]).fit(X, z)
        importancia = pd.Series(ajustado.feature_importances_, index=X.columns).sort_values()
        eje.barh(importancia.index, importancia.to_numpy())
        eje.set_title(nombre, fontsize=10)
        eje.tick_params(labelsize=7)
        arriba = ", ".join(f"{columna} {valor:.2f}"
                           for columna, valor in importancia.tail(3)[::-1].items())
        print(f"    {nombre}, las tres que más pesan: {arriba}")
    figura.tight_layout()
    figura.savefig(FIGURAS / "importancia.png", dpi=120)
    plt.close(figura)


# ============================================================
# 4. Aprendizaje profundo en PyTorch
# ============================================================
# Redes pequeñas y pocas épocas: el guion completo tiene que correr en CPU
# en pocos minutos.
EPOCAS_MAXIMAS = 300
TASA = 0.01
# En Colab con GPU la red y sus tensores van a cuda; sin GPU, a la CPU. Red se
# encarga de los dos: las clases de cada red no tienen que mover nada.
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"


class Red:
    """Una red de PyTorch con fit y predict, para entrar a la tabla como un
    modelo de scikit-learn. Escala X y z con lo que ve en fit, que es el
    entrenamiento del origen, y devuelve el pronóstico en la escala original.
    Con secuencia=True la red recibe los rezagos en orden, con forma
    (lote, 12, 1), del más viejo al más reciente."""

    def __init__(self, construir, secuencia=False, epocas=EPOCAS_MAXIMAS, semilla=SEMILLA):
        self.construir = construir
        self.secuencia = secuencia
        self.epocas = epocas
        self.semilla = semilla

    def entradas(self, X):
        if self.secuencia:
            columnas = [f"rezago_{k}" for k in range(REZAGOS, 0, -1)]
            x = (X[columnas].to_numpy() - self.media) / self.desv
            return torch.tensor(x[:, :, None], dtype=torch.float32, device=DISPOSITIVO)
        return torch.tensor(self.escala_x.transform(X), dtype=torch.float32, device=DISPOSITIVO)

    def fit(self, X, z, X_val=None, z_val=None):
        torch.manual_seed(self.semilla)
        self.media, self.desv = z.mean(), z.std()
        self.escala_x = StandardScaler().fit(X)
        x = self.entradas(X)
        objetivo_t = torch.tensor(((z - self.media) / self.desv).to_numpy()[:, None],
                                  dtype=torch.float32, device=DISPOSITIVO)
        self.red = self.construir(x.shape[-1]).to(DISPOSITIVO)
        optimizador = torch.optim.Adam(self.red.parameters(), lr=TASA)
        perdida = nn.MSELoss()
        self.perdidas = []
        for _ in range(self.epocas):
            self.red.train()
            optimizador.zero_grad()
            error = perdida(self.red(x), objetivo_t)
            error.backward()
            optimizador.step()
            fila = [error.item()]
            if X_val is not None:
                fila.append(float(np.mean(((self.predict(X_val) - z_val) / self.desv) ** 2)))
            self.perdidas.append(fila)
        return self

    def predict(self, X):
        self.red.eval()
        with torch.no_grad():
            salida = self.red(self.entradas(X)).cpu().numpy()[:, 0]
        return salida * self.desv + self.media


class MLP(nn.Module):
    def __init__(self, entradas, ocultas=16):
        super().__init__()
        self.capas = nn.Sequential(nn.Linear(entradas, ocultas), nn.ReLU(),
                                   nn.Linear(ocultas, 1))

    def forward(self, x):
        return self.capas(x)


class LSTM(nn.Module):
    """Una capa LSTM sobre la secuencia (lote, 12, 1) y una capa lineal que
    lee el último estado oculto, que es el resumen de los 12 rezagos."""

    def __init__(self, ocultas=16):
        super().__init__()
        self.recurrente = nn.LSTM(input_size=1, hidden_size=ocultas,
                                  num_layers=1, batch_first=True)
        self.salida = nn.Linear(ocultas, 1)

    def forward(self, x):
        secuencia, _ = self.recurrente(x)
        return self.salida(secuencia[:, -1, :])


class Conv1D(nn.Module):
    """La tercera red: una convolución sobre la secuencia, la media de sus
    filtros y una capa lineal. Recibe la secuencia como la LSTM. La convolución
    aprende patrones locales de dos o tres meses vecinos, que es lo que la
    estacionalidad de esta serie necesita ver junta."""

    def __init__(self, filtros=16):
        super().__init__()
        self.convolucion = nn.Conv1d(in_channels=1, out_channels=filtros,
                                     kernel_size=3, padding=1)
        self.salida = nn.Linear(filtros, 1)

    def forward(self, x):
        # (lote, 12, 1) -> (lote, 1, 12), que es la forma que espera Conv1d.
        filtrada = torch.relu(self.convolucion(x.transpose(1, 2)))
        return self.salida(filtrada.mean(dim=2))


# Las candidatas: nombre, cómo se construye y si recibe la secuencia.
CANDIDATAS = {
    "MLP (8 ocultas)": (lambda n: MLP(n, 8), False),
    "MLP (32 ocultas)": (lambda n: MLP(n, 32), False),
    "LSTM (16 ocultas)": (lambda n: LSTM(16), True),
    "Conv1D (16 filtros)": (lambda n: Conv1D(16), True),
}


def validar_redes(serie):
    """Elige las épocas de cada red con validación temporal.

    Arma la tabla con serie.iloc[:-N_PRUEBA] y deja sus últimos 12 meses para
    validar. Entrena cada candidata con las épocas máximas, se queda con las
    de menor pérdida de validación y guarda las curvas en figuras/perdida.png.
    Devuelve {nombre: épocas}.
    """
    entrenamiento = serie.iloc[:-N_PRUEBA]
    X, z = filas_completas(entrenamiento)
    X_ent, z_ent = X.iloc[:-PERIODO], z.iloc[:-PERIODO]
    X_val, z_val = X.iloc[-PERIODO:], z.iloc[-PERIODO:]
    print(f"  Épocas por red: {len(X_ent)} filas para entrenar y las {PERIODO} de "
          f"validación, de {X_val.index[0]:%Y-%m} a {X_val.index[-1]:%Y-%m}")
    figura, eje = plt.subplots(figsize=(8, 3.8))
    elegidas = {}
    for nombre, (construir, secuencia) in CANDIDATAS.items():
        ajustada = Red(construir, secuencia, EPOCAS_MAXIMAS, SEMILLA).fit(
            X_ent, z_ent, X_val, z_val)
        perdidas = np.array(ajustada.perdidas)
        elegidas[nombre] = int(np.argmin(perdidas[:, 1])) + 1
        eje.plot(perdidas[:, 0], linewidth=1, label=f"{nombre}, entrenamiento")
        eje.plot(perdidas[:, 1], linewidth=1, linestyle="--",
                 label=f"{nombre}, validación")
        print(f"    {nombre}: {elegidas[nombre]} épocas, pérdida de validación "
              f"{perdidas[elegidas[nombre] - 1, 1]:.3f}")
    eje.set_yscale("log")
    eje.set_xlabel("época")
    eje.set_ylabel("error cuadrático medio, en unidades de z")
    eje.set_title("Curvas de pérdida con validación temporal")
    eje.legend(fontsize=7)
    figura.tight_layout()
    figura.savefig(FIGURAS / "perdida.png", dpi=120)
    plt.close(figura)
    return elegidas


def red(nombre, epocas, semilla=SEMILLA):
    """La fábrica de una red para con_tabla."""
    construir, secuencia = CANDIDATAS[nombre]
    return lambda: Red(construir, secuencia, epocas, semilla)


def repetir_con_semillas(serie, nombre, epocas):
    """La mejor red con cinco semillas, cada una por el backtest completo.

    Guarda resultados/semillas.csv con una fila por semilla e imprime la media
    y la desviación del RMSE: el número que entra a la tabla depende de dónde
    arranque el entrenamiento.
    """
    filas_semilla = []
    for semilla in range(SEMILLA, SEMILLA + 5):
        _, promedio = backtest(serie, con_tabla(red(nombre, epocas, semilla)))
        filas_semilla.append({"semilla": semilla,
                              "RMSE": round(promedio["RMSE"], 2),
                              "MAE": round(promedio["MAE"], 2),
                              "MAPE": round(promedio["MAPE"], 2)})
    tabla = pd.DataFrame(filas_semilla)
    tabla.to_csv(RESULTADOS / "semillas.csv", index=False)
    print(tabla.to_string(index=False))
    print(f"  {nombre}, {epocas} épocas: RMSE medio {tabla['RMSE'].mean():.2f} con "
          f"desviación {tabla['RMSE'].std():.2f} en {len(tabla)} semillas")
    return tabla


# ============================================================
# 5. La tabla y la elección
# ============================================================

def contar_parametros(ajustado):
    """Cuántos números guarda el modelo ajustado. En los clásicos y las redes
    son los parámetros; en KNN, los valores que memoriza; en SVR, los vectores
    de soporte y sus coeficientes; en los árboles, los nodos."""
    if hasattr(ajustado, "params"):
        return len(ajustado.params)
    if isinstance(ajustado, Red):
        return sum(p.numel() for p in ajustado.red.parameters())
    final = ajustado.steps[-1][1] if hasattr(ajustado, "steps") else ajustado
    if isinstance(final, KNeighborsRegressor):
        return final.n_samples_fit_ * final.n_features_in_
    if isinstance(final, SVR):
        return final.support_vectors_.size + final.dual_coef_.size + 1
    if isinstance(final, RandomForestRegressor):
        return sum(arbol.tree_.node_count for arbol in final.estimators_)
    if isinstance(final, XGBRegressor):
        return len(final.get_booster().trees_to_dataframe())
    return 0


def fila(nombre, serie, modelo):
    ULTIMO.clear()
    inicio = time.perf_counter()
    _, promedio = backtest(serie, modelo)
    segundos = time.perf_counter() - inicio
    parametros = contar_parametros(ULTIMO["modelo"]) if "modelo" in ULTIMO else 0
    print(f"  {nombre}: RMSE {promedio['RMSE']:.2f} en {segundos:.1f} s")
    return {"modelo": nombre, "RMSE": round(promedio["RMSE"], 2),
            "MAE": round(promedio["MAE"], 2), "MAPE": round(promedio["MAPE"], 2),
            "parametros": parametros, "segundos": round(segundos, 1)}


def graficar_pronosticos(serie, modelos):
    """Pronóstico desde el último origen del backtest de cada modelo elegido."""
    entrenamiento, real = serie.iloc[:-12], serie.iloc[-12:]
    figura, eje = plt.subplots(figsize=(9, 3.6))
    eje.plot(serie.index[-36:], serie.iloc[-36:].to_numpy(), color="black", label="real")
    for nombre, modelo in modelos.items():
        pronostico = modelo(entrenamiento, 12)
        eje.plot(real.index, pronostico, marker="o", markersize=3,
                 label=f"{nombre} (RMSE {rmse(real, pronostico):.1f})")
    eje.set_ylabel(VALOR)
    eje.set_title("Pronóstico a 12 meses desde el último origen")
    eje.legend(fontsize=8)
    figura.tight_layout()
    figura.savefig(FIGURAS / "pronosticos.png", dpi=120)
    plt.close(figura)


def pendiente(parte, error):
    print(f"  pendiente: {parte} ({error})")


def main():
    inicio = time.perf_counter()
    cruda = cargar_serie(ARCHIVO, FECHA, VALOR)
    print(f"Serie: {len(cruda)} meses, de {cruda.index[0]:%Y-%m} a "
          f"{cruda.index[-1]:%Y-%m}, {cruda.isna().sum()} vacíos")
    serie = preparar(cruda)
    graficar_serie(serie)

    print("1. Línea base y modelo a superar")
    filas = [fila("estacional ingenuo", serie,
                  lambda t, h: estacional_ingenuo(t, h, PERIODO))]
    elegidos_para_figura = {}
    try:
        clasico = modelo_clasico(serie)
        filas.append(fila(NOMBRE_CLASICO, serie, clasico))
        elegidos_para_figura[NOMBRE_CLASICO] = clasico
        atipicos_stl(serie)
        experimento_imputacion(serie)
    except NotImplementedError as error:
        pendiente("parte 1", error)

    print("2. Tabla supervisada")
    try:
        comprobar_sin_fuga(serie)
        tabla_lista = True
    except NotImplementedError as error:
        pendiente("parte 2", error)
        tabla_lista = False

    if tabla_lista:
        print("3. Aprendizaje automático")
        try:
            elegidos = ajustar_hiperparametros(serie)
            graficar_importancia(serie, elegidos)
            modelos_aa = {nombre: con_tabla(lambda f=fabrica, p=elegidos[nombre]: f(**p))
                          for nombre, (fabrica, _) in REJILLAS.items()}
            filas_aa = [fila(nombre, serie, modelo) for nombre, modelo in modelos_aa.items()]
            filas += filas_aa
            mejor_aa = min(filas_aa, key=lambda f: f["RMSE"])["modelo"]
            elegidos_para_figura[mejor_aa] = modelos_aa[mejor_aa]
        except NotImplementedError as error:
            pendiente("parte 3", error)

        print("4. Aprendizaje profundo")
        try:
            epocas = validar_redes(serie)
            filas_red = [fila(f"{nombre}, {e} épocas", serie, con_tabla(red(nombre, e)))
                         for nombre, e in epocas.items()]
            filas += filas_red
            mejor = min(zip(epocas, filas_red), key=lambda par: par[1]["RMSE"])[0]
            elegidos_para_figura[mejor] = con_tabla(red(mejor, epocas[mejor]))
            repetir_con_semillas(serie, mejor, epocas[mejor])
        except NotImplementedError as error:
            pendiente("parte 4", error)

    print("5. La tabla")
    tabla = pd.DataFrame(filas).sort_values("RMSE")
    tabla.to_csv(RESULTADOS / "metricas.csv", index=False)
    print(tabla.to_string(index=False))
    print("Mejor modelo por RMSE:", tabla.iloc[0]["modelo"])
    if len(elegidos_para_figura) > 1:
        graficar_pronosticos(serie, elegidos_para_figura)
    print(f"Tiempo total: {time.perf_counter() - inicio:.0f} s")


if __name__ == "__main__":
    main()
