# datos

La serie del grupo, la misma del miniproyecto 1, y la exógena de su mejor
modelo clásico.

| Campo | Valor |
|---|---|
| Archivo | `comercio_minorista_colombia.csv` |
| Qué mide | Índice de volumen del comercio minorista de Colombia, base 2015=100, sin desestacionalizar |
| Fuente | OCDE (Main Economic Indicators) vía FRED — https://fred.stlouisfed.org/series/COLSLRTTO01IXOBM |
| Licencia | Datos públicos de FRED, de uso libre con cita a la fuente |
| Frecuencia | mensual |
| Periodo | 2013-01 a 2024-03 (135 observaciones) |
| Columna de fecha | `observation_date` |
| Columna que se pronostica | `COLSLRTTO01IXOBM` |

## Exógena del SARIMAX

`festivos_colombia.csv`: número de festivos de Colombia en cada mes, de
2013-01 a 2025-03 (147 meses: la serie y los 12 que siguen a su final).
Columnas `mes` (primer día del mes) y `festivos`. Se generó una sola vez con
el paquete `holidays` (`holidays.country_holidays("CO")`), con el mismo
cálculo de la función `exogenas` del miniproyecto 1, y se guarda aquí para
que el guion solo use las librerías permitidas. El calendario de festivos es
fijo y se conoce de antemano, así que usarlo en los meses que se pronostican
no es fuga.
