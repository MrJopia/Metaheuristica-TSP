############ Librerias ############
library(ggplot2)
library(dplyr)

############ Archivos ############
# Leer el archivo CSV y preparar los datos
setwd("C:/Users/Benjamin Gonzalez/Desktop/Workspace/Metaheuristica-TSP/Results/Experimentals")
df_C_TS <- read.csv("TS_converge_38.csv")
df_C_GLS <- read.csv("GLS_converge_38.csv")

df_TS_int38 <- read.csv("TS_results_38.txt")
df_TS_int76 <- read.csv("TS_results_76.txt")
df_TS_int194 <- read.csv("TS_results_194_1.txt")
df_GLS_int38 <- read.csv("GLS_results_38.txt")
df_GLS_int76 <- read.csv("GLS_results_76.txt")
df_GLS_int194 <- read.csv("GLS_results_194_1.txt")

############ Convergencia ############
# Preparar los datos para el gráfico de convergencia
convergence_data_TS <- data.frame(
  Iteration = 1:length(df_C_TS$Mejor),  # Número de iteración
  ObjectiveValue = df_C_TS$Mejor        # Valores de la función objetivo
)

convergence_data_GLS <- data.frame(
  Iteration = 1:length(df_C_GLS$Mejor),  # Número de iteración
  ObjectiveValue = df_C_GLS$Mejor        # Valores de la función objetivo
)

# Encontrar el valor mínimo registrado en el dataframe
min_value_TS <- min(df_C_TS$Mejor)
min_value_GLS <- min(df_C_GLS$Mejor)

# Crear el gráfico de convergencia
g_TS <- ggplot(convergence_data_TS, aes(x = Iteration, y = ObjectiveValue)) +
  geom_line(color = "blue") +             # Línea que conecta los puntos
  geom_point(color = "red") +             # Puntos en cada iteración
  geom_hline(yintercept = min_value_TS,      # Línea horizontal con el valor mínimo
             color = "green",             # Color de la línea
             linetype = "dashed",        # Tipo de línea (puedes cambiarlo a "solid" si prefieres una línea continua)
             size = 1) +                 # Grosor de la línea
  labs(title = "Gráfico de Convergencia dj38 TS (80.000)",
       x = "Iteración",
       y = "GAP normalizado") +
  theme_minimal()


g_GLS <- ggplot(convergence_data_GLS, aes(x = Iteration, y = ObjectiveValue)) +
  geom_line(color = "blue") +             # Línea que conecta los puntos
  geom_point(color = "red") +             # Puntos en cada iteración
  geom_hline(yintercept = min_value_GLS,      # Línea horizontal con el valor mínimo
             color = "green",             # Color de la línea
             linetype = "dashed",        # Tipo de línea (puedes cambiarlo a "solid" si prefieres una línea continua)
             size = 1) +                 # Grosor de la línea
  labs(title = "Gráfico de Convergencia dj38 GLS (80.000)",
       x = "Iteración",
       y = "GAP normalizado") +
  theme_minimal()

# Mostrar el gráfico
print(g_GLS)
print(g_TS)

############ Boxplots ############

df_38 <- bind_rows(
  data.frame(Caso = "TS", Error = df_TS_int38$Error),
  data.frame(Caso = "GLS", Error = df_GLS_int38$Error),
)
df_76 <- bind_rows(
  data.frame(Caso = "TS", Error = df_TS_int76$Error),
  data.frame(Caso = "GLS", Error = df_GLS_int76$Error),
)
df_194 <- bind_rows(
  data.frame(Caso = "TS", Error = df_TS_int194$Error),
  data.frame(Caso = "GLS", Error = df_GLS_int194$Error),
)

g_r_38 <- ggplot(df_38, aes(x = Caso, y = Error, fill = Caso)) +
  geom_boxplot(alpha = 0.7) + 
  labs(
    title = "Boxplots de Error normalizado dj38 (80.000 llamadas)",
    x = "Caso",
    y = "Error"
  ) +
  theme_minimal() +
  theme(legend.position = "none", 
        axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1))

g_r_76 <- ggplot(df_76, aes(x = Caso, y = Error, fill = Caso)) +
  geom_boxplot(alpha = 0.7) + 
  labs(
    title = "Boxplots de Error normalizado pr76 (80.000 llamadas)",
    x = "Caso",
    y = "Error"
  ) +
  theme_minimal() +
  theme(legend.position = "none", 
        axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1))

g_r_194 <- ggplot(df_194, aes(x = Caso, y = Error, fill = Caso)) +
  geom_boxplot(alpha = 0.7) + 
  labs(
    title = "Boxplots de Error normalizado qa194 (80.000 llamadas)",
    x = "Caso",
    y = "Error"
  ) +
  theme_minimal() +
  theme(legend.position = "none", 
        axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1))


print(g_r_38)
print(g_r_76)
print(g_r_194)

############ Normalidad ############
resultado_shapiro <- shapiro.test(df_TS_int38$Error)
print(resultado_shapiro)
resultado_shapiro <- shapiro.test(df_TS_int76$Error)
print(resultado_shapiro)
resultado_shapiro <- shapiro.test(df_TS_int194$Error)
print(resultado_shapiro)

resultado_shapiro <- shapiro.test(df_GLS_int38$Error)
print(resultado_shapiro)
resultado_shapiro <- shapiro.test(df_GLS_int76$Error)
print(resultado_shapiro)
resultado_shapiro <- shapiro.test(df_GLS_int194$Error)
print(resultado_shapiro)


############ Test Estadístico ############
resultado <- t.test(df_TS_int38$Error, df_GLS_int38$Error, paired = TRUE)
print(resultado$p.value)
resultado <- t.test(df_TS_int76$Error, df_GLS_int76$Error, paired = TRUE)
print(resultado$p.value)
resultado <- t.test(df_TS_int194$Error, df_GLS_int194$Error, paired = TRUE)
print(resultado$p.value)