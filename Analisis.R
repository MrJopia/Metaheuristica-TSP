############ Librerias ############


############ Archivos ############
setwd("C:/Users/benja/OneDrive/Escritorio/WorkSpace/Metaheuristica-TSP/Results/Parameters")
df_P_TS <- read.csv("trials_TS.csv")
df_P_TSS <- read.csv("trials_TSS.csv")


############ BoxPlots Parametrizaciones ############
boxplot(df$variable, 
        main = "Boxplot de Variable", 
        ylab = "Valores", 
        col = "lightblue", 
        border = "blue")