setwd("/psychometrics/analysis")

#####PCA#####
llama_pca <- read.csv('./preprocessed_final/incontext/gpt35_incontext_personas_factor_data.csv')

ncomp <- 5

pca_rotated <- psych::principal(llama_pca, rotate="varimax", nfactors=ncomp, scores=TRUE)
print(pca_rotated$loadings)
#write.csv(pca_rotated$loadings, file = './preprocessed_final/incontext/pca_varimax_gpt35_incontext_personas.csv', row.names = TRUE)

