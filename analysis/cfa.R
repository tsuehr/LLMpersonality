setwd("/psychometrics/analysis")
library(lavaan)
library(ltm)
library(psych)


##nocontext personas
data_O <- read.csv('./preprocessed_final/llama_2_personas_factor_data_O.csv')
data_E <- read.csv('./preprocessed_final/llama_2_personas_factor_data_E.csv')
data_C <- read.csv('./preprocessed_final/llama_2_personas_factor_data_C.csv')
data_A <- read.csv('./preprocessed_final/llama_2_personas_factor_data_A.csv')
data_N <- read.csv('./preprocessed_final/llama_2_personas_factor_data_N.csv')

data_O <- read.csv('./preprocessed_final/human_factor_data_O.csv')
data_E <- read.csv('./preprocessed_final/human_factor_data_E.csv')
data_C <- read.csv('./preprocessed_final/human_factor_data_C.csv')
data_A <- read.csv('./preprocessed_final/human_factor_data_A.csv')
data_N <- read.csv('./preprocessed_final/human_factor_data_N.csv')

data_O <- read.csv('./preprocessed_final/gpt4_personas_factor_data_O.csv')
data_E <- read.csv('./preprocessed_final/gpt4_personas_factor_data_E.csv')
data_C <- read.csv('./preprocessed_final/gpt4_personas_factor_data_C.csv')
data_A <- read.csv('./preprocessed_final/gpt4_personas_factor_data_A.csv')
data_N <- read.csv('./preprocessed_final/gpt4_personas_factor_data_N.csv')

data_O <- read.csv('./preprocessed_final/chatgpt_personas_factor_data_O.csv')
data_E <- read.csv('./preprocessed_final/chatgpt_personas_factor_data_E.csv')
data_C <- read.csv('./preprocessed_final/chatgpt_personas_factor_data_C.csv')
data_A <- read.csv('./preprocessed_final/chatgpt_personas_factor_data_A.csv')
data_N <- read.csv('./preprocessed_final/chatgpt_personas_factor_data_N.csv')

##incontext personas
data_O <- read.csv('./preprocessed_final/incontext/llama2_70b_chat_incontext_personas_factor_data_O.csv')
data_E <- read.csv('./preprocessed_final/incontext/llama2_70b_chat_incontext_personas_factor_data_E.csv')
data_C <- read.csv('./preprocessed_final/incontext/llama2_70b_chat_incontext_personas_factor_data_C.csv')
data_A <- read.csv('./preprocessed_final/incontext/llama2_70b_chat_incontext_personas_factor_data_A.csv')
data_N <- read.csv('./preprocessed_final/incontext/llama2_70b_chat_incontext_personas_factor_data_N.csv')

data_O <- read.csv('./preprocessed_final/incontext/gpt35_incontext_personas_factor_data_O.csv')
data_E <- read.csv('./preprocessed_final/incontext/gpt35_incontext_personas_factor_data_E.csv')
data_C <- read.csv('./preprocessed_final/incontext/gpt35_incontext_personas_factor_data_C.csv')
data_A <- read.csv('./preprocessed_final/incontext/gpt35_incontext_personas_factor_data_A.csv')
data_N <- read.csv('./preprocessed_final/incontext/gpt35_incontext_personas_factor_data_N.csv')

data_O <- read.csv('./preprocessed_final/incontext/gpt4_incontext_personas_factor_data_O.csv')
data_E <- read.csv('./preprocessed_final/incontext/gpt4_incontext_personas_factor_data_E.csv')
data_C <- read.csv('./preprocessed_final/incontext/gpt4_incontext_personas_factor_data_C.csv')
data_A <- read.csv('./preprocessed_final/incontext/gpt4_incontext_personas_factor_data_A.csv')
data_N <- read.csv('./preprocessed_final/incontext/gpt4_incontext_personas_factor_data_N.csv')

######seeded empty personas incontext##############

data_O <- read.csv('./preprocessed_final/incontext/llama2_70b_chat_incontext_empty_personas_seeded_factor_data_O.csv')
data_E <- read.csv('./preprocessed_final/incontext/llama2_70b_chat_incontext_empty_personas_seeded_factor_data_E.csv')
data_C <- read.csv('./preprocessed_final/incontext/llama2_70b_chat_incontext_empty_personas_seeded_factor_data_C.csv')
data_A <- read.csv('./preprocessed_final/incontext/llama2_70b_chat_incontext_empty_personas_seeded_factor_data_A.csv')
data_N <- read.csv('./preprocessed_final/incontext/llama2_70b_chat_incontext_empty_personas_seeded_factor_data_N.csv')

data_O <- read.csv('./preprocessed_final/incontext/gpt35_incontext_empty_personas_seeded_factor_data_O.csv')
data_E <- read.csv('./preprocessed_final/incontext/gpt35_incontext_empty_personas_seeded_factor_data_E.csv')
data_C <- read.csv('./preprocessed_final/incontext/gpt35_incontext_empty_personas_seeded_factor_data_C.csv')
data_A <- read.csv('./preprocessed_final/incontext/gpt35_incontext_empty_personas_seeded_factor_data_A.csv')
data_N <- read.csv('./preprocessed_final/incontext/gpt35_incontext_empty_personas_seeded_factor_data_N.csv')

data_O <- read.csv('./preprocessed_final/incontext/gpt4_incontext_empty_personas_seeded_factor_data_O.csv')
data_E <- read.csv('./preprocessed_final/incontext/gpt4_incontext_empty_personas_seeded_factor_data_E.csv')
data_C <- read.csv('./preprocessed_final/incontext/gpt4_incontext_empty_personas_seeded_factor_data_C.csv')
data_A <- read.csv('./preprocessed_final/incontext/gpt4_incontext_empty_personas_seeded_factor_data_A.csv')
data_N <- read.csv('./preprocessed_final/incontext/gpt4_incontext_empty_personas_seeded_factor_data_N.csv')

#item names of the openpsy dataset
model_all_openpsy <- 'C=~CSN1+CSN2+CSN3+CSN4+CSN5+CSN6+CSN7+CSN8+CSN9+CSN10
                      E=~EXT1+EXT2+EXT3+EXT4+EXT5+EXT6+EXT7+EXT8+EXT9+EXT10
                      N=~EST1+EST2+EST3+EST4+EST5+EST6+EST7+EST8+EST9+EST10
                      A=~AGR1+AGR2+AGR3+AGR4+AGR5+AGR6+AGR7+AGR8+AGR9+AGR10
                      O=~OPN1+OPN2+OPN3+OPN4+OPN5+OPN6+OPN7+OPN8+OPN9+OPN10'

'
#this is the 5 component (full model) in the paper
model_all_free <- '
              E =~ E0 + E1 + E2 + E3 + E4 + E5 + E6 + E7 + E8 + E9 + E10 + E11
              A =~ A0 + A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A10 + A11 + A9
              C =~ C0 + C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8 + C9 + C10 + C11
              N =~ N0 + N1 + N2 + N3 + N4 + N5 + N6 + N7 + N8 + N9 + N10 + N11
              O =~ O0 + O1 + O2 + O3 + O4 + O5 + O6 + O7 + O8 + O9 + O10 + O11

'

data_all<- cbind(data_O, data_E, data_C, data_A, data_N)
fit_all <- cfa(model_all_free,std.lv=TRUE,check.gradient = FALSE, data=data_all)
fitMeasures(fit_all, c('cfi', 'tli', 'rmsea', 'srmr'))


###Single component
model_E <- 'E =~ E0 + E1 + E2 + E3 + E4 + E5 + E6 + E7 + E8 + E9 + E10 + E11'
model_A <- 'A =~ A0 + A2 + A3 + A4 + A5 + A6 + A7 + A8  + A10 + A11 + A9 + A1'
model_C <- 'C =~ C0 + C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8 + C9 + C10 + C11'
model_N <- 'N =~ N0 + N1 + N2 + N3 + N4 + N5 + N6 + N7 + N8 + N9 + N10 + N11'
model_O <- 'O =~ O0 + O1 + O2 + O3 + O4 + O5 + O6 + O7 + O8 + O9 + O10 + O11'

##3 sub components###

model_E <- 'Sociability =~ E0 + E1 + E2 + E3 
            Assertiveness =~ E4 + E6 + E7 + E5 
            EnergyLevel =~ E8 + E9 + E10 + E11
            Sociability ~~ 0*Assertiveness
            Sociability ~~ 0*EnergyLevel
            Assertiveness ~~0*EnergyLevel'

model_A <- 'EF1 =~ A0 + A2 + A3 + A1
            EF2 =~ A4 + A5 + A6 + A7 
            EF3 =~ A8 + A10 + A11 + A9
            EF1 ~~ 0*EF2
            EF1 ~~ 0*EF3
            EF2 ~~ 0*EF3'


model_C <- 'EF1 =~ C0 + C1 + C2 + C3 
            EF2 =~ C4 + C5 + C6 + C7 
            EF3 =~ C8 + C9 + C10 + C11
            EF1 ~~ 0*EF2
            EF1 ~~ 0*EF3
            EF2 ~~ 0*EF3
            '

model_N <- 'EF1 =~ N0 + N1 + N2 + N3 
            EF2 =~ N4 + N5 + N6 + N7 
            EF3 =~ N8 + N9 + N10 + N11
            EF1 ~~ 0*EF2
            EF1 ~~ 0*EF3
            EF2 ~~ 0*EF3
'

model_O <- 'EF1 =~ O0 + O1 + O2 + O3 
            EF2 =~ O4 + O5 + O6 + O7 
            EF3 =~ O8 + O9 + O10 + O11
            EF1 ~~ 0*EF2
            EF1 ~~ 0*EF3
            EF2 ~~ 0*EF3'

######
####### 3 sub-components + acquiescence####

model_E <- 'Sociability =~ E0 + E1 + E2 + E3 
            Assertiveness =~ E4 + E5 + E6 + E7 
            EnergyLevel =~ E8 + E9 + E10 + E11
            general_factor =~ E0 + E1 + E2 + E3 + E4 + E5 + E6 + E7 + E8 + E9 + E10 + E11
            Sociability ~~ 0*Assertiveness
            Sociability ~~ 0*EnergyLevel
            Assertiveness ~~0*EnergyLevel
            Sociability ~~ 0*general_factor
            Assertiveness ~~ 0*general_factor
            EnergyLevel ~~ 0*general_factor
'

model_A <- 'EF1 =~ A0 + A1 + A2 + A3 
            EF2 =~ A4 + A5 + A6 + A7 
            EF3 =~ A8 + A9 + A10 + A11
            general_factor =~ A0 + A1 + A2 + A3 + A4 + A5 + A6 + A7+ A8 + A9 + A10 + A11
            EF1 ~~ 0*EF2
            EF1 ~~ 0*EF3
            EF2 ~~ 0*EF3
            EF1 ~~ 0*general_factor
            EF2 ~~ 0*general_factor
            EF3 ~~ 0*general_factor
            
'

model_C <- 'EF1 =~ C0 + C1 + C2 + C3 
            EF2 =~ C4 + C5 + C6 + C7 
            EF3 =~ C8 + C9 + C10 + C11
            general_factor =~ C0 + C1 + C2 + C3 + C4 + C5 + C6 + C7+ C8 + C9 + C10 + C11
            EF1 ~~ 0*EF2
            EF1 ~~ 0*EF3
            EF2 ~~ 0*EF3
            EF1 ~~ 0*general_factor
            EF2 ~~ 0*general_factor
            EF3 ~~ 0*general_factor
            '

model_N <- 'EF1 =~ N0 + N1 + N2 + N3 
            EF2 =~ N4 + N5 + N6 + N7 
            EF3 =~ N8 + N9 + N10 + N11
            general_factor =~ N0 + N1 + N2 + N3 + N4 + N5 + N6 + N7 + N8 + N9 + N10 + N11
            EF1 ~~ 0*EF2
            EF1 ~~ 0*EF3
            EF2 ~~ 0*EF3
            EF1 ~~ 0*general_factor
            EF2 ~~ 0*general_factor
            EF3 ~~ 0*general_factor
'

model_O <- 'EF1 =~ O0 + O1 + O2 + O3 
            EF2 =~ O4 + O5 + O6 + O7 
            EF3 =~ O8 + O9 + O10 + O11
            g=~ O0 + O1 + O2 + O3 + O4 + O5 + O6 + O7 + O8 + O9 + O10 + O11
            EF1 ~~ 0*EF2
            EF1 ~~ 0*EF3
            EF2 ~~ 0*EF3
            EF1 ~~ 0*g
            EF2 ~~ 0*g
            EF3 ~~ 0*g
'
############################################

fit_E <- cfa(model_E,check.gradient = FALSE, std.lv=TRUE, data=data_E)
##gpt3.5 drop item A37_-1
data_A <- subset(data_A, select = -A9)
fit_A <- cfa(model_A,check.gradient = FALSE, std.lv=TRUE, data=data_A)

fit_C <- cfa(model_C,check.gradient = FALSE, std.lv=TRUE, data=data_C)
fit_N <- cfa(model_N,check.gradient = FALSE ,std.lv=TRUE, data=data_N)
fit_O <- cfa(model_O,check.gradient = FALSE, std.lv=TRUE, data=data_O)
summary(fit_E)

E<-fitMeasures(fit_E, c('cfi', 'tli', 'rmsea'))
A<-fitMeasures(fit_A, c('cfi', 'tli', 'rmsea'))
C<-fitMeasures(fit_C, c('cfi', 'tli', 'rmsea'))
N<-fitMeasures(fit_N, c('cfi', 'tli', 'rmsea'))
O<-fitMeasures(fit_O, c('cfi', 'tli', 'rmsea'))
(E['cfi'] + A['cfi'] + C['cfi']+N['cfi']+O['cfi'])/5
(E['tli'] + A['tli'] + C['tli']+N['tli']+O['tli'])/5
(E['rmsea'] + A['rmsea'] + C['rmsea']+N['rmsea']+O['rmsea'])/5

################################Reliability##################################################

w_E <- omegah(data_E,nfactors=1)$omega_h 
a_E <- omegah(data_E,nfactors=1)$alpha

w_A <- omegah(data_A,nfactors=1)$omega_h
a_A <- omegah(data_A,nfactors=1)$alpha

w_C <- omegah(data_C,nfactors=1)$omega_h
a_C <- omegah(data_C,nfactors=1)$alpha

w_N <- omegah(data_N,nfactors=1)$omega_h
a_N <- omegah(data_N,nfactors=1)$alpha

w_O <- omegah(data_O,nfactors=1)$omega_h
a_O <- omegah(data_O,nfactors=1)$alpha

(w_E + w_A + w_C + w_N + w_O)/5
(a_E + a_A + a_C + a_N + a_O)/5
