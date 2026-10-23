# Ottimizzazione Stocastica e Modellazione Markoviana: Algoritmi MCMC per Spazi ad Alta Dimensionalità

**Autore:** Claudia Puliga
**Università:** Università degli Studi di Perugia - Laurea Magistrale in Matematica (2025/2026)

## Obiettivo del Progetto
Questo repository presenta l'implementazione in Python di algoritmi avanzati di ottimizzazione stocastica, sviluppati originariamente per la crittanalisi di cifrari storici a sostituzione omofonica. Il codice dimostra la capacità di modellare sistemi ad altissima entropia e di forzare la convergenza verso massimi globali in spazi di ricerca combinatori estremamente vasti.

## Architettura Algoritmica
*   **Markov Chain Monte Carlo (MCMC) & Simulated Annealing:** Sviluppo di un motore stocastico con cooling schedule dinamico (a raffreddamento continuo e ad epoche) per la navigazione del paesaggio energetico e il superamento dei massimi locali.
*   **Modellazione Probabilistica (N-grammi):** Implementazione di modelli markoviani (fino a 4-grammi) per la previsione sequenziale e la massimizzazione della cross-entropia tra il modello linguistico e l'output.
*   **Gestione del Rumore e Smoothing:** Integrazione di tecniche di regolarizzazione matematica (Laplace/Add-k Smoothing, Interpolazione Lineare e Floor Penalty) per gestire la variazione dei dati e il problema della frequenza zero.
*   **High Performance e Ottimizzazione:** Strutturazione del codice per supportare Random Restarts multipli e ancoraggi deterministici (metodo Crib), riducendo drasticamente il *mixing time* della catena e abbattendo l'equivocazione residua per raggiungere un'accuratezza superiore al 90%.
