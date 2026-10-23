import gensim.downloader as api
import random
import math
import string
import collections

# =====================================================================
# FASE 1: MODELLO LINGUISTICO A 4-GRAMMI (PROBABILITÀ ASSOLUTA E FLOOR)
# =====================================================================
def train_language_model(n_gram=4):
    print(f"Caricamento del corpus 'text8' per modello a {n_gram}-grammi")
    dataset = api.load("text8")
    
    ngram_counts = collections.defaultdict(int)
    alphabet = set(string.ascii_uppercase)
    
    total_ngrams = 0
    buffer = []
    
    for words in dataset:
        for word in words:
            word = word.upper()
            for char in word:
                if char in alphabet:
                    buffer.append(char)
                    if len(buffer) == n_gram:
                        ngram = "".join(buffer)
                        ngram_counts[ngram] += 1
                        total_ngrams += 1
                        buffer.pop(0) 
                        
    log_probs = {}
    for ngram, count in ngram_counts.items():
        log_probs[ngram] = math.log(count / total_ngrams)
            
    #Il Floor Penalty per i tetragrammi inventati/non visti.
    floor_penalty = math.log(0.01 / total_ngrams)
    
    print(f"Trovate {len(log_probs)} combinazioni uniche.")
    print(f"Totale {n_gram}-grammi nel corpus: {total_ngrams}")
    print(f"Floor Penalty matematica calcolata a: {floor_penalty:.2f}\n")
    return log_probs, floor_penalty

# =====================================================================
# FASE 2: FUNZIONE FITNESS 
# =====================================================================
def calculate_fitness(text, log_probs, floor_penalty, n_gram=4):
    score = 0.0
    
    for i in range(len(text) - (n_gram - 1)):
        chunk = text[i:i+n_gram]
        score += log_probs.get(chunk, floor_penalty)
            
    counts = collections.Counter(text)
    entropy = 0.0
    for count in counts.values():
        p = count / len(text)
        entropy -= p * math.log(p)
        
    if entropy < 2.5:
        score -= 5000.0
        
    return score

# =====================================================================
# FASE 3: CIFRARIO OMOFONICO 
# =====================================================================
def encrypt_homophonic(plaintext):
    
    symbol_string = "Δ∇⊕⊗★☾☀♈♉♊♋♌♍♎♏♐♑♒♓♠♣♥♦†‡✓✗△▽◁▷ΩΣΦΨΠΓΘΞΛΥαβγδεζηθικλμνξοπρστυφχψω"
    pool = list(symbol_string)
    random.shuffle(pool)
    
    extra_homophones = {
        'E': 6, 'T': 4, 'A': 3, 'O': 3, 'I': 3, 'N': 2, 
        'S': 1, 'R': 1, 'H': 1
    }
    
    cipher_dict = {}
    for char in string.ascii_uppercase:
        num_symbols = 1 + extra_homophones.get(char, 0)
        cipher_dict[char] = [pool.pop() for _ in range(num_symbols)]
            
    ciphertext = ""
    for char in plaintext:
        if char in cipher_dict:
            ciphertext += random.choice(cipher_dict[char])
            
    return ciphertext, cipher_dict

def apply_key(ciphertext, key):
    return ''.join(key.get(c, c) for c in ciphertext)

# =====================================================================
# FASE 4: SIMULATED ANNEALING A EPOCHE (DINAMICO)
# =====================================================================
def decrypt_mcmc_sa_epoch(ciphertext, log_probs, floor_penalty, n_gram=4, T_start=15.0, T_fin=0.5, cooling_rate=0.90, iter_per_epoch=15000):
    
    # Calcolo dinamico basato su matematica termodinamica
    epochs_needed = math.ceil(math.log(T_fin / T_start) / math.log(cooling_rate))
    
    unique_symbols = list(set(ciphertext))
    alphabet = list(string.ascii_uppercase)
    
    current_key = {sym: random.choice(alphabet) for sym in unique_symbols}
    current_text = apply_key(ciphertext, current_key)
    current_score = calculate_fitness(current_text, log_probs, floor_penalty, n_gram)
    
    best_text = current_text
    best_score = current_score
    
    T = T_start
    history = []
    
    for epoch in range(epochs_needed):
        for i in range(iter_per_epoch):
            sym = random.choice(unique_symbols)
            old_val = current_key[sym]
            new_val = random.choice(alphabet)
            
            if old_val == new_val:
                continue
                
            current_key[sym] = new_val
            new_text = apply_key(ciphertext, current_key)
            new_score = calculate_fitness(new_text, log_probs, floor_penalty, n_gram)
            
            delta_S = new_score - current_score
            
            if delta_S > 0:
                alpha = 1.0 
            else:
                alpha = 0.0 if T < 1e-10 else math.exp(delta_S / T) 
                
            if random.random() < alpha:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_text = new_text
            else:
                current_key[sym] = old_val
                
            if (i + 1 + epoch * iter_per_epoch) % 1000 == 0:
                history.append(current_score)
                
        T *= cooling_rate
        
    return best_text, best_score, history, epochs_needed

# =====================================================================
# ESECUZIONE 
# =====================================================================
if __name__ == "__main__":
    # Si sceglie il testo in chiaro , deve essere tutto maiuscolo e attaccato senza spazi o punteggiatura 
    plaintext = (
        "IHOPEYOUAREHAVINGLOTSOFFUNINTRYINGTOCATCHME"
        "THATWASNTMEONTHETVSHOWWHICHBRINGSUPAPOINTABOUTME"
        "IAMNOTAFRAIDOFTHEGASCHAMBERBECAUSEITWILLSENDMETO"
        "PARADISEALLTHESOONERBECAUSEINOWHAVEENOUGHSLAVES"
        "TOWORKFORMEWHEREASEEVERYONEELSEHASNOTHINGWHENTHEY"
        "REACHPARADISESOTHEYAREAFRAIDOFDEATHIAMNOTAFRAID"
        "BECAUSEIKNOWTHATMYNEWLIFEWILLBEANEASYONEIN"
        "PARADISEDEATH"
    )
    
    print("="*75)
    print("TEST CRITTANALISI: TETRAGRAMMI CON FLOOR PENALTY (N=4)")
    print("="*75)
    
    ciphertext, real_key = encrypt_homophonic(plaintext)
    
    n_gram_size = 4
    log_probs, floor_penalty = train_language_model(n_gram=n_gram_size)
    
    restarts = 5 
    global_best_score = float('-inf')
    global_best_text = ""
    # Parametri da modificare per capire come si comporta il sistema rispetto alla variazione dei parametri fissi 
    T_INIZIALE = 15.0
    T_FINALE = 0.5
    LAMBDA_COOLING = 0.90
    ITER_EPOCA = 15000
    
    for run in range(restarts):
        print(f"\n--- AVVIO TENTATIVO {run + 1} DI {restarts} ---")
        
        text, score, hist, epoche_usate = decrypt_mcmc_sa_epoch(
            ciphertext, log_probs, floor_penalty, n_gram=n_gram_size, 
            T_start=T_INIZIALE, T_fin=T_FINALE, cooling_rate=LAMBDA_COOLING, iter_per_epoch=ITER_EPOCA 
        )
        
        matches_partial = sum(1 for a, b in zip(plaintext, text) if a == b)
        accuracy_partial = (matches_partial / len(plaintext)) * 100
        
        print(f"Epoche calcolate: {epoche_usate} | Iterazioni Totali: {epoche_usate * ITER_EPOCA}")
        #Visualizza solo i primi 50 caratteri del testo decifrato per evitare un output troppo lungo
        print(f"Risultato: {text[:50]}... | Score: {score:.2f} | Accuratezza: {accuracy_partial:.2f}%")
        # Qui si potrebbe porre una modifica per valutare la funzione fitness corrente invece della migliore globale 
        if score > global_best_score:
            global_best_score = score
            global_best_text = text
            
    print("\n" + "="*75)
    print("RISULTATO FINALE DELLA CATENA MCMC (Migliore dei 5 tentativi):")
    print(f"OBIETTIVO : {plaintext}")
    print(f"OTTENUTO  : {global_best_text}")
    
    matches_final = sum(1 for a, b in zip(plaintext, global_best_text) if a == b)
    accuracy_final = (matches_final / len(plaintext)) * 100
    print(f"\nACCURATEZZA FINALE (MIGLIORE): {accuracy_final:.2f}% delle lettere corrette")