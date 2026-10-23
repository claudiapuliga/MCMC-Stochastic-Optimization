import gensim.downloader as api
import random
import math
import string
import collections
from collections import defaultdict

# =====================================================================
# FASE 1: MODELLO LINGUISTICO A TRIGRAMMI
# =====================================================================
def train_language_model(k_smoothing=1.0):
    print("Caricamento del corpus 'text8' in corso...")
    dataset = api.load("text8")
    
    trigram_counts = defaultdict(int)
    bigram_counts = defaultdict(int)
    alphabet = set(string.ascii_uppercase)
    
    prev1 = None
    prev2 = None
    for words in dataset:
        for word in words:
            word = word.upper()
            for char in word:
                if char in alphabet:
                    if prev1 is not None and prev2 is not None:
                        trigram = prev2 + prev1 + char
                        bigram = prev2 + prev1
                        trigram_counts[trigram] += 1
                        bigram_counts[bigram] += 1
                    prev2 = prev1
                    prev1 = char
                    
    V = 26 
    log_probs = {}
    for a in string.ascii_uppercase:
        for b in string.ascii_uppercase:
            for c in string.ascii_uppercase:
                prob = (trigram_counts[a+b+c] + k_smoothing) / (bigram_counts[a+b] + k_smoothing * V)
                log_probs[a+b+c] = math.log(prob)
                
    print("Modello linguistico a Trigrammi pronto!\n")
    return log_probs

# =====================================================================
# FASE 2: FUNZIONE FITNESS CON ENTROPIA
# =====================================================================
def calculate_fitness(text, log_probs):
    score = 0.0
    for i in range(len(text) - 2):
        trigram = text[i:i+3]
        if trigram in log_probs:
            score += log_probs[trigram]
        else:
            score -= 20.0 
            
    # Calcolo dell'Entropia di Shannon per evitare la ridondanza dei trigrammi
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
    # 1. Creiamo un bacino con almeno 65 simboli per coprire tutte le esigenze (15 in più del necessario per eventuali implementazioni future))
    symbol_string = "Δ∇⊕⊗★☾☀♈♉♊♋♌♍♎♏♐♑♒♓♠♣♥♦†‡✓✗△▽◁▷ΩΣΦΨΠΓΘΞΛΥαβγδεζηθικλμνξοπρστυφχψω"

    pool = list(symbol_string)
    random.shuffle(pool)
    
    # 2. Definiamo 24 omofoni extra (Totale simboli: 26 + 24 = 50)
    # Questa distribuzione ricalca fedelmente le frequenze della lingua inglese
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
# FASE 4: METROPOLIS-HASTINGS & SIMULATED ANNEALING
# =====================================================================
def decrypt_mcmc_sa_epoch(ciphertext, log_probs, T_start=15.0, T_fin=0.5, cooling_rate=0.90, iter_per_epoch=15000):
    
    # 1. CALCOLO DINAMICO DELLE EPOCHE
    epochs_needed = math.ceil(math.log(T_fin / T_start) / math.log(cooling_rate))
    
    unique_symbols = list(set(ciphertext))
    alphabet = list(string.ascii_uppercase)
    
    current_key = {sym: random.choice(alphabet) for sym in unique_symbols}
    current_text = apply_key(ciphertext, current_key)
    current_score = calculate_fitness(current_text, log_probs)
    
    best_text = current_text
    best_score = current_score
    
    T = T_start
    history = []
    global_iter = 0
    
    # 2. Ciclo esterno basato sul calcolo matematico
    for epoch in range(epochs_needed):
        
        # 3. Ciclo interno a temperatura fissa (15.000 iterazioni)
        for _ in range(iter_per_epoch):
            sym = random.choice(unique_symbols)
            old_val = current_key[sym]
            new_val = random.choice(alphabet)
            
            if old_val == new_val:
                continue
                
            current_key[sym] = new_val
            new_text = apply_key(ciphertext, current_key)
            new_score = calculate_fitness(new_text, log_probs)
            
            delta_S = new_score - current_score
            
            if delta_S > 0:
                alpha = 1.0 
            else:
                if T < 1e-10:
                    alpha = 0.0
                else:
                    alpha = math.exp(delta_S / T) 
                
            if random.random() < alpha:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_text = new_text
            else:
                current_key[sym] = old_val
                
            global_iter += 1
            if global_iter % 1000 == 0:
                history.append(current_score)
                
        # 4. Raffreddamento
        T *= cooling_rate
        
    return best_text, best_score, history, epochs_needed

# =====================================================================
# ESECUZIONE
# =====================================================================
if __name__ == "__main__":
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
    
    ciphertext, real_key = encrypt_homophonic(plaintext)
    
    print("Testo originale   :", plaintext[:65], "...")
    print(f"Simboli unici nel crittogramma: {len(set(ciphertext))}")
    print("-" * 65)
    
    log_probs = train_language_model(k_smoothing=1.0)
    
    restarts = 5  # 5 restart necessari per garantire un test affidabile
    global_best_score = float('-inf')
    global_best_text = ""
    
    # Parametri esatti per il raffreddamento, eventualmente modificabili per test
    T_INIZIALE = 15.0
    T_FINALE = 0.5
    LAMBDA_COOLING = 0.90
    ITER_EPOCA = 15000
    
    for run in range(restarts):
        print(f"\n--- AVVIO TENTATIVO {run + 1} DI {restarts} ---")
        text, score, _, epoche_usate = decrypt_mcmc_sa_epoch(
            ciphertext, 
            log_probs, 
            T_start=T_INIZIALE, 
            T_fin=T_FINALE, 
            cooling_rate=LAMBDA_COOLING, 
            iter_per_epoch=ITER_EPOCA
        )
        
        # Calcolo percentuale sul singolo tentativo
        matches_run = sum(1 for a, b in zip(plaintext, text) if a == b)
        acc_run = (matches_run / len(plaintext)) * 100
        
        print(f"Epoche calcolate: {epoche_usate} | Iterazioni Totali: {epoche_usate * ITER_EPOCA}")
        print(f"Risultato: {text[:50]}... | Score: {score:.2f} | Acc: {acc_run:.2f}%")
        
        if score > global_best_score:
            global_best_score = score
            global_best_text = text
            
    print("\n" + "=" * 65)
    print("RISULTATO FINALE DELLA CATENA MCMC (Migliore dei tentativi):")
    print(f"OBIETTIVO : {plaintext}")
    print(f"OTTENUTO  : {global_best_text}")
    
    # Calcolo dell'accuratezza totale sul testo migliore
    matches = sum(1 for a, b in zip(plaintext, global_best_text) if a == b)
    accuracy = (matches / len(plaintext)) * 100
    print(f"\nACCURATEZZA FINALE ASSOLUTA: {accuracy:.2f}% delle lettere corrette")