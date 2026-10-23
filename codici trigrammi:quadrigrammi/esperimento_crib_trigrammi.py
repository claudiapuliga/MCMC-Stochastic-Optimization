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
                
    print("Modello linguistico a Trigrammi pronto\n")
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
# FASE 4: MCMC CON CRIBBING A BLOCCHI FISSI
# =====================================================================
def decrypt_mcmc_sa_crib_epoch(ciphertext, log_probs, T_start=15.0, T_fin=0.5, cooling_rate=0.90, iter_per_epoch=15000, locked_keys=None):
    if locked_keys is None:
        locked_keys = {}

    epochs_needed = math.ceil(math.log(T_fin / T_start) / math.log(cooling_rate))

    unique_symbols = list(set(ciphertext))
    alphabet = list(string.ascii_uppercase)
    
    current_key = {}
    for sym in unique_symbols:
        if sym in locked_keys:
            current_key[sym] = locked_keys[sym]
        else:
            current_key[sym] = random.choice(alphabet)
            
    # Lista di simboli che l'algoritmo può modificare (esclusi quelli fissati)
    unlocked_symbols = [s for s in unique_symbols if s not in locked_keys]
    
    current_text = apply_key(ciphertext, current_key)
    current_score = calculate_fitness(current_text, log_probs)
    
    best_text = current_text
    best_score = current_score
    T = T_start
    
    if not unlocked_symbols:
        return best_text, best_score
    
    for epoch in range(epochs_needed):
        for _ in range(iter_per_epoch):
            sym = random.choice(unlocked_symbols)
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
                alpha = 0.0 if T < 1e-10 else math.exp(delta_S / T) 
                
            if random.random() < alpha:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_text = new_text
            else:
                current_key[sym] = old_val
                
        T *= cooling_rate
        
    return best_text, best_score

# =====================================================================
# ESECUZIONE DELL'ATTACCO CON CRIB FISSATI
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
    
    # --- ANCORAGGIO MANUALE E FISSO DEI CRIB ---
    fixed_cribs = {
        "I": 0,
        "CATCH": 36,
        "THAT": 43,
        "THE": 56,        
        "ABOUT": 84,
        "BECAUSE": 118,    
        "THEYREACH": 231
    }
    
    locked_keys = {}
    for word, start_idx in fixed_cribs.items():
        for i in range(len(word)):
            simbolo_cifrato = ciphertext[start_idx + i]
            lettera_chiaro = word[i]
            locked_keys[simbolo_cifrato] = lettera_chiaro
            
    print(f"\nCRIB FISSATI MANUALMENTE: {list(fixed_cribs.keys())}")
    print(f"Simboli bloccati e intoccabili: {len(locked_keys)} su {len(set(ciphertext))}")
    print("-" * 65)
    
    log_probs = train_language_model(k_smoothing=1.0)
    
    restarts = 5
    global_best_score = float('-inf')
    global_best_text = ""
    
    T_INIZIALE = 15.0
    T_FINALE = 0.5
    LAMBDA_COOLING = 0.90
    ITER_EPOCA = 15000
    
    for run in range(restarts):
        print(f"--- AVVIO TENTATIVO {run + 1} DI {restarts} ---")
        text, score = decrypt_mcmc_sa_crib_epoch(
            ciphertext, 
            log_probs, 
            T_start=T_INIZIALE, 
            T_fin=T_FINALE, 
            cooling_rate=LAMBDA_COOLING, 
            iter_per_epoch=ITER_EPOCA,
            locked_keys=locked_keys
        )
        
        matches = sum(1 for a, b in zip(plaintext, text) if a == b)
        acc_attuale = (matches / len(plaintext)) * 100
        
        print(f"Risultato parziale: {text[:50]}... | Score: {score:.2f} | Acc: {acc_attuale:.2f}%")
        
        if score > global_best_score:
            global_best_score = score
            global_best_text = text
            
    print("-" * 65)
    print("\nRISULTATO FINALE DELLA CATENA MCMC (Con Cribbing Rigido):")
    print(f"OBIETTIVO : {plaintext}")
    print(f"OTTENUTO  : {global_best_text}")
    
    matches = sum(1 for a, b in zip(plaintext, global_best_text) if a == b)
    accuracy = (matches / len(plaintext)) * 100
    print(f"\nACCURATEZZA FINALE ASSOLUTA: {accuracy:.2f}%")