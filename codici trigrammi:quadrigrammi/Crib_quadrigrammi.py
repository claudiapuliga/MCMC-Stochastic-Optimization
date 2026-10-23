import gensim.downloader as api
import random
import math
import string
import collections
from collections import defaultdict

# =====================================================================
# FASE 1: MODELLO LINGUISTICO A 4-GRAMMI
# =====================================================================
def train_language_model_4grams():
    print("Caricamento corpus 'text8' per 4-grammi...", flush=True)
    dataset = api.load("text8")
    
    tetra_counts = defaultdict(int)
    alphabet = set(string.ascii_uppercase)
    buffer = []
    total_tetragrams = 0
    
    for words in dataset:
        for word in words:
            word = word.upper()
            for char in word:
                if char in alphabet:
                    buffer.append(char)
                    if len(buffer) == 4:
                        tetra_counts["".join(buffer)] += 1
                        total_tetragrams += 1
                        buffer.pop(0)
                    
    log_probs = {}
    for tetra, count in tetra_counts.items():
        log_probs[tetra] = math.log(count / total_tetragrams)
        
    floor_penalty = math.log(0.01 / total_tetragrams)
    return log_probs, floor_penalty

def calculate_fitness(text, log_probs, floor_penalty):
    score = 0.0
    for i in range(len(text) - 3):
        score += log_probs.get(text[i:i+4], floor_penalty)
            
    counts = collections.Counter(text)
    entropy = 0.0
    for count in counts.values():
        p = count / len(text)
        entropy -= p * math.log(p)
        
    if entropy < 2.5:
        score -= 5000.0
    return score

# =====================================================================
# FASE 2: CIFRARIO OMOFONICO
# =====================================================================
def encrypt_homophonic(plaintext):
    symbol_string = "Δ∇⊕⊗★☾☀♈♉♊♋♌♍♎♏♐♑♒♓♠♣♥♦†‡✓✗△▽◁▷ΩΣΦΨΠΓΘΞΛΥαβγδεζηθικλμνξοπρστυφχψω"
    pool = list(symbol_string)
    random.shuffle(pool)
    
    extra_homophones = {'E': 6, 'T': 4, 'A': 3, 'O': 3, 'I': 3, 'N': 2, 'S': 1, 'R': 1, 'H': 1}
    cipher_dict = {}
    for char in string.ascii_uppercase:
        num_symbols = 1 + extra_homophones.get(char, 0)
        cipher_dict[char] = [pool.pop() for _ in range(num_symbols)]
            
    ciphertext = "".join(random.choice(cipher_dict[c]) for c in plaintext)
    return ciphertext, cipher_dict

def apply_key(ciphertext, key):
    return ''.join(key.get(c, c) for c in ciphertext)

# =====================================================================
# FASE 3: MOTORE MCMC FISSANDO LE PAROLE CON CRIB (PAROLE CORRETTE)
# =====================================================================
def decrypt_mcmc_sa_crib(ciphertext, log_probs, floor_penalty, crib_dict=None, T_start=15.0, T_fin=0.5, cooling_rate=0.90, iter_per_epoch=15000):
    epochs_needed = math.ceil(math.log(T_fin / T_start) / math.log(cooling_rate))
    unique_symbols = list(set(ciphertext))
    alphabet = list(string.ascii_uppercase)
    
    current_key = {}
    crib_dict = crib_dict or {}
    locked_symbols = set(crib_dict.keys())
    mutable_symbols = [sym for sym in unique_symbols if sym not in locked_symbols]
    
    for sym in unique_symbols:
        if sym in locked_symbols:
            current_key[sym] = crib_dict[sym] 
        else:
            current_key[sym] = random.choice(alphabet)
            
    current_text = apply_key(ciphertext, current_key)
    current_score = calculate_fitness(current_text, log_probs, floor_penalty)
    
    best_score = current_score
    best_text = current_text
    T = T_start
    
    for epoch in range(epochs_needed):
        for _ in range(iter_per_epoch):
            if not mutable_symbols: 
                break
            sym = random.choice(mutable_symbols)
            
            old_val = current_key[sym]
            new_val = random.choice(alphabet)
            if old_val == new_val: continue
                
            current_key[sym] = new_val
            new_text = apply_key(ciphertext, current_key)
            new_score = calculate_fitness(new_text, log_probs, floor_penalty)
            
            delta_S = new_score - current_score
            alpha = 1.0 if delta_S > 0 else (0.0 if T < 1e-10 else math.exp(delta_S / T))
                
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
    
    ciphertext, _ = encrypt_homophonic(plaintext)
    log_probs, floor_penalty = train_language_model_4grams()
    
    # 1. Definiamo i frammenti di testo che l'algoritmo aveva già decifrato correttamente
    good_cribs = [
        "IHOPE", "LOTSOFFUN", "BRINGSUP", "ABOUT", "BECAUSE", 
        "ALLTHESOONERBECAUSE", "ENOUGHS", "EVERYONEELSEHASNOTHING", 
        "REACH", "LIFE", "ONEIN"
    ]
    
    crib_dict = {}
    for crib in good_cribs:
        crib_start = plaintext.find(crib)
        if crib_start != -1:
            for i in range(len(crib)):
                ct_sym = ciphertext[crib_start + i]
                pt_char = crib[i]
                crib_dict[ct_sym] = pt_char
    
    print("\n" + "="*75)
    print(f"PAROLE CORRETTE BLOCCATE: {', '.join(good_cribs)}")
    print(f"Simboli crittografici fissati: {len(crib_dict)} su {len(set(ciphertext))} totali")
    print("="*75)
    
    RESTARTS = 3 # 3 restart sono sufficienti avendo ridotto lo spazio delle chiavi
    T_INIZIALE = 15.0
    T_FINALE = 0.5
    LAMBDA_COOLING = 0.90
    ITER_EPOCA = 15000
    
    best_score_with_crib = float('-inf')
    best_text_with_crib = ""
    
    print("\n--- ESECUZIONE MCMC PER MIGLIORAMENTO ACCURATEZZA ---", flush=True)
    for run in range(RESTARTS):
        text, score = decrypt_mcmc_sa_crib(
            ciphertext, log_probs, floor_penalty, crib_dict=crib_dict, 
            T_start=T_INIZIALE, T_fin=T_FINALE, cooling_rate=LAMBDA_COOLING, iter_per_epoch=ITER_EPOCA
        )
        matches = sum(1 for a, b in zip(plaintext, text) if a == b)
        acc = (matches / len(plaintext)) * 100
        print(f"  Run {run+1} | Score: {score:.2f} | Acc: {acc:.2f}%", flush=True)
        
        if score > best_score_with_crib:
            best_score_with_crib = score
            best_text_with_crib = text
            
    final_acc_with_crib = (sum(1 for a, b in zip(plaintext, best_text_with_crib) if a == b) / len(plaintext)) * 100
    
    print("\n" + "="*75)
    print("RISULTATO FINALE POST-OTTIMIZZAZIONE CRIB")
    print("="*75)
    print(f"OBIETTIVO :\n{plaintext}\n")
    print(f"OTTENUTO ({final_acc_with_crib:.2f}%):\n{best_text_with_crib}")