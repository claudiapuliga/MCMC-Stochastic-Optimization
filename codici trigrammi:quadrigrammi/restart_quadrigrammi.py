import gensim.downloader as api
import random
import math
import string
import collections
from collections import defaultdict
import matplotlib.pyplot as plt
import os

# =====================================================================
# FASE 1: MODELLO LINGUISTICO A 4-GRAMMI CON FLOOR PENALTY
# =====================================================================
def train_language_model_4grams():
    print("Caricamento del corpus 'text8' per 4-grammi in corso...", flush=True)
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
                        tetra = "".join(buffer)
                        tetra_counts[tetra] += 1
                        total_tetragrams += 1
                        buffer.pop(0)
                    
    log_probs = {}
    for tetra, count in tetra_counts.items():
        log_probs[tetra] = math.log(count / total_tetragrams)
        
    floor_penalty = math.log(0.01 / total_tetragrams)
                
    print(f"Modello linguistico a 4-Grammi pronto. Trovate {len(log_probs)} combinazioni.\n", flush=True)
    return log_probs, floor_penalty

# =====================================================================
# FASE 2: FUNZIONE FITNESS A 4-GRAMMI
# =====================================================================
def calculate_fitness(text, log_probs, floor_penalty):
    score = 0.0
    for i in range(len(text) - 3):
        tetra = text[i:i+4]
        score += log_probs.get(tetra, floor_penalty)
            
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
# FASE 4: METROPOLIS-HASTINGS & SIMULATED ANNEALING
# =====================================================================
def decrypt_mcmc_sa_epoch(ciphertext, log_probs, floor_penalty, T_start=15.0, T_fin=0.5, cooling_rate=0.90, iter_per_epoch=15000):
    epochs_needed = math.ceil(math.log(T_fin / T_start) / math.log(cooling_rate))
    
    unique_symbols = list(set(ciphertext))
    alphabet = list(string.ascii_uppercase)
    
    current_key = {sym: random.choice(alphabet) for sym in unique_symbols}
    current_text = apply_key(ciphertext, current_key)
    current_score = calculate_fitness(current_text, log_probs, floor_penalty)
    
    best_text = current_text
    best_score = current_score
    
    T = T_start
    
    for epoch in range(epochs_needed):
        for _ in range(iter_per_epoch):
            sym = random.choice(unique_symbols)
            old_val = current_key[sym]
            new_val = random.choice(alphabet)
            
            if old_val == new_val:
                continue
                
            current_key[sym] = new_val
            new_text = apply_key(ciphertext, current_key)
            new_score = calculate_fitness(new_text, log_probs, floor_penalty)
            
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
                
        T *= cooling_rate
        
    return best_text, best_score, epochs_needed

# =====================================================================
# ESECUZIONE E CREAZIONE GRAFICO RESTART
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
    
    print("Testo originale   :", plaintext[:65], "...", flush=True)
    print(f"Simboli unici nel crittogramma: {len(set(ciphertext))}", flush=True)
    print("-" * 65, flush=True)
    
    log_probs, floor_penalty = train_language_model_4grams()
    
    restarts = 10 
    global_best_score = float('-inf')
    global_best_text = ""
    current_best_acc = 0.0
    
    T_INIZIALE = 15.0
    T_FINALE = 0.5
    LAMBDA_COOLING = 0.90
    ITER_EPOCA = 15000
    
    x_axis = list(range(1, restarts + 1))
    y_fitness_cumulativa = []
    y_accuratezza_correlata = []
    
    for run in range(restarts):
        print(f"\n--- AVVIO TENTATIVO {run + 1} DI {restarts} ---", flush=True)
        text, score, epoche_usate = decrypt_mcmc_sa_epoch(
            ciphertext, 
            log_probs, 
            floor_penalty,
            T_start=T_INIZIALE, 
            T_fin=T_FINALE, 
            cooling_rate=LAMBDA_COOLING, 
            iter_per_epoch=ITER_EPOCA
        )
        
        matches_run = sum(1 for a, b in zip(plaintext, text) if a == b)
        acc_run = (matches_run / len(plaintext)) * 100
        
        print(f"Epoche calcolate: {epoche_usate} | Iterazioni Totali: {epoche_usate * ITER_EPOCA}", flush=True)
        print(f"Risultato: {text[:50]}... | Score: {score:.2f} | Acc: {acc_run:.2f}%", flush=True)
        
        if score > global_best_score:
            global_best_score = score
            global_best_text = text
            current_best_acc = acc_run
            
        y_fitness_cumulativa.append(global_best_score)
        y_accuratezza_correlata.append(current_best_acc)
            
    print("\n" + "=" * 65, flush=True)
    print("RISULTATO FINALE DELLA CATENA MCMC (Migliore dei tentativi):", flush=True)
    print(f"OBIETTIVO : {plaintext}", flush=True)
    print(f"OTTENUTO  : {global_best_text}", flush=True)
    
    matches = sum(1 for a, b in zip(plaintext, global_best_text) if a == b)
    accuracy = (matches / len(plaintext)) * 100
    print(f"\nACCURATEZZA FINALE ASSOLUTA: {accuracy:.2f}% delle lettere corrette", flush=True)
    
    # ==========================================
    # DISEGNO DEL GRAFICO A DOPPIO ASSE
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Numero di Riavvii (Restarts)', fontsize=11)
    ax1.set_ylabel('Log-Fitness Massima Cumulativa', color=color1, fontsize=11)
    ax1.step(x_axis, y_fitness_cumulativa, where='post', color=color1, linewidth=2.5, marker='o')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x_axis)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2 = ax1.twinx()  
    color2 = 'tab:green'
    ax2.set_ylabel('Accuratezza (%) associata al Best Score', color=color2, fontsize=11)
    ax2.step(x_axis, y_accuratezza_correlata, where='post', color=color2, linewidth=2.5, linestyle='--', marker='x')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title("Impatto dei Riavvii (10 Restarts) - Modello a 4-Grammi (Laplace Ibrido)", fontsize=12, fontweight='bold')
    
    save_path = os.path.expanduser('~/grafico_esperimento_restarts_4grammi.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nGrafico salvato in: {save_path}", flush=True)