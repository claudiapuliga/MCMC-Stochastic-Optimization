import random
import math
import string
import collections
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import gensim.downloader as api

# =====================================================================
# ADDESTRAMENTO DEI DUE MODELLI LINGUISTICI (4-GRAMMI)
# =====================================================================
def train_comparative_models_tetragrams(l4=0.4, l3=0.3, l2=0.2, l1=0.1):
    print(f"Caricamento corpus 'text8' per 4-Grammi (Lambda: {l4}, {l3}, {l2}, {l1})...", flush=True)
    dataset = api.load("text8")
    
    uni_counts = defaultdict(int)
    bi_counts = defaultdict(int)
    tri_counts = defaultdict(int)
    tetra_counts = defaultdict(int)
    alphabet = string.ascii_uppercase
    
    total_unigrams = 0
    total_tetragrams = 0
    prev1 = prev2 = prev3 = None
    
    for words in dataset:
        for word in words:
            word = word.upper()
            for char in word:
                if char in alphabet:
                    uni_counts[char] += 1
                    total_unigrams += 1
                    if prev1 is not None:
                        bi_counts[prev1+char] += 1
                        if prev2 is not None:
                            tri_counts[prev2+prev1+char] += 1
                            if prev3 is not None:
                                tetra_counts[prev3+prev2+prev1+char] += 1
                                total_tetragrams += 1
                    prev3, prev2, prev1 = prev2, prev1, char

    print("Calcolo delle probabilità in corso (generazione di 456.976 combinazioni per l'Interpolazione)...", flush=True)
    
    # 1. Modello Laplace Ibrido (MLE puro per le combinazioni esistenti + Floor Penalty)
    log_probs_hybrid = {}
    for tetra, count in tetra_counts.items():
        log_probs_hybrid[tetra] = math.log(count / total_tetragrams)
        
    floor_penalty = math.log(0.01 / total_tetragrams)
    
    # 2. Modello Interpolazione Lineare a 4 livelli
    log_probs_interp = {}
    for a in alphabet:
        for b in alphabet:
            for c in alphabet:
                for d in alphabet:
                    tetra = a + b + c + d
                    tri_pre = a + b + c
                    tri_suf = b + c + d
                    bi_pre = b + c
                    bi_suf = c + d
                    
                    p4 = tetra_counts[tetra] / tri_counts[tri_pre] if tri_counts[tri_pre] > 0 else 0
                    p3 = tri_counts[tri_suf] / bi_counts[bi_pre] if bi_counts[bi_pre] > 0 else 0
                    p2 = bi_counts[bi_suf] / uni_counts[c] if uni_counts[c] > 0 else 0
                    p1 = uni_counts[d] / total_unigrams
                    
                    prob_interp = (l4 * p4) + (l3 * p3) + (l2 * p2) + (l1 * p1)
                    if prob_interp > 0:
                        log_probs_interp[tetra] = math.log(prob_interp)
                        
    print("Modelli a 4-Grammi pronti!\n", flush=True)
    return log_probs_hybrid, floor_penalty, log_probs_interp

# =====================================================================
# FUNZIONI DI FITNESS E CIFRARIO
# =====================================================================
def calculate_fitness(text, log_probs, fallback_penalty, n_gram=4):
    score = 0.0
    for i in range(len(text) - (n_gram - 1)):
        score += log_probs.get(text[i:i+n_gram], fallback_penalty)
        
    counts = collections.Counter(text)
    entropy = 0.0
    for count in counts.values():
        p = count / len(text)
        entropy -= p * math.log(p)
        
    if entropy < 2.5:
        score -= 5000.0
    return score

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
        
    return "".join(random.choice(cipher_dict[c]) for c in plaintext)

def apply_key(ciphertext, key):
    return ''.join(key.get(c, c) for c in ciphertext)

# =====================================================================
# MOTORE MCMC DINAMICO CON TRACCIAMENTO ACCURATEZZA
# =====================================================================
def decrypt_mcmc_tracked_epochs(ciphertext, log_probs, plaintext, fallback_penalty, n_gram, T_start=15.0, T_fin=0.5, cooling_rate=0.90, iter_per_epoch=15000):
    epochs_needed = math.ceil(math.log(T_fin / T_start) / math.log(cooling_rate))
    
    unique_symbols = list(set(ciphertext))
    alphabet = list(string.ascii_uppercase)
    current_key = {sym: random.choice(alphabet) for sym in unique_symbols}
    
    best_text = apply_key(ciphertext, current_key)
    current_score = calculate_fitness(best_text, log_probs, fallback_penalty, n_gram)
    best_score = current_score
    T = T_start
    
    history_acc = []
    
    for epoch in range(epochs_needed):
        for _ in range(iter_per_epoch):
            sym = random.choice(unique_symbols)
            old_val = current_key[sym]
            new_val = random.choice(alphabet)
            
            if old_val == new_val: continue
                
            current_key[sym] = new_val
            new_text = apply_key(ciphertext, current_key)
            new_score = calculate_fitness(new_text, log_probs, fallback_penalty, n_gram)
            
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
        matches = sum(1 for a, b in zip(plaintext, best_text) if a == b)
        history_acc.append((matches / len(plaintext)) * 100)
            
    return history_acc, epochs_needed

# =====================================================================
# ESECUZIONE DEL TEST (SALVANDO IL TENTATIVO MIGLIORE DI N RUN)
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
    
    # Parametri Interpolazione Tetragrammi
    l4, l3, l2, l1 = 0.7, 0.2, 0.05, 0.05
    ciphertext = encrypt_homophonic(plaintext)
    
    log_probs_hybrid, floor_penalty, log_probs_interp = train_comparative_models_tetragrams(l4=l4, l3=l3, l2=l2, l1=l1)
    
    runs = 10  
    label_interp = f"Interpolazione ($\lambda_4={l4}, \lambda_3={l3}, \lambda_2={l2}, \lambda_1={l1}$)"
    
    # PARAMETRI TERMODINAMICI (Epoche Lento - Settati come il test migliore precedente)
    N_GRAM_SIZE = 4
    T_START = 15.0
    T_FIN = 0.5
    LAMBDA_COOL = 0.90
    ITER_EP = 15000

    print(f"\n--- Parametri Fissi: T_start={T_START}, T_fin={T_FIN}, Lambda={LAMBDA_COOL}, Iter={ITER_EP} ---", flush=True)
    
    print(f"\n--- Esecuzione Test 1: LAPLACE IBRIDO (con Floor Penalty) ---", flush=True)
    results_hybrid = []
    epochs_hybrid = 0
    for r in range(runs):
        # Utilizza il vero Floor Penalty dinamico
        acc, ep = decrypt_mcmc_tracked_epochs(ciphertext, log_probs_hybrid, plaintext, fallback_penalty=floor_penalty, n_gram=N_GRAM_SIZE, T_start=T_START, T_fin=T_FIN, cooling_rate=LAMBDA_COOL, iter_per_epoch=ITER_EP)
        results_hybrid.append(acc)
        epochs_hybrid = ep
        print(f"Run {r+1} completato (Accuratezza finale: {acc[-1]:.2f}%)", flush=True)
        
    best_hybrid_idx = np.argmax([res[-1] for res in results_hybrid])
    best_hist_acc_hybrid = results_hybrid[best_hybrid_idx]
    
    print(f"\n--- Esecuzione Test 2: {label_interp.upper()} ---", flush=True)
    results_interp = []
    epochs_interp = 0
    for r in range(runs):
        # L'interpolazione contiene tutte le combinazioni > 0, il fallback estremo interviene solo per underflow di probabilità
        acc, ep = decrypt_mcmc_tracked_epochs(ciphertext, log_probs_interp, plaintext, fallback_penalty=-50.0, n_gram=N_GRAM_SIZE, T_start=T_START, T_fin=T_FIN, cooling_rate=LAMBDA_COOL, iter_per_epoch=ITER_EP)
        results_interp.append(acc)
        epochs_interp = ep
        print(f"Run {r+1} completato (Accuratezza finale: {acc[-1]:.2f}%)", flush=True)
        
    best_interp_idx = np.argmax([res[-1] for res in results_interp])
    best_hist_acc_interp = results_interp[best_interp_idx]
    
    x_axis = np.arange(1, epochs_hybrid + 1)
    
    print("\n" + "="*60, flush=True)
    print("RIEPILOGO ACCURATEZZA (IL MIGLIORE DEI 5 TENTATIVI) - 4-GRAMMI", flush=True)
    print("="*60, flush=True)
    print(f"Laplace Ibrido (Floor Penalty) : {best_hist_acc_hybrid[-1]:.2f}%", flush=True)
    print(f"Interpolazione Lineare         : {best_hist_acc_interp[-1]:.2f}%", flush=True)
    print("="*60 + "\n", flush=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, best_hist_acc_hybrid, label="Laplace Ibrido (Floor Penalty)", color='#2ca02c', linewidth=2.5, marker='o', markersize=4)
    plt.plot(x_axis, best_hist_acc_interp, label=label_interp, color='#d62728', linestyle='--', linewidth=2.5, marker='x', markersize=4)
    plt.title(f"Evoluzione Accuratezza 4-Grammi (Miglior Tentativo | $T_{{start}}={T_START}$)", fontsize=13, fontweight='bold')
    plt.xlabel("Epoche di Raffreddamento", fontsize=11)
    plt.ylabel("% Lettere Corrette", fontsize=11)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_path_acc = os.path.expanduser('~/grafico_accuratezza_4grammi_T15.png')
    plt.savefig(save_path_acc, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Grafico Accuratezza salvato in: {save_path_acc}", flush=True)