import random
import math
import string
import collections
import matplotlib.pyplot as plt
import numpy as np
import os
import gensim.downloader as api

def train_language_model(n_gram):
    print(f"Caricamento del corpus 'text8' per modello a {n_gram}-grammi...", flush=True)
    dataset = api.load("text8")
    ngram_counts = collections.defaultdict(int)
    alphabet = set(string.ascii_uppercase)
    buffer = []
    total_ngrams = 0
    
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
        
    floor_penalty = math.log(0.01 / total_ngrams)
    
    print(f"Modello pronto! Trovate {len(log_probs)} combinazioni.\n", flush=True)
    return log_probs, floor_penalty

def calculate_fitness(text, log_probs, floor_penalty, n_gram):
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

def encrypt_homophonic(plaintext):
    symbol_string = "Δ∇⊕⊗★☾☀♈♉♊♋♌♍♎♏♐♑♒♓♠♣♥♦†‡✓✗△▽◁▷ΩΣΦΨΠΓΘΞΓΔΛΞΠΣΥΦΨΩαβγδεζηθ"
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

def mcmc_epoch_dynamic(ciphertext, log_probs, floor_penalty, n_gram, T_start, T_fin, cooling_rate, iter_per_epoch):
    epochs_needed = math.ceil(math.log(T_fin / T_start) / math.log(cooling_rate))
    total_iterations = epochs_needed * iter_per_epoch
    
    unique_symbols = list(set(ciphertext))
    alphabet = list(string.ascii_uppercase)
    current_key = {sym: random.choice(alphabet) for sym in unique_symbols}
    current_score = calculate_fitness(apply_key(ciphertext, current_key), log_probs, floor_penalty, n_gram)
    
    T = T_start
    history = []
    global_iter = 0
    
    for epoch in range(epochs_needed):
        for _ in range(iter_per_epoch):
            sym = random.choice(unique_symbols)
            old_val = current_key[sym]
            new_val = random.choice(alphabet)
            
            if old_val != new_val:
                current_key[sym] = new_val
                new_score = calculate_fitness(apply_key(ciphertext, current_key), log_probs, floor_penalty, n_gram)
                delta_S = new_score - current_score
                alpha = 1.0 if delta_S > 0 else (0.0 if T < 1e-10 else math.exp(delta_S / T))
                    
                if random.random() < alpha:
                    current_score = new_score
                else:
                    current_key[sym] = old_val
            
            global_iter += 1
            if global_iter % 100 == 0: 
                history.append(current_score)
                
        T *= cooling_rate
        
    return history, total_iterations, epochs_needed

def mcmc_continuous_dynamic(ciphertext, log_probs, floor_penalty, n_gram, T_start, T_fin, cooling_rate):
    total_iterations = math.ceil(math.log(T_fin / T_start) / math.log(cooling_rate))
    
    unique_symbols = list(set(ciphertext))
    alphabet = list(string.ascii_uppercase)
    current_key = {sym: random.choice(alphabet) for sym in unique_symbols}
    current_score = calculate_fitness(apply_key(ciphertext, current_key), log_probs, floor_penalty, n_gram)
    
    T = T_start
    history = []
    
    for i in range(total_iterations):
        sym = random.choice(unique_symbols)
        old_val = current_key[sym]
        new_val = random.choice(alphabet)
        
        if old_val != new_val:
            current_key[sym] = new_val
            new_score = calculate_fitness(apply_key(ciphertext, current_key), log_probs, floor_penalty, n_gram)
            delta_S = new_score - current_score
            alpha = 1.0 if delta_S > 0 else (0.0 if T < 1e-10 else math.exp(delta_S / T))
                
            if random.random() < alpha:
                current_score = new_score
            else:
                current_key[sym] = old_val
                
        T *= cooling_rate
        
        if (i + 1) % 100 == 0: 
            history.append(current_score)
            
    return history, total_iterations

if __name__ == "__main__":
    plaintext = "IHOPEYOUAREHAVINGLOTSOFFUNINTRYINGTOCATCHMETHATWASNTMEONTHETVSHOWWHICHBRINGSUPAPOINTABOUTMEIAMNOTAFRAIDOFTHEGASCHAMBERBECAUSEITWILLSENDMETOPARADISEALLTHESOONERBECAUSEINOWHAVEENOUGHSLAVESTOWORKFORMEWHEREASEEVERYONEELSEHASNOTHINGWHENTHEYREACHPARADISESOTHEYAREAFRAIDOFDEATHIAMNOTAFRAIDBECAUSEIKNOWTHATMYNEWLIFEWILLBEANEASYONEINPARADISEDEATH"
    ciphertext, _ = encrypt_homophonic(plaintext)
    
    N_GRAM_SIZE = 4 
    T_START = 15.0
    T_FIN = 0.5
    ITER_EP = 15000
    
    log_probs, floor_penalty = train_language_model(n_gram=N_GRAM_SIZE)
    tests_data = []
    
    print(f"--- Esecuzione TEST CONTINUO 1 (0.99995) in corso...", flush=True)
    h_c1, i_c1 = mcmc_continuous_dynamic(ciphertext, log_probs, floor_penalty, N_GRAM_SIZE, T_START, T_FIN, cooling_rate=0.99995)
    tests_data.append({"hist": h_c1, "iters": i_c1, "label": "Continuo Ottimale ($\lambda=0.99995$)", "color": "#1f77b4", "lw": 1.0})
    
    print(f"--- Esecuzione TEST CONTINUO 2 (0.999) in corso...", flush=True)
    h_c2, i_c2 = mcmc_continuous_dynamic(ciphertext, log_probs, floor_penalty, N_GRAM_SIZE, T_START, T_FIN, cooling_rate=0.999)
    tests_data.append({"hist": h_c2, "iters": i_c2, "label": "Continuo Veloce ($\lambda=0.999$)", "color": "#9467bd", "lw": 1.5})
    
    print(f"--- Esecuzione TEST EPOCHE 1 (0.78) in corso...", flush=True)
    h_e1, i_e1, ep_e1 = mcmc_epoch_dynamic(ciphertext, log_probs, floor_penalty, N_GRAM_SIZE, T_START, T_FIN, cooling_rate=0.78, iter_per_epoch=ITER_EP)
    tests_data.append({"hist": h_e1, "iters": i_e1, "label": f"Epoche ($\lambda=0.78$, {ep_e1} ep)", "color": "#2ca02c", "lw": 1.0})

    print(f"--- Esecuzione TEST EPOCHE 2 (0.85) in corso...", flush=True)
    h_e2, i_e2, ep_e2 = mcmc_epoch_dynamic(ciphertext, log_probs, floor_penalty, N_GRAM_SIZE, T_START, T_FIN, cooling_rate=0.85, iter_per_epoch=ITER_EP)
    tests_data.append({"hist": h_e2, "iters": i_e2, "label": f"Epoche ($\lambda=0.85$, {ep_e2} ep)", "color": "#ff7f0e", "lw": 1.0})

    print(f"--- Esecuzione TEST EPOCHE 3 (0.90) in corso...", flush=True)
    h_e3, i_e3, ep_e3 = mcmc_epoch_dynamic(ciphertext, log_probs, floor_penalty, N_GRAM_SIZE, T_START, T_FIN, cooling_rate=0.90, iter_per_epoch=ITER_EP)
    tests_data.append({"hist": h_e3, "iters": i_e3, "label": f"Epoche ($\lambda=0.90$, {ep_e3} ep)", "color": "#d62728", "lw": 1.0})

    print("\n" + "=" * 50, flush=True)
    print("RISULTATI TERMODINAMICI", flush=True)
    print("=" * 50, flush=True)
    
    for test in tests_data:
        raw_hist = test["hist"]
        max_abs = max(raw_hist)
        
        top_5_count = max(1, len(raw_hist) // 20)
        top_scores = sorted(raw_hist, reverse=True)[:top_5_count]
        media_massimi = sum(top_scores) / len(top_scores)
        
        tail_count = max(1, len(raw_hist) // 10)
        media_plateau = sum(raw_hist[-tail_count:]) / tail_count
        
        print(f"{test['label']}:")
        print(f"  Max Assoluto     : {max_abs:.2f}")
        print(f"  Media Max (5%)   : {media_massimi:.2f}")
        print(f"  Plateau Finale   : {media_plateau:.2f}\n", flush=True)

    max_iters = max(test["iters"] for test in tests_data)
    max_len = max_iters // 100
    x_axis = np.arange(100, max_iters + 1, 100)
    
    plt.figure(figsize=(14, 8))
    for test in tests_data:
        hist = test["hist"]
        if len(hist) < max_len:
            hist.extend([hist[-1]] * (max_len - len(hist)))
        label_with_iters = f"{test['label']} - {test['iters']//1000}k iter"
        plt.plot(x_axis, hist, label=label_with_iters, color=test["color"], linewidth=test["lw"], alpha=0.8)

    plt.title(f"Sismografo Termodinamico a {N_GRAM_SIZE}-Grammi", fontsize=15, fontweight='bold')
    plt.xlabel("Iterazioni Totali", fontsize=12)
    plt.ylabel("Energia (Log-Fitness Corrente)", fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_path = os.path.expanduser('~/grafico_sismografo.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Grafico salvato in: {save_path}", flush=True)