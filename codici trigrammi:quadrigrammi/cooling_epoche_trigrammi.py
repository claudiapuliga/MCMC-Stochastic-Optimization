import random
import math
import string
import collections
import matplotlib.pyplot as plt
import numpy as np
import os
import gensim.downloader as api

def train_language_model(n_gram, k_smoothing=1.0):
    dataset = api.load("text8")
    ngram_counts = collections.defaultdict(int)
    prefix_counts = collections.defaultdict(int)
    alphabet = set(string.ascii_uppercase)
    buffer = []
    
    for words in dataset:
        for word in words:
            word = word.upper()
            for char in word:
                if char in alphabet:
                    buffer.append(char)
                    if len(buffer) == n_gram:
                        ngram = "".join(buffer)
                        prefix = ngram[:-1]
                        ngram_counts[ngram] += 1
                        prefix_counts[prefix] += 1
                        buffer.pop(0)
                        
    V = 26 
    log_probs = {}
    for ngram, count in ngram_counts.items():
        prefix = ngram[:-1]
        prob = (count + k_smoothing) / (prefix_counts[prefix] + k_smoothing * V)
        log_probs[ngram] = math.log(prob)
    return log_probs

def calculate_fitness(text, log_probs, n_gram):
    score = 0.0
    penalty = -15.0 if n_gram >= 4 else -20.0
    for i in range(len(text) - (n_gram - 1)):
        chunk = text[i:i+n_gram]
        score += log_probs.get(chunk, penalty)
    return score

def encrypt_homophonic(plaintext):
    symbol_string = "Δ∇⊕⊗★☾☀♈♉♊♋♌♍♎♏♐♑♒♓♠♣♥♦†‡✓✗△▽◁▷ΩΣΦΨΠΓΘΞΛΥαβγδεζηθικλμνξοπρστυφχψω"

    pool = list(symbol_string)
    random.shuffle(pool)

    extra_homophones = {'E': 6, 'T': 4, 'A': 3, 'O': 3, 'I': 3, 'N': 2, 'S': 1, 'R': 1, 'H': 1}

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

def mcmc_epoch_dynamic(ciphertext, log_probs, n_gram, T_start, T_fin, cooling_rate, iter_per_epoch):
    epochs_needed = math.ceil(math.log(T_fin / T_start) / math.log(cooling_rate))
    total_iterations = epochs_needed * iter_per_epoch
    unique_symbols = list(set(ciphertext))
    alphabet = list(string.ascii_uppercase)
    current_key = {sym: random.choice(alphabet) for sym in unique_symbols}
    current_score = calculate_fitness(apply_key(ciphertext, current_key), log_probs, n_gram)
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
                new_score = calculate_fitness(apply_key(ciphertext, current_key), log_probs, n_gram)
                delta_S = new_score - current_score
                alpha = 1.0 if delta_S > 0 else math.exp(delta_S / T)
                if random.random() < alpha:
                    current_score = new_score
                else:
                    current_key[sym] = old_val
            global_iter += 1
            if global_iter % 100 == 0: 
                history.append(current_score)
        T *= cooling_rate
    return history, total_iterations, epochs_needed

def mcmc_continuous_dynamic(ciphertext, log_probs, n_gram, T_start, T_fin, cooling_rate):
    total_iterations = math.ceil(math.log(T_fin / T_start) / math.log(cooling_rate))
    unique_symbols = list(set(ciphertext))
    alphabet = list(string.ascii_uppercase)
    current_key = {sym: random.choice(alphabet) for sym in unique_symbols}
    current_score = calculate_fitness(apply_key(ciphertext, current_key), log_probs, n_gram)

    T = T_start
    history = []
    for i in range(total_iterations):
        sym = random.choice(unique_symbols)
        old_val = current_key[sym]
        new_val = random.choice(alphabet)
        if old_val != new_val:
            current_key[sym] = new_val
            new_score = calculate_fitness(apply_key(ciphertext, current_key), log_probs, n_gram)
            delta_S = new_score - current_score
            alpha = 1.0 if delta_S > 0 else math.exp(delta_S / T)
            if random.random() < alpha:
                current_score = new_score
            else:
                current_key[sym] = old_val
        T *= cooling_rate
        if (i + 1) % 100 == 0: 
            history.append(current_score)
    return history, total_iterations

if __name__ == "__main__":
    plaintext = (
        "IHOPEYOUAREHAVINGLOTSOFFUNINTRYINGTOCATCHMETHATWA"
        "SNTMEONTHETVSHOWWHICHBRINGSUPAPOINTABOUTMEIAMNOTAFRAI"
        "DOFTHEGASCHAMBERBECAUSEITWILLSENDMETOPARADISEALLTHESO"
        "ONERBECAUSEINOWHAVEENOUGHSLAVESTOWORKFORMEWHEREASEEVE"
        "RYONEELSEHASNOTHINGWHENTHEYREACHPARADISESOTHEYAREAFRA"
        "IDOFDEATHIAMNOTAFRAIDBECAUSEIKNOWTHATMYNEWLIFEWILLBEA"
        "NEASYONEINPARADISEDEATH"
    )
    ciphertext, _ = encrypt_homophonic(plaintext)
    
    N_GRAM_SIZE = 3 
    T_START = 15.0
    T_FIN = 0.5
    ITER_EP = 15000
    
    log_probs = train_language_model(n_gram=N_GRAM_SIZE)
    tests_data = []
    
    h_c1, i_c1 = mcmc_continuous_dynamic(ciphertext, log_probs, N_GRAM_SIZE, T_START, T_FIN, cooling_rate=0.99995)
    tests_data.append({"hist": h_c1, "iters": i_c1, "label": "Continuo Ottimale ($\lambda=0.99995$)", "color": "#1f77b4", "lw": 1.0})
    
    h_e1, i_e1, ep_e1 = mcmc_epoch_dynamic(ciphertext, log_probs, N_GRAM_SIZE, T_START, T_FIN, cooling_rate=0.78, iter_per_epoch=ITER_EP)
    tests_data.append({"hist": h_e1, "iters": i_e1, "label": "Epoche Ottimale ($\lambda=0.78$)", "color": "#2ca02c", "lw": 1.0})

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
    plt.savefig('grafico_sismografo.png', dpi=300, bbox_inches='tight')