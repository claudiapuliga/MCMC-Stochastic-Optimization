import gensim.downloader as api
import random
import math
import string
import collections
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os

def train_comparative_models_trigrams(l3=0.7, l2=0.2, l1=0.1):
    dataset = api.load("text8")
    uni_counts = defaultdict(int)
    bi_counts = defaultdict(int)
    tri_counts = defaultdict(int)
    alphabet = string.ascii_uppercase
    
    total_unigrams = 0
    prev1 = prev2 = None
    
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
                    prev2, prev1 = prev1, char

    V = 26
    log_probs_laplace = {}
    log_probs_interp = {}
    k = 1.0  
    
    for tri, count in tri_counts.items():
        bi_pre = tri[:-1]
        prob_add = (count + k) / (bi_counts[bi_pre] + k * V)
        log_probs_laplace[tri] = math.log(prob_add)
        
    for a in alphabet:
        for b in alphabet:
            for c in alphabet:
                tri = a + b + c
                bi_pre = a + b
                bi_suf = b + c
                
                p3 = tri_counts[tri] / bi_counts[bi_pre] if bi_counts[bi_pre] > 0 else 0
                p2 = bi_counts[bi_suf] / uni_counts[b] if uni_counts[b] > 0 else 0
                p1 = uni_counts[c] / total_unigrams
                
                prob_interp = (l3 * p3) + (l2 * p2) + (l1 * p1)
                if prob_interp > 0:
                    log_probs_interp[tri] = math.log(prob_interp)
                    
    return log_probs_laplace, log_probs_interp

def calculate_fitness(text, log_probs, fallback_penalty):
    score = 0.0
    for i in range(len(text) - 2):
        score += log_probs.get(text[i:i+3], fallback_penalty)
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
    extra_homophones = {'E': 6, 'T': 4, 'A': 3, 'O': 3, 'I': 3, 'N': 2, 'S': 1, 'R': 1, 'H': 1}
    cipher_dict = {}
    for char in string.ascii_uppercase:
        num_symbols = 1 + extra_homophones.get(char, 0)
        cipher_dict[char] = [pool.pop() for _ in range(num_symbols)]
    return "".join(random.choice(cipher_dict[c]) for c in plaintext)

def apply_key(ciphertext, key):
    return ''.join(key.get(c, c) for c in ciphertext)

def decrypt_mcmc_tracked_epochs(ciphertext, log_probs, plaintext, fallback_penalty, T_start=15.0, T_fin=0.5, cooling_rate=0.90, iter_per_epoch=15000):
    epochs_needed = math.ceil(math.log(T_fin / T_start) / math.log(cooling_rate))
    unique_symbols = list(set(ciphertext))
    alphabet = list(string.ascii_uppercase)
    current_key = {sym: random.choice(alphabet) for sym in unique_symbols}
    best_text = apply_key(ciphertext, current_key)
    current_score = calculate_fitness(best_text, log_probs, fallback_penalty)
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
            new_score = calculate_fitness(new_text, log_probs, fallback_penalty)
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
    
    ciphertext = encrypt_homophonic(plaintext)
    log_probs_laplace, log_probs_interp = train_comparative_models_trigrams()

    runs = 5
    T_START, T_FIN, LAMBDA_COOL, ITER_EP = 15.0, 0.5, 0.90, 15000

    results_laplace = []
    for r in range(runs):
        acc, _ = decrypt_mcmc_tracked_epochs(ciphertext, log_probs_laplace, plaintext, fallback_penalty=-20.0, T_start=T_START, T_fin=T_FIN, cooling_rate=LAMBDA_COOL, iter_per_epoch=ITER_EP)
        results_laplace.append(acc)
    best_laplace = results_laplace[np.argmax([res[-1] for res in results_laplace])]

    results_interp = []
    for r in range(runs):
        acc, _ = decrypt_mcmc_tracked_epochs(ciphertext, log_probs_interp, plaintext, fallback_penalty=-50.0, T_start=T_START, T_fin=T_FIN, cooling_rate=LAMBDA_COOL, iter_per_epoch=ITER_EP)
        results_interp.append(acc)
    best_interp = results_interp[np.argmax([res[-1] for res in results_interp])]

    plt.figure(figsize=(10, 6))
    plt.plot(best_laplace, label="Laplace Smoothing (K=1)", color='blue', linewidth=2)
    plt.plot(best_interp, label="Interpolazione Lineare", color='orange', linewidth=2)
    plt.title("Evoluzione Accuratezza a 3-Grammi")
    plt.xlabel("Epoche")
    plt.ylabel("Accuratezza (%)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('grafico_confronto_modelli_3grammi.png', dpi=300, bbox_inches='tight')