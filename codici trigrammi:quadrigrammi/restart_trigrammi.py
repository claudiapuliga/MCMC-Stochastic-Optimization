import gensim.downloader as api
import random
import math
import string
import collections
from collections import defaultdict
import matplotlib.pyplot as plt

# ============================================================
# FASE 1: MODELLO LINGUISTICO A TRIGRAMMI
# ============================================================
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

# ============================================================
# FASE 2: FUNZIONE FITNESS CON ENTROPIA
# ============================================================
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

# ============================================================
# FASE 3: CIFRARIO OMOFONICO
# ============================================================
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

# ============================================================
# FASE 4: METROPOLIS-HASTINGS E SIMULATED ANNEALING
# ============================================================
def decrypt_mcmc_sa_epoch(ciphertext, log_probs, T_start=15.0, T_fin=0.5, cooling_rate=0.90, iter_per_epoch=15000):
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
    
    for epoch in range(epochs_needed):
        for _ in range(iter_per_epoch):
            sym = random.choice(unique_symbols)
            old_val = current_key[sym]
            new_val = random.choice(alphabet)
            if old_val == new_val: continue
            current_key[sym] = new_val
            new_text = apply_key(ciphertext, current_key)
            new_score = calculate_fitness(new_text, log_probs)
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
    return best_text, best_score, history, epochs_needed

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
    ciphertext, real_key = encrypt_homophonic(plaintext)
    log_probs = train_language_model(k_smoothing=1.0)
    
    restarts = 10
    global_best_score = float('-inf')
    global_best_text = ""
    current_best_acc = 0.0
    
    x_axis = list(range(1, restarts + 1))
    y_fitness_cumulativa = []
    y_accuratezza_correlata = []
    
    for run in range(restarts):
        text, score, _, epoche_usate = decrypt_mcmc_sa_epoch(
            ciphertext, log_probs, T_start=15.0, T_fin=0.5, cooling_rate=0.90, iter_per_epoch=15000
        )
        matches_run = sum(1 for a, b in zip(plaintext, text) if a == b)
        acc_run = (matches_run / len(plaintext)) * 100
        
        if score > global_best_score:
            global_best_score = score
            global_best_text = text
            current_best_acc = acc_run
            
        y_fitness_cumulativa.append(global_best_score)
        y_accuratezza_correlata.append(current_best_acc)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color1 = 'tab:blue'
    ax1.set_xlabel('Numero di Riavvii (Restarts)')
    ax1.set_ylabel('Log-Fitness Massima Cumulativa', color=color1)
    ax1.step(x_axis, y_fitness_cumulativa, where='post', color=color1, linewidth=2.5, marker='o')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x_axis)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2 = ax1.twinx()  
    color2 = 'tab:green'
    ax2.set_ylabel('Accuratezza (%) associata al Best Score', color=color2)
    ax2.step(x_axis, y_accuratezza_correlata, where='post', color=color2, linewidth=2.5, linestyle='--', marker='x')
    ax2.tick_params(axis='y', labelcolor=color2)
    plt.title("Impatto dei Riavvii (10 Restarts)")
    plt.savefig('grafico_esperimento_restarts.png', dpi=300, bbox_inches='tight')