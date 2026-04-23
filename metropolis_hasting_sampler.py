import pandas as pd
import re
import math
import random
from collections import defaultdict
from multiprocessing import Pool
import matplotlib.pyplot as plt


def build_weight_vector(csv_path, target_product):
    """
    Percorre o dataset de vulnerabilidades e constrói W:
      +1 se a feature é requerida (defined) em uma flag vulnerável
      -1 se a feature é explicitamente não requerida
    """
    df = pd.read_csv(csv_path)
    df = df[df["product"] == target_product]

    W = defaultdict(int)

    defined_re = re.compile(r"defined\s*\(\s*(CONFIG_[A-Z0-9_]+)\s*\)")
    not_defined_re = re.compile(r"\(~\s*\(\s*\$\(\s*(CONFIG_[A-Z0-9_]+)\s*\)\s*=\s*\)\s*\)")

    for flag_expr in df["flag"].dropna():
        for feat in defined_re.findall(flag_expr):
            W[feat] += 1
        for feat in not_defined_re.findall(flag_expr):
            W[feat] -= 1

    return dict(W)


def save_W_to_csv(W, output_path):
    df_out = pd.DataFrame(
        [(feat, weight) for feat, weight in W.items()],
        columns=["feature", "weight"]
    )
    df_out.to_csv(output_path, index=False)


def parse_kernel_config(file_path):
    """/
    Lê um arquivo .config do kernel e retorna dicionário {CONFIG_X: 0 ou 1}.
    """
    configs = {}
    try:
        with open(file_path, 'r') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Erro: arquivo {file_path} não encontrado.")
        return {}

    assignment_re = re.compile(r'^(CONFIG_\w+)=(.*)')
    is_not_set_re = re.compile(r'^#\s+(CONFIG_\w+)\s+is\s+not\s+set')

    for line in text.splitlines():
        line = line.strip()

        m = is_not_set_re.match(line)
        if m:
            configs[m.group(1)] = 0
            continue

        if not line or line.startswith('#'):
            continue

        m = assignment_re.match(line)
        if m:
            name, value = m.groups()
            value = value.strip('"')
            if value in ('y', 'm'):
                configs[name] = 1
            elif value == 'n':
                configs[name] = 0
            elif value.isdigit():
                configs[name] = int(value)
            else:
                configs[name] = 1 if value else 0

    return configs


class FeatureModel:

    def __init__(self, dimacs_path):
        self.variables = {}
        self.clauses= []   
        self._cache = {}   
        self._parse_dimacs(dimacs_path)

    def _parse_dimacs(self, path):
        mapping_re = re.compile(r'^c\s+(\d+)\s+(\w+)')
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('c '):
                    m = mapping_re.match(line)
                    if m:
                        var_id, name = m.groups()
                        self.variables[int(var_id)] = name
                    continue
                if line.startswith('p cnf'):
                    continue
                parts = [int(x) for x in line.split()]
                if parts and parts[-1] == 0:
                    parts.pop()
                    if parts:
                        self.clauses.append(parts)

    def count_violations(self, config_dict):
        violations = 0
        for clause in self.clauses:
            satisfied = False
            for literal in clause:
                var_id = abs(literal)
                feature_name = self.variables.get(var_id)
                val = config_dict.get(feature_name, 0)
                if (literal > 0 and val == 1) or (literal < 0 and val == 0):
                    satisfied = True
                    break
            if not satisfied:
                violations += 1
        return violations
    # Tentei com cache mas ainda continua lento
    def count_violations_cached(self, config_dict):
        key = tuple(sorted(config_dict.items()))
        if key not in self._cache:
            self._cache[key] = self.count_violations(config_dict)
        return self._cache[key]


class EnergyModel:
    """
    Define log π(c) = -alpha*(W·c) -beta*violations(c)

    Sinal negativo em ambos os termos:
      - features com alto peso W reduzem π  => sampler evita configs vulneráveis
      - violações de constraints reduzem π  => sampler evita configs inválidas

    em que, 
      alpha : peso do score de vulnerabilidade
      beta : peso da penalidade por violações de constraints
    """

    def __init__(self, feature_model, vulnerabilities, alpha=1.0, beta=1.0):
        self.fm = feature_model
        self.w = vulnerabilities
        self.alpha = alpha
        self.beta  = beta

    def log_prob(self, config):
        e_vun = sum(self.w.get(f, 0) for f, v in config.items() if v == 1)
        e_fm  = self.fm.count_violations_cached(config)
        prob = -self.alpha * e_vun - self.beta * e_fm
        return prob 
        

class Metropolis:
    """
    Implementa o algoritmo Metropolis-Hastings sobre o espaço de configurações.

    A cadeia parte de init_config e propõe novos estados via k flips aleatórios. 

    A distribuição estacionária é proporcional a exp(log_prob(c)).
    """

    def __init__(self, init_config, options, energy_model):
        self.options = options
        self.energy_model = energy_model
        self.bin_vec = {opt: init_config.get(opt, 0) for opt in self.options}

    def flip(self, estado):
        op = random.choice(self.options)
        novo = estado.copy()
        novo[op] = 1 - novo[op]
        return novo

    def k_flips(self, estado, k):
        novo = estado.copy()
        for op in random.choices(self.options, k=k):
            novo[op] = 1 - novo[op]
        return novo

 
    def metropolis_hasting(self, repeticoes, burnin, k=1):
        """
        Executa a cadeia MH por "repeticoes" passos.

        Retorna:
          estados : lista de configurações amostradas (após burn-in)
          taxa_aceitacao : fração de propostas aceitas ao longo de toda a cadeia

        """
        estados = []
        estado_atual = self.bin_vec.copy()
        n_aceitos = 0

        log_p_atual = self.energy_model.log_prob(estado_atual)

        for i in range(repeticoes):

            novo_estado = self.k_flips(estado_atual, k) if k > 1 else self.flip(estado_atual)

            log_p_novo = self.energy_model.log_prob(novo_estado)
            log_alpha = log_p_novo - log_p_atual  # = log(π(c*)/π(c))

            if math.log(random.random()) < log_alpha:
                estado_atual = novo_estado
                log_p_atual = log_p_novo
                n_aceitos += 1

            if i >= burnin:
                estados.append(estado_atual.copy())

        taxa_aceitacao = n_aceitos / repeticoes
        return estados, taxa_aceitacao   


def run_one_mcmc_chain(args):
    """
    Executa uma cadeia MCMC independente.
    Recebe (seed, dimacs_path, config_path, W, alpha, beta, k_penalty, repeticoes, burnin, k_flips)
    como tupla para compatibilidade com Pool.map.
    """
    (seed, dimacs_path, config_path, W, alpha, beta, repeticoes, burnin, k) = args

    random.seed(seed)

    fm = FeatureModel(dimacs_path)
    config_dict = parse_kernel_config(config_path)

    init_config = {f: config_dict.get(f, 0) for f in fm.variables.values()}

    em = EnergyModel(fm, W, alpha=alpha, beta=beta)
    sampler = Metropolis(init_config, list(fm.variables.values()), energy_model=em)

    samples, acceptance_rate = sampler.metropolis_hasting(
        repeticoes=repeticoes, burnin=burnin, k=k
    )

    return {
        'samples': samples,
        'acceptance_rate': acceptance_rate,
        'seed': seed,
    }



def run_diagnostics(samples, fm, em, options):

    energies = [em.log_prob(s) for s in samples]
    violations_per_sample = [fm.count_violations(s) for s in samples]
    valid_count = sum(1 for v in violations_per_sample if v == 0)

    feature_usage = defaultdict(int)
    for state in samples:
        for feat, val in state.items():
            if val == 1:
                feature_usage[feat] += 1

    print("\n" + "=" * 70)
    print("DIAGNÓSTICOS")
    print("=" * 70)

    print(f"\n[Qualidade das amostras]")
    print(f"Total de amostras: {len(samples)}")
    print(f"Amostras válidas (0 violações): {valid_count} ({valid_count/len(samples)*100:.1f}%)")
    print(f"Log-prob médio: {sum(energies)/len(energies):.4f}")
    print(f"Log-prob mínimo: {min(energies):.4f}")
    print(f"Log-prob máximo: {max(energies):.4f}")

    unique_states = len(set(tuple(sorted(s.items())) for s in samples))
    print(f"\n[Diversidade]")
    print(f"Estados únicos : {unique_states} / {len(samples)}")
    if unique_states > len(samples) * 0.7:
        print("Cadeia explorando bem")
    else:
        print("Cadeia pode estar presa")

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Diagnósticos MCMC — Feature Model', fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(energies, linewidth=0.8, alpha=0.7)
    ax.axhline(y=sum(energies)/len(energies), color='red', linestyle='--', label='Média')
    ax.set_xlabel('Índice da amostra')
    ax.set_ylabel('Log-probabilidade')
    ax.set_title('Trace Plot (log π)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.hist(energies, bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Log-probabilidade')
    ax.set_ylabel('Contagem')
    ax.set_title('Distribuição de log π')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 0]
    feats_short = [f.replace('CONFIG_', '') for f in sorted(options)]
    usage_vals  = [feature_usage[f] / len(samples) * 100 for f in sorted(options)]
    ax.barh(feats_short, usage_vals, color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Frequência de ativação (%)')
    ax.set_title('Frequência de ativação por feature')
    ax.grid(True, alpha=0.3, axis='x')

    ax = axes[1, 1]
    max_v = max(violations_per_sample) if violations_per_sample else 1
    ax.hist(violations_per_sample, bins=range(0, max_v + 2),
            edgecolor='black', alpha=0.7, align='left')
    ax.set_xlabel('Violações de constraints')
    ax.set_ylabel('Contagem')
    ax.set_title('Violações por amostra')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('mcmc_diagnostics.png', dpi=100, bbox_inches='tight')
    print("\nPlot salvo: mcmc_diagnostics.png")



if __name__ == "__main__":

    DIMACS_PATH  = '2.6.28.6-icse11.dimacs'
    CONFIG_PATH  = 'config.x86_64'
    DATASET_PATH = 'final_dataset.csv'
    PRODUCT = 'linux/linux_kernel'

    N_CHAINS = 4
    REPETICOES  = 5000
    BURNIN = 500
    K_FLIPS = 1      # inicialmente 1 flip 
    ALPHA = 1.0    # peso do score de vulnerabilidade
    BETA = 50.0    # peso da penalidade por violações

    W_linux = build_weight_vector(DATASET_PATH, PRODUCT)
    save_W_to_csv(W_linux, "W_linux.csv")
    print(f"Features com peso no dataset: {len(W_linux)}")

    config_dict = parse_kernel_config(CONFIG_PATH)
    print(f"Features no arquivo de config: {len(config_dict)}")

    # Roda n cadeias em paralelo 
    args_list = [
        (seed, DIMACS_PATH, CONFIG_PATH, W_linux,
         ALPHA, BETA, REPETICOES, BURNIN, K_FLIPS)
        for seed in range(N_CHAINS)
    ]

    with Pool(N_CHAINS) as pool:
        results = pool.map(run_one_mcmc_chain, args_list)

    print(f"\nChains concluídas")
    print(f"Total de amostras: {sum(len(r['samples']) for r in results)}")
    print(f"Taxa de aceitação média: {sum(r['acceptance_rate'] for r in results) / N_CHAINS:.1%}")

    # Combina amostras de todas as cadeias
    all_samples = [s for r in results for s in r['samples']]

    fm = FeatureModel(DIMACS_PATH)
    em = EnergyModel(fm, W_linux, alpha=ALPHA, beta=BETA)

    init_config = {f: config_dict.get(f, 0) for f in fm.variables.values()}
    options = list(fm.variables.values())

    run_diagnostics(all_samples, fm, em, options)

    df_samples = pd.DataFrame(all_samples)
    df_samples.to_csv('mcmc_samples.csv', index=False)
