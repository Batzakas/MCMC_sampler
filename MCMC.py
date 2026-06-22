import pandas as pd
import re
import math
import random
from collections import defaultdict
from multiprocessing import Pool
import matplotlib.pyplot as plt




def load_configs_from_dir(directory: str) -> list:
    """
    Retorna a lista de caminhos de todos os arquivos .txt e sem extensão
    encontrados em *directory* que pareçam arquivos .config do kernel.

    Parâmetros
    ----------
    directory : str — caminho para o diretório com os arquivos de configuração

    Retorna
    -------
    list[str] — caminhos absolutos/relativos dos arquivos encontrados
    """
    import os
    import glob

    paths = []
    # Aceita .txt (ex: arch_config_*.txt) e arquivos sem extensão (ex: config.x86_64)
    for pattern in ('*.txt', 'config*', '*.config', '*.cfg'):
        paths.extend(glob.glob(os.path.join(directory, pattern)))

    # Remove duplicatas mantendo a ordem
    seen = set()
    result = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            result.append(p)

    if not result:
        print(f"[AVISO] Nenhum arquivo de configuração encontrado em: {directory}")
    else:
        print(f"[INFO] {len(result)} arquivo(s) de configuração carregado(s) de: {directory}")

    return sorted(result)


def build_weight_vector_from_configs(config_paths: list) -> dict:
    """
    Constrói W a partir de arquivos .config do kernel, sem usar VICs.

    W[feat] = número de configs em que a feature está ativa.
    Features inativas em todas as configs não entram em W.

    Uso típico (sem data leakage):
      configs = load_configs_from_dir("Linux-Kernel-Configuration-Analysis/kernel_configs_files")
      W = build_weight_vector_from_configs(configs)

    Parâmetros
    ----------
    config_paths : lista de caminhos para arquivos .config do kernel

    Retorna
    -------
    dict {CONFIG_X: int}  — contagem de ativações por feature
    """
    from collections import Counter
    W: Counter = Counter()
    for path in config_paths:
        config = parse_kernel_config(path)
        for feat, val in config.items():
            if val and val != 0:
                W[feat] += 1
    return dict(W)


def build_W_for_dimacs(config_paths: list) -> dict:
    """
    Constrói W a partir de arquivos .config do kernel com chaves no formato
    DIMACS (sem prefixo CONFIG_), para casar diretamente com os nomes de
    variáveis do feature model.

    Exemplo: CONFIG_USB_SUPPORT → USB_SUPPORT

    Parâmetros
    ----------
    config_paths : lista de caminhos para arquivos .config do kernel

    Retorna
    -------
    dict {FEATURE_NAME: int}  — contagem de ativações, sem prefixo CONFIG_
    """
    W_raw = build_weight_vector_from_configs(config_paths)
    return {
        (k[7:] if k.startswith('CONFIG_') else k): v
        for k, v in W_raw.items()
    }


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
    Define log π(c) = (+alpha*(W·c) - beta*violations(c)) / temperature

      - features com alto peso W aumentam π  => sampler prefere configs vulneráveis
      - violações de constraints reduzem π   => sampler evita configs inválidas
      - temperature > 1 achata a distribuição (exploração — usado em Parallel Tempering)
      - temperature = 1 reproduz o comportamento original

    em que,
      alpha       : peso do score de vulnerabilidade
      beta        : peso da penalidade por violações de constraints
      temperature : fator de temperatura (T=1 padrão; T>1 para réplicas quentes em PT)
    """

    def __init__(self, feature_model, vulnerabilities, alpha=1.0, beta=1.0, temperature=1.0):
        self.fm = feature_model
        self.w = vulnerabilities
        self.alpha = alpha
        self.beta  = beta
        self.temperature = temperature

    def raw_energy(self, config):
        """Energia sem fator de temperatura (alpha*(W·c) - beta*violations)."""
        e_vun = sum(self.w.get(f, 0) for f, v in config.items() if v == 1)
        e_fm  = self.fm.count_violations_cached(config)
        return self.alpha * e_vun - self.beta * e_fm

    def log_prob(self, config):
        return self.raw_energy(config) / self.temperature
        

class Metropolis:
    """
    Implementa o algoritmo Metropolis-Hastings sobre o espaço de configurações.

    A cadeia parte de init_config e propõe novos estados via k flips aleatórios. 

    A distribuição estacionária é proporcional a exp(log_prob(c)).
    """

    def __init__(self, init_config, options, thinning, energy_model):
        self.options = options
        self.thinning = thinning
        self.energy_model = energy_model
        self.bin_vec = {opt: init_config.get(opt, 0) for opt in self.options}
        # Features com maior |W| têm mais chance de serem flippadas na proposta
        eps = 1e-3
        self.flip_weights = [abs(energy_model.w.get(opt, 0)) + eps for opt in self.options]

    def flip(self, estado):
        op = random.choices(self.options, weights=self.flip_weights, k=1)[0]
        novo = estado.copy()
        novo[op] = 1 - novo[op]
        return novo

    def k_flips(self, estado, k):
        novo = estado.copy()
        for op in random.choices(self.options, weights=self.flip_weights, k=k):
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

            if i >= burnin and (i - burnin) % self.thinning == 0:
                estados.append(estado_atual.copy())
        taxa_aceitacao = n_aceitos / repeticoes
        return estados, taxa_aceitacao

def run_one_mcmc_chain(args):
    """
    Executa uma cadeia MCMC independente.
    Recebe (seed, dimacs_path, config_path, W, alpha, beta, k_penalty, repeticoes, burnin, k_flips)
    recebe diferentes estados iniciais
    """
    (seed, dimacs_path, config_path, W,init_config, alpha, beta, repeticoes, burnin, k, thinning) = args

    random.seed(seed)


    fm = FeatureModel(dimacs_path)

    em = EnergyModel(fm, W, alpha=alpha, beta=beta)
    sampler = Metropolis(init_config, list(fm.variables.values()),thinning, energy_model=em)

    samples, acceptance_rate = sampler.metropolis_hasting(
        repeticoes=repeticoes, burnin=burnin, k=k
    )

    return {
        'samples': samples,
        'acceptance_rate': acceptance_rate,
        'seed': seed,
    }



def print_tempo(elapsed_time):
    horas = int(elapsed_time//3600)
    minutos = int((elapsed_time%3600)//60)
    segundos =int(elapsed_time%60)

    print(f'Tempo Total: {horas} h, {minutos}m, {segundos}s')

