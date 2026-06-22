from metropolis_hasting_sampler import *
import time

DIMACS_PATH  = 'dataset/2.6.28.6-icse11.dimacs'
CONFIGS_DIR  = 'Linux-Kernel-Configuration-Analysis/kernel_configs_files'

REPETICOES = 300
BURNIN     = 10
THINNING   = 10

N_CHAINS = [1,4, 8]
K_FLIPS  = [1, 100, 1000, 400]
ALPHAS   = [0.3, 0.5, 1, 2, 100, 1000]
BETAS    = [1000000, 1000]




def _load_init_configs(variables, configs_dir, n):
    """
    Carrega n configs iniciais a partir dos arquivos reais em configs_dir.

    Cada config é mapeada para o espaço de variáveis do DIMACS:
      - chave no arquivo: CONFIG_USB_SUPPORT  →  variável DIMACS: USB_SUPPORT
      - features do DIMACS ausentes no arquivo ficam como 0

    Se n > número de arquivos disponíveis, faz round-robin entre os arquivos.
    """
    paths   = load_configs_from_dir(configs_dir)
    parsed  = [parse_kernel_config(p) for p in paths]

    configs = []
    for i in range(n):
        cfg  = parsed[i % len(parsed)]
        init = {}
        for var in variables:
            # DIMACS usa nomes sem CONFIG_; busca CONFIG_<var> no arquivo
            val = cfg.get('CONFIG_' + var, cfg.get(var, 0))
            init[var] = 1 if val else 0
        configs.append(init)
    return configs


def run_standard(fm, variables, W, init_configs, alpha, beta, n_chain, k_flip):
    args_list = [
        (seed, DIMACS_PATH, None, W, init_configs[seed],
         alpha, beta, REPETICOES, BURNIN, k_flip, THINNING)
        for seed in range(n_chain)
    ]
    with Pool(n_chain) as pool:
        results = pool.map(run_one_mcmc_chain, args_list)

    avg_accept = sum(r['acceptance_rate'] for r in results) / n_chain
    print(f"  [standard]  amostras={sum(len(r['samples']) for r in results)}"
          f"  accept={avg_accept:.1%}")
    return [s for r in results for s in r['samples']]


def save_samples(all_samples, tag, k_flip, beta, alpha, n_chain):
    if not all_samples:
        return
    df = pd.DataFrame(all_samples).astype(int)
    fname = (f'mcmc_samples_{tag}_K{k_flip}_i{REPETICOES}'
             f'_b{beta}_a{alpha}_t{THINNING}_c{n_chain}')
    df.to_csv(fname, index=False)


if __name__ == "__main__":
    # Carrega configs do diretório
    config_paths = load_configs_from_dir(CONFIGS_DIR)

    # W com chaves no formato DIMACS (sem CONFIG_) para o MCMC
    W_linux = build_W_for_dimacs(config_paths)
    save_W_to_csv(W_linux, "W_linux.csv")
    print(f"Features com peso (DIMACS, sem CONFIG_): {len(W_linux)}")

    fm        = FeatureModel(DIMACS_PATH)
    variables = list(fm.variables.values())
    print(f"Variáveis no DIMACS: {len(variables)}")
    print(f"Interseção W ∩ DIMACS: {len(set(W_linux) & set(variables))}")

    for N_CHAIN in N_CHAINS:
        for K_FLIP in K_FLIPS:
            for ALPHA in ALPHAS:
                for BETA in BETAS:
                    print(f"\n--- chains={N_CHAIN}  k={K_FLIP}  alpha={ALPHA}  beta={BETA} ---")
                    # Configs iniciais a partir dos arquivos reais do Linux-Kernel
                    init_configs = _load_init_configs(variables, CONFIGS_DIR, N_CHAIN)

                    #  run MH padrão 
                    t0 = time.time()
                    samples_std = run_standard(fm, variables, W_linux, init_configs,
                                                ALPHA, BETA, N_CHAIN, K_FLIP)
                    print_tempo(time.time() - t0)
                    save_samples(samples_std, 'standard', K_FLIP, BETA, ALPHA, N_CHAIN)
                                 

