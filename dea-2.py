from math import isclose
import numpy as np
import pandas as pd
import pulp

def read_csv(filepath):
    return pd.read_csv(filepath, sep=';', index_col=0)

def get_hcu(inputs_or_outputs, weights):
    hcu = {}
    for col in inputs_or_outputs:
        hcu_value = 0
        for i, value in enumerate(inputs_or_outputs[col]):
            hcu_value += value * pulp.value(weights[i])
        hcu[col] = hcu_value
    return hcu

def print_difference(dmu1, dmu2):
    print('        DMU      HCU     DIFF')
    for col in dmu1.index:
        print(f'{col} {dmu1[col]:8.4f} {dmu2[col]:8.4f} {dmu2[col] - dmu1[col]:8.4f}')

def print_cre_matrix(rows, names, cre):
    header = "      "
    for n in names:
        header += n + "      "
    header += "CRE"
    print(header)
    for i in range(len(names)):
        print(f'{names[i]} {" ".join([f"{x:8.4f}" for x in rows[i]])} {cre[i]:8.4f}')
    print()

def print_efficiency_histogram(hist, means, names):
    print('     [0.0,0.2] (0.2,0.4] (0.4,0.6] (0.6,0.8] (0.8,1.0]     MEAN')
    for i, name in enumerate(names):
        print(f'{name} {" ".join([f"{x:9.0f}" for x in hist[i]])} {means[i]:9.4f}')
    print()

def get_efficiency(inputs, outputs, input_weights, output_weights):
    inp = inputs.dot(input_weights)
    out = outputs.dot(output_weights)
    return out / inp

def solve_dea_combinations(inputs, outputs, dmu, input_oriented=True):
    """
        Solve dea in dmu combinations space.
        Return tuple: efficiency, dmu weights
    """
    name = f'dea_{dmu}_{"input" if input_oriented else "ouput"}'
    problem = pulp.LpProblem(name, pulp.LpMinimize if input_oriented else pulp.LpMaximize)

    # variables
    efficiency = pulp.LpVariable(f'theta_{dmu}', 0)
    weights = [pulp.LpVariable(f'lambda_{i}', 0) for i in range(inputs.shape[0])]

    # objective
    problem += efficiency

    # constraints
    for col in inputs:
        problem += (pulp.lpDot(list(inputs[col]), weights) 
            <= inputs[col][dmu] * (efficiency if input_oriented else 1))

    for col in outputs:
        problem += (pulp.lpDot(list(outputs[col]), weights)
            >= outputs[col][dmu] * (1 if input_oriented else efficiency))

    solver = pulp.PULP_CBC_CMD(msg=False)
    status = problem.solve(solver)

    # print(pulp.LpStatus[status])

    eff_value = pulp.value(efficiency) if input_oriented else 1 / pulp.value(efficiency)

    if not isclose(eff_value, 1, rel_tol=1e-9, abs_tol=1e-9):
        print(f'{dmu} nie jest efektywna')
        print(f'Efektywność: {eff_value:.3f}')

        print('Wagi:')
        for i, city in enumerate(inputs.index):
            weight = pulp.value(weights[i])
            if not isclose(weight, 0, rel_tol=1e-9, abs_tol=1e-9):
                print(city, pulp.value(weights[i]))

        hcu = get_hcu(inputs, weights) if input_oriented else get_hcu(outputs, weights)

        if input_oriented:
            print_difference(inputs.loc[dmu], hcu)
        else:
            print_difference(outputs.loc[dmu], hcu)
    else:
        print(f'{dmu} jest efektywna')
        print(f'Efektywność: {eff_value:.3f}')
    
    return eff_value, weights

def solve_dea_efficiency(inputs, outputs, dmu, super_eff=False):
    """
        Solve dea in efficiency space, input oriented.
        Returns tuple: efficiency value, output weights, input weights
    """

    name = f'dea_{dmu}'
    problem = pulp.LpProblem(name, pulp.LpMaximize)

    # variables
    mi = [pulp.LpVariable(f'mi_{col}', 0) for col in outputs]
    v = [pulp.LpVariable(f'v_{col}', 0) for col in inputs]

    # objective
    objective = pulp.lpDot(mi, outputs.loc[dmu])
    problem += objective

    # constraints
    problem += (pulp.lpDot(v, inputs.loc[dmu]) == 1)
    for dmu_i in inputs.index:
        if super_eff and dmu == dmu_i:
            continue

        problem += (pulp.lpDot(mi, outputs.loc[dmu_i])
                <= pulp.lpDot(v, inputs.loc[dmu_i]))

    solver = pulp.PULP_CBC_CMD(msg=False)
    status = problem.solve(solver)
    # print(pulp.LpStatus[status])

    eff_value = pulp.value(objective)

    return eff_value, mi, v

def solve_for_super_eff(inputs, outputs, dmu, eff_value):
    # supereff doesn't change if dmu wasn't effective
    if not isclose(eff_value, 1, rel_tol=1e-9, abs_tol=1e-9):
        print(f'Superefektywność: {eff_value:.3f}')
        return eff_value
    else:
        eff_value, _, _ = solve_dea_efficiency(inputs, outputs, dmu, super_eff=True)
        print(f'Superefektywność: {eff_value:.3f}')

    return eff_value

def solve_for_cross_eff_row(inputs, outputs, dmu, eff_value):
    name = f'dea_{dmu}_{"cross_eff"}'
    # kind approach
    problem = pulp.LpProblem(name, pulp.LpMaximize)

    # variables
    mi = [pulp.LpVariable(f'mi_{col}', 0) for col in outputs]
    v = [pulp.LpVariable(f'v_{col}', 0) for col in inputs]

    # objective
    objective = 0
    for dmu_i in inputs.index:
        if dmu_i != dmu:
            objective += pulp.lpDot(mi, outputs.loc[dmu_i])

    problem += objective

    # constraints
    problem += (pulp.lpDot(v, inputs.loc[dmu]) == 1)
    for dmu_i in inputs.index:
        if dmu_i != dmu:
            problem += (pulp.lpDot(mi, outputs.loc[dmu_i])
                    <= pulp.lpDot(v, inputs.loc[dmu_i]))

    problem += (pulp.lpDot(mi, outputs.loc[dmu])
                == eff_value * pulp.lpDot(v, inputs.loc[dmu]))

    solver = pulp.PULP_CBC_CMD(msg=False)
    status = problem.solve(solver)
    # print(pulp.LpStatus[status])

    effs = []
    mi_values = list(map(pulp.value, mi))
    v_values = list(map(pulp.value, v))
    for i, dmu_i in enumerate(inputs.index):
        if dmu_i != dmu:
            eff = get_efficiency(inputs.loc[dmu_i], outputs.loc[dmu_i], v_values, mi_values)
            effs.append(eff)
        else:
            effs.append(eff_value)

    return effs

def get_expected_efficiency(inputs, outputs, samples):
    n_inp = inputs.shape[1]
    n_out = outputs.shape[1]
    efficiences = np.zeros(shape=(samples.shape[0], inputs.shape[0]))
    for i, sample in enumerate(samples.index):
        input_weights = samples.iloc[i, :n_inp]
        output_weights = samples.iloc[i, n_inp:n_inp + n_out]
        for j, dmu in enumerate(inputs.index):
            efficiences[i][j] = get_efficiency(inputs.loc[dmu], outputs.loc[dmu], input_weights, output_weights)
    
    efficiences /= np.max(efficiences, axis=1, keepdims=True)

    eff_hist = np.zeros(shape=(inputs.shape[0], 5))
    for i in range(inputs.shape[0]):
        hist, _ = np.histogram(efficiences[:, i], bins=5, range=(0, 1))
        eff_hist[i, :] = hist
    efficiences = np.mean(efficiences, axis=0)

    print_efficiency_histogram(eff_hist, efficiences, inputs.index)

    return dict(zip(inputs.index, efficiences))


def main():
    inputs = read_csv('inputs.csv')
    outputs = read_csv('outputs.csv')

    super_eff_ranking = {}
    cross_eff_matrix = np.zeros(shape=(inputs.shape[0], inputs.shape[0]))
    cross_eff_ranking = {}
    cres = []
    for i, city in enumerate(inputs.index):
        eff_value, _ = solve_dea_combinations(inputs, outputs, city, input_oriented=True)
        super_eff = solve_for_super_eff(inputs, outputs, city, eff_value)
        super_eff_ranking[city] = super_eff
        cross_eff_matrix[:, i] = solve_for_cross_eff_row(inputs, outputs, city, eff_value)
        print()

    for i, city in enumerate(inputs.index):
        cre = np.sum(cross_eff_matrix[i, :]) / inputs.shape[0]
        cross_eff_ranking[city] = cre
        cres.append(cre)

    print_cre_matrix(cross_eff_matrix, list(inputs.index), cres)

    samples = read_csv('samples_homework.csv')
    expected_eff = get_expected_efficiency(inputs, outputs, samples)

    super_eff_sorted = sorted(super_eff_ranking.items(), key=lambda item: item[1], reverse=True)
    cross_eff_sorted = sorted(cross_eff_ranking.items(), key=lambda item: item[1], reverse=True)
    expected_eff_sorted = sorted(expected_eff.items(), key=lambda item: item[1], reverse=True)

    print("Ranking jednostek dla superefektywności:")
    print([f'{city}-{eff:.4f}' for city, eff in super_eff_sorted])

    print("Ranking jednostek dla efektywności krzyżowej:")
    print([f'{city}-{eff:.4f}' for city, eff in cross_eff_sorted])

    print("Ranking jednostek dla oczekiwanej efektywności:")
    print([f'{city}-{eff:.4f}' for city, eff in expected_eff_sorted])

if __name__ == '__main__':
    main()
