import pandas as pd
import pulp
from math import isclose

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
        print(f'{names[i]} {rows[0][i]:8.4f} {rows[1][i]:8.4f} {rows[2][i]:8.4f} {rows[3][i]:8.4f} {rows[4][i]:8.4f} {rows[5][i]:8.4f} {rows[6][i]:8.4f} {rows[7][i]:8.4f} {rows[8][i]:8.4f} {rows[9][i]:8.4f} {rows[10][i]:8.4f} {cre[i]:8.4f}')

def solve_for_dmu(inputs, outputs, dmu, input_oriented=True):
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

def solve_for_super_eff(inputs, outputs, dmu, eff_value, i_):
    #supereff doesn't change if dmu wasn't effective
    if not isclose(eff_value, 1, rel_tol=1e-9, abs_tol=1e-9):
        print(f'Superefektywność: {eff_value:.3f}')
        return eff_value
    else:
        name = f'dea_{dmu}_{"super_eff"}'
        problem = pulp.LpProblem(name, pulp.LpMaximize)

        # variables
        mi = [pulp.LpVariable(f'mi_{col}', 0) for col in outputs]
        v = [pulp.LpVariable(f'v_{col}', 0) for col in inputs]
        # objective
        objective = pulp.lpDot(mi, outputs.iloc[i_])
        problem += objective

        # constraints
        problem += (pulp.lpDot(v, inputs.iloc[i_])
                    == 1)
        for dmu_ in range(inputs.shape[0]):
            if dmu_ != i_:
                problem += (pulp.lpDot(mi, outputs.iloc[dmu_])
                        <= pulp.lpDot(v, inputs.iloc[dmu_]))

        solver = pulp.PULP_CBC_CMD(msg=False)
        status = problem.solve(solver)
        # print(pulp.LpStatus[status])

        eff_value = 1 / pulp.value(objective)

        print(f'Superefektywność: {eff_value:.3f}')

    return eff_value

def solve_for_cross_eff_row(inputs, outputs, dmu, eff_value, i_):
    name = f'dea_{dmu}_{"cross_eff"}'
    #kind approach
    problem = pulp.LpProblem(name, pulp.LpMaximize)

    # variables
    mi = [pulp.LpVariable(f'mi_{col}', 0) for col in outputs]
    v = [pulp.LpVariable(f'v_{col}', 0) for col in inputs]
    # objective
    objective = 0
    for dmu_ in range(inputs.shape[0]):
        if dmu_ != i_:
            objective += pulp.lpDot(mi, outputs.iloc[dmu_])
    problem += objective

    # constraints
    problem += (pulp.lpDot(v, inputs.iloc[i_])
                == 1)
    for dmu_ in range(inputs.shape[0]):
        if dmu_ != i_:
            problem += (pulp.lpDot(mi, outputs.iloc[dmu_])
                    <= pulp.lpDot(v, inputs.iloc[dmu_]))
    problem += (pulp.lpDot(mi, outputs.iloc[i_])
                == eff_value * pulp.lpDot(v, inputs.iloc[i_]))
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = problem.solve(solver)
    # print(pulp.LpStatus[status])
    effs = []
    for dmu_ in range(inputs.shape[0]):
        if dmu_ != i_:
            ins = sum([pulp.value(v[i])*inputs.iat[dmu_, i] for i in range(inputs.shape[1])])
            outs = sum([pulp.value(mi[i])*outputs.iat[dmu_, i] for i in range(outputs.shape[1])])
            effs.append(outs/ins)
        else:
            effs.append(eff_value)

    return effs

def main():
    inputs = read_csv('inputs.csv')
    outputs = read_csv('outputs.csv')

    super_eff_ranking = {}
    cross_eff_rows = []
    cross_eff_ranking = {}
    cres = []
    for i, city in enumerate(inputs.index):
        eff_value, _ = solve_for_dmu(inputs, outputs, city, input_oriented=True)
        super_eff = solve_for_super_eff(inputs, outputs, city, eff_value, i)
        super_eff_ranking[city] = super_eff
        cross_eff_rows.append(solve_for_cross_eff_row(inputs, outputs, city, eff_value, i))
        print()

    for i, city in enumerate(inputs.index):
        cre = sum([cross_eff_rows[j][i] for j in range(inputs.shape[0])])/inputs.shape[0]
        cross_eff_ranking[city] = cre
        cres.append(cre)

    print_cre_matrix(cross_eff_rows, list(inputs.index), cres)
    super_eff_sorted = dict(sorted(super_eff_ranking.items(), key=lambda item: item[1]))
    cross_eff_sorted = dict(sorted(cross_eff_ranking.items(), key=lambda item: item[1]))

    print("Ranking jednostek dla superefektywności: ")
    print(super_eff_sorted.keys())

    print("Ranking jednostek dla efektywności krzyżowej: ")
    print(cross_eff_sorted.keys())

if __name__ == '__main__':
    main()
