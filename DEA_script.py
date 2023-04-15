import pandas as pd
import pulp


inputs = pd.read_csv('inputs.csv', delimiter=';')
outputs = pd.read_csv('outputs.csv', delimiter=';')

cols_input = list(inputs.columns)
cols_output = list(outputs.columns)

names = inputs[cols_input[0]].tolist()
names_vars = ["v1", "v2", "v3", "v4", "u1", "u2"]
x1 = dict(zip(names, inputs[cols_input[1]]))
x2 = dict(zip(names, inputs[cols_input[2]]))
x3 = dict(zip(names, inputs[cols_input[3]]))
x4 = dict(zip(names, inputs[cols_input[4]]))
y1 = dict(zip(names, outputs[cols_output[1]]))
y2 = dict(zip(names, outputs[cols_output[2]]))


for k in names:
    prob = pulp.LpProblem("DEA", pulp.LpMaximize)
    prob_vars = pulp.LpVariable.dicts("vars", names_vars, lowBound=0, cat='Continuous')
    prob += pulp.lpSum([y1[k]*prob_vars[names_vars[4]], y2[k]*prob_vars[names_vars[5]]])
    prob += pulp.lpSum([x1[k]*prob_vars[names_vars[0]], x2[k]*prob_vars[names_vars[1]], x3[k]*prob_vars[names_vars[2]], x4[k]*prob_vars[names_vars[3]]]) == 1.0
    for k_ in names:
        prob += pulp.lpSum([y1[k_]*prob_vars[names_vars[4]], y2[k_]*prob_vars[names_vars[5]], -1.0*x1[k_]*prob_vars[names_vars[0]], -1.0*x2[k_]*prob_vars[names_vars[1]], -1.0*x3[k_]*prob_vars[names_vars[2]], -1.0*x4[k_]*prob_vars[names_vars[3]]]) <= 0.0

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    eff = prob.variables()[0].varValue * y1[k] + prob.variables()[1].varValue * y2[k]
    if eff >= 1.0:
        effective = "efektywna"
    else:
        effective = "nieefektywna"

    print("{} jest {}".format(k, effective))

    if eff < 1.0:
        #HCU
        n_vars = names[:]
        n_vars.append("O")
        prob1 = pulp.LpProblem("DEA1", pulp.LpMinimize)
        prob1_vars = pulp.LpVariable.dicts("vars1", n_vars, lowBound=0, cat='Continuous')
        prob1 += pulp.lpSum([prob1_vars[n_vars[-1]]])
        for x in [x1, x2, x3, x4]:
            tmp = [x[k_] * prob1_vars[n_vars[num]] for num, k_ in enumerate(names)]
            tmp.append(-1.0 * x[k] * prob1_vars[n_vars[-1]])
            prob1 += pulp.lpSum([tmp]) <= 0.0
        for y in [y1, y2]:
            tmp = [y[k_] * prob1_vars[n_vars[num]] for num, k_ in enumerate(names)]
            tmp.append(-1.0 * y[k])
            prob1 += pulp.lpSum([tmp]) >= 0.0

        prob1.solve(pulp.PULP_CBC_CMD(msg=0))
        print("HCU:")
        for ind, v in enumerate(prob1.variables()):
            if v.varValue > 0 and v.name != 'vars1_O':
                print(v.name, "współczynnik: ", v.varValue)
            elif v.name == 'vars1_O':
                o = v.varValue
        b = x1[k] - x1[k] * o
        a1 = x1[k] - x1[k] * o
        a2 = x2[k] - x2[k] * o
        a3 = x3[k] - x3[k] * o
        a4 = x4[k] - x4[k] * o
        print(
            "{} musi ograniczyć nakłady na wejściu 1 o: {}, na wejściu 2 o: {}, na wejściu 3 o: {}, na wejściu 4 o: {},".format(k, a1, a2, a3, a4))

    print(" ")










