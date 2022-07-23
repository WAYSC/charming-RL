def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    for i in range(env.row):
        for j in range(env.col):
            if (i * env.col + j) in disaster:
                print('****', end=' ')
            elif (i * env.col + j) in end:
                print('EEEE', end=' ')
            else:
                a = agent.best_action(i * env.col + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()