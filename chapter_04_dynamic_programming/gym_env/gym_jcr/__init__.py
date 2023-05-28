from gymnasium.envs.registration import register

environments = [['JacksCarRental', 'v0']]

for name, version in environments:
    register(
        id=f'{name}-{version}',
        entry_point=f'gym_jcr.jcr_env:{name}Env'
    )