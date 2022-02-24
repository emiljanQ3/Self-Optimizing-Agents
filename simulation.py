from init import init_agents_data, init_agents_pos
from tqdm import tqdm


def simulate(world, mover, data_recorder, data_modifier, params, models=None):
    for rep in tqdm(range(params.num_repeats), leave=False):
        agents_pos = init_agents_pos(params)
        agents_data = init_agents_data(params, models)
        for step in tqdm(range(params.num_steps), leave=False):
            new_agents_pos, agents_data = mover.step(agents_pos, agents_data, params)
            new_agents_pos = world.step(agents_pos, new_agents_pos, params)
            data_recorder.record(agents_pos, new_agents_pos, agents_data, world, mover, rep, step)
            data_modifier.modify(agents_data, params)
            agents_pos = new_agents_pos
            
        data_recorder.new_repeat()
        data_modifier.new_repeat(agents_data, params)

    return data_recorder.get_results()
